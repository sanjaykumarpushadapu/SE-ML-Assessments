"""
build_report.py - refresh the report from the latest notebook run and make 13.pdf.

Run it after the notebook has been run from top to bottom (Restart + Run All) and saved:

    python build_report.py            # update 13.md values, retake screenshots, make 13.pdf
    python build_report.py --no-shots # same, but keep the existing screenshots
    python build_report.py --no-pdf   # update 13.md and screenshots, skip 13.pdf

Needs (once):  pip install markdown playwright   and   playwright install chromium

How it works
  * Numbers come from artifacts/metrics.json (written by the pipeline) and from the saved outputs of 13.ipynb.
    * When Docker outputs are present, keep its services running. Current metrics must match the live API and
        registry; older notebook outputs are labelled historical rather than blocking a verified current report.
  * 13.md has blocks marked  <!--A:name--> ... <!--/A:name-->  . Only the text inside these blocks is rewritten,
    everything else in 13.md is left alone, so it is safe to edit the rest by hand.
    * Screenshots are saved in docs/screenshots/. MLflow and FastAPI use live Docker services when the notebook
        contains successful Docker checks; otherwise temporary native services use available localhost ports.
        Other screenshots show saved notebook output, live Compose status, and read-only Docker SQLite inspection.
  * Charts (Figures 1-7) are not kept as files. Each one is copied from the notebook output into its
    <!--A:fig_...--> block in 13.md as an embedded image, with a caption built from the same run.
"""
import argparse, ast, html, json, os, re, socket, subprocess, sys, time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
NB, MD, METRICS = ROOT / "13.ipynb", ROOT / "13.md", ROOT / "artifacts" / "metrics.json"
SHOTS = ROOT / "docs" / "screenshots"
NAMES = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest"}


# ---------------------------------------------------------------- read the run ----------------
def cell_text(nb, marker):
    """Text printed by the first code cell whose source contains `marker`."""
    markers = (marker,) if isinstance(marker, str) else marker
    for c in nb["cells"]:
        source = "".join(c["source"])
        if c["cell_type"] == "code" and not source.startswith("%%writefile") and any(text in source for text in markers):
            return "".join("".join(o.get("text", [])) for o in c.get("outputs", []) if o["output_type"] == "stream")
    raise SystemExit(f"Cell with '{marker}' has no output. Run the notebook from top to bottom first.")


def cell_html(nb, marker):
    for c in nb["cells"]:
        if c["cell_type"] == "code" and marker in "".join(c["source"]):
            for o in c.get("outputs", []):
                if o["output_type"] == "execute_result" and "text/html" in o["data"]:
                    return "".join(o["data"]["text/html"])
    return ""


def cell_image(nb, marker):
    """First chart (base64 PNG) drawn by the code cell whose source contains `marker`."""
    for c in nb["cells"]:
        if c["cell_type"] == "code" and marker in "".join(c["source"]):
            for o in c.get("outputs", []):
                if "image/png" in o.get("data", {}):
                    return "".join(o["data"]["image/png"]).replace("\n", "")
    raise SystemExit(f"Cell with '{marker}' has no chart. Run the notebook from top to bottom first.")


def collect():
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    nb = json.loads(NB.read_text(encoding="utf-8"))
    pipe = cell_text(nb, ("# Cell: rebuild the registry", "# Cell: rebuild the native registry", "!python -m pipeline.run_pipeline"))
    if "[1/6 ingest]" not in pipe:
        pipe = docker_output(nb)
    if "[1/6 ingest]" not in pipe:
        raise SystemExit("No saved pipeline output. Run and save the Docker training cell before building the report.")
    c = {"m": m, "nb": nb, "pipe": pipe}
    c["rows"] = int(re.search(r"\[1/6 ingest\]\s+rows=(\d+)", pipe).group(1))
    cl = re.search(r"rows=(\d+) \(removed (\d+)\), fraud=(\d+) \(([\d.]+)%\)", pipe)
    c["clean_rows"], c["removed"], c["clean_fraud"], c["clean_pct"] = int(cl[1]), int(cl[2]), int(cl[3]), cl[4]
    sp = re.search(r"train=(\d+) val=(\d+) test=(\d+)", pipe)
    c["train"], c["val"], c["test"] = (int(x) for x in sp.groups())
    # raw data facts (EDA cell): shape, duplicates, fraud rate
    eda = cell_text(nb, "# Cell: first look")
    c["raw_fraud"] = int(re.search(r"^1\s+(\d+)", eda, re.M).group(1))
    c["raw_pct"] = re.search(r"Fraud rate: ([\d.]+)%", eda).group(1)
    lat = re.search(r"median ([\d.]+) ms, p95 ([\d.]+) ms, max ([\d.]+) ms",
                    cell_text(nb, ("# Cell: send 100 requests", "# Cell: measure 100 successful responses")))
    c["lat"] = lat.groups()
    c["tests"] = re.search(r"(\d+) passed", re.sub(r"\x1b\[[0-9;]*m", "",
                           cell_text(nb, ("# Cell: run all automated tests", "# Cell: run tests with the active kernel")))).group(1)
    calls = cell_text(nb, "# Cell: send one fraud row")
    c["fraud_call"] = json.loads(calls.split("--- fraud transaction (HTTP 200)")[1].split("--- genuine")[0])
    c["genuine_call"] = json.loads(calls.split("--- genuine transaction (HTTP 200)")[1])
    c["rows_logged"] = re.search(r"Rows logged: (\d+)", cell_text(nb, "# Cell: show the prediction log")).group(1)
    reg = cell_text(nb, "# Cell: show the MLflow runs")
    c["version"] = re.search(r"points to version (\d+)", reg).group(1)
    docker_version = re.search(r"(?:Docker champion version:|champion -> version)\s*(\d+)", docker_output(nb))
    c["docker_version"] = docker_version.group(1) if docker_version else None
    docker_health = re.search(r"^Health:\s*(.+)$", docker_output(nb), re.M)
    c["docker_health"] = ast.literal_eval(docker_health.group(1)) if docker_health else None
    c["metrics_registry"] = "native"
    if c["docker_version"] is not None:
        verify_live_docker(c)
    elif str(m.get("model_version")) != c["version"] or m["threshold"] != c["fraud_call"]["threshold"]:
        raise SystemExit("Metrics do not match the saved native evaluation. Run the notebook and save its outputs first.")
    return c


def verify_live_docker(c):
    m = c["m"]
    saved_version = c["docker_version"]
    try:
        api_url = docker_url("prediction-api", 8000)
        registry_url = docker_url("registry", 5000)
        health = read_json(f"{api_url}/health")
        model = read_json(f"{registry_url}/api/2.0/mlflow/registered-models/alias?name=fraud-model&alias=champion")["model_version"]
        run = read_json(f"{registry_url}/api/2.0/mlflow/runs/get?run_id={model['run_id']}")["run"]["data"]
        params = {item["key"]: item["value"] for item in run["params"]}
        metrics = {item["key"]: item["value"] for item in run["metrics"]}
        matches = (
            health["status"] == "ok"
            and str(health["model_version"]) == str(model["version"]) == str(m["model_version"])
            and health["threshold"] == float(params["threshold"]) == m["threshold"]
            and params["model_name"] == m["winner"]
            and all(metrics.get(f"{split}_{key}") == value
                    for split in ("val", "test") for key, value in m[split].items()
                    if key not in ("tn", "fp", "fn", "tp"))
        )
        if not matches:
            raise ValueError("current metrics, live API, champion, or logged evaluation metrics do not match")
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, SystemExit) as error:
        raise SystemExit(
            f"Could not verify metrics version {m.get('model_version')} against live Docker: {error}. "
            "Start the matching Docker registry and API, or rerun the Docker training workflow. "
            "Older saved notebook versions alone do not prevent report generation."
        ) from error
    c["saved_docker_version"] = saved_version
    c["version"] = c["docker_version"] = str(model["version"])
    c["docker_health"] = health
    c["metrics_registry"] = "Docker"
    c["live_docker"] = (
        f"Live Docker champion version: {model['version']}\n"
        f"MLflow run: {model['run_id']}\n"
        f"Health: {health}\n"
        "Current evaluation metrics verified against the live MLflow run."
    )
    print(c["live_docker"])


# ---------------------------------------------------------------- text blocks ------------------
def pct(x, d=1): return f"{x * 100:.{d}f}%"
def n(x): return f"{x:,}"


def g_q1(c):
    return f"with {n(c['rows'])} transactions of which only {c['raw_fraud']} ({c['raw_pct']}%) are fraud"


def g_results(c):
    m, w = c["m"], c["m"]["winner"]
    v, t = m["val"], m["test"]
    cand = m["candidates"]
    ok_r, ok_f = m["targets_met"]["recall_ok"], m["targets_met"]["fpr_ok"]
    rows = "\n".join(f"| {NAMES[k]} | {x['val']['threshold']:.4f} | {x['val']['recall']:.4f} | {x['val']['precision']:.4f} | "
                     f"{pct(x['val']['fpr'], 2)} | {x['val']['pr_auc']:.3f} |" for k, x in cand.items())
    other = [k for k in cand if k != w][0]
    ov = cand[other]["val"]
    if ov["fpr"] >= 0.02 > v["fpr"]:
        why = (f"{NAMES[other]} gives {pct(ov['fpr'], 2)} false positives on validation, which is above the 2% limit, "
               f"but {NAMES[w]} stays within it, so we selected {NAMES[w]} (version {c['version']} in the registry).")
    else:
        why = (f"{NAMES[w]} has the lower validation false-positive rate ({pct(v['fpr'], 2)} against {pct(ov['fpr'], 2)}), "
               f"so we selected it (version {c['version']} in the registry).")
    verdict = []
    verdict.append("The FPR target (below 2%) is " + ("met" if ok_f else "not met"))
    verdict.append("the recall target (at least 90%) is " + ("met" if ok_r else "not met") + f" on the test set ({pct(t['recall'])})")
    note = ""
    if not ok_r:
        note = (f" On validation the recall was {pct(v['recall'])}. The test set has only {t['tp'] + t['fn']} frauds and the "
                f"validation set {v['tp'] + v['fn']}, so missing just one more fraud changes recall by about one percentage point, and a "
                "threshold that is only just enough on validation can fall short on new data. We report the result as it is and did "
                "not tune the threshold on the test set.")
    loss_par = ""
    if "loss_reduction" in t:
        loss_par = (f"**Fraud loss (goal G4):** We measured the loss in money using the Amount column and assumed that every flagged fraud is stopped. "
                    f"The {t['tp'] + t['fn']} frauds in the test set add up to {t['fraud_amount_total']:,.2f} and the model catches {t['fraud_amount_caught']:,.2f} of it, "
                    f"so the fraud loss goes down by {pct(t['loss_reduction'])}, against the target of 30% ({'met' if m['targets_met'].get('loss_ok') else 'not met'}). "
                    f"The cost is that genuine transactions worth {t['genuine_amount_blocked']:,.2f} were also flagged, and a real bank would have to consider the cost of these blocked payments too. "
                    "This is only an estimate from the test set and it does not include chargeback fees or the cost of reviewing the flagged transactions.\n\n")
    evidence = (f"**Run evidence:** These metrics describe {c['metrics_registry']} champion version {c['version']}. "
                f"Docker screenshots describe the separate Docker champion version {c['docker_version']}. "
                f"The saved native prediction examples serve version {c['fraud_call']['model_version']}; "
                "latency measurements and charts come from the saved native notebook run. "
                "Native and Docker registries have independent version numbers.\n\n") if c["docker_version"] else ""
    if c.get("live_docker"):
        evidence += ("**Evidence source:** The current Docker version, threshold, selected model, and evaluation metrics were verified against "
                     "the live Docker API and champion's MLflow run; saved notebook outputs are historical evidence.\n\n")
    return f"""{evidence}**Model results:** The dataset has {n(c['rows'])} rows with {c['raw_fraud']} frauds ({c['raw_pct']}%). After removing {n(c['removed'])} duplicate rows, {n(c['clean_rows'])} rows remain with {c['clean_fraud']} frauds ({c['clean_pct']}%). The stratified split gives {n(c['train'])} training, {n(c['val'])} validation and {n(c['test'])} test rows. For each model the threshold is the highest one that still gives a validation recall of at least 0.90.

| Model (validation) | Threshold | Recall | Precision | FPR | PR-AUC |
|---|---|---|---|---|---|
{rows}

Both models were tuned to the recall target on validation. {why} Below are the results of the selected model on the test set, which we used only once:

| Recall | Precision | F1 | PR-AUC | ROC-AUC | FPR |
|---|---|---|---|---|---|
| {t['recall']:.4f} | {t['precision']:.4f} | {t['f1']:.4f} | {t['pr_auc']:.4f} | {t['roc_auc']:.4f} | {pct(t['fpr'], 2)} |

Confusion matrix on the test set: TN {n(t['tn'])}, FP {n(t['fp'])}, FN {n(t['fn'])}, TP {n(t['tp'])}. **{verdict[0]}, {'and' if ok_f == ok_r else 'but'} {verdict[1]}.**{note}

{loss_par}**Note on the selection rule:** In our first run we chose the winner by the best validation PR-AUC and Random Forest won. On the test set it missed both targets (recall 87.4%, FPR 4.74%). After seeing this we changed the rule to the lowest validation FPR, with PR-AUC as tie-break, because the false-alarm limit is one of our stated goals. The change was based on validation results only, but we made it after seeing the first run, so we mention it here."""


def g_analytics(c):
    m = c["m"]; w = m["winner"]; v = m["val"]; t = m["test"]
    other = [k for k in m["candidates"] if k != w][0]
    ov = m["candidates"][other]["val"]
    return (f"In our run {NAMES[w]} was selected because it had the lower validation false-positive rate "
            f"({pct(v['fpr'], 2)} against {pct(ov['fpr'], 2)} for {NAMES[other]}), although {NAMES[other]} had the better PR-AUC "
            f"({ov['pr_auc']:.3f} against {v['pr_auc']:.3f}). The recall of {NAMES[w]} was {pct(v['recall'])} on validation and "
            f"{pct(t['recall'])} on the test set.")


def g_registry(c):
    m = c["m"]
    registry = f"{c['metrics_registry']} registry"
    return (f"After the run, the {registry} has fraud-model with {NAMES[m['winner']]} as version {c['version']} with the alias champion, and the stored "
            f"threshold is {m['threshold']:.4f}. At startup the prediction service resolves champion once, loads the "
            "fixed numeric model version, and reads the threshold from that same version's run. This prevents a "
            "concurrent promotion from mixing a model with another version's threshold. Restart the API after promotion to load it.")


def g_latency(c):
    med, p95, mx = c["lat"]
    return (f"**Measured latency:** For 100 requests the median is {med} ms, the p95 is {p95} ms and the maximum is {mx} ms. "
            f"The target of p95 below 200 ms is {'met' if float(p95) < 200 else 'not met'}.")


def g_tests(c):
    return f"The last pytest run shows **{c['tests']} passed**."


def g_shots(c):
    f, g, t = c["fraud_call"], c["genuine_call"], c["m"]["test"]
    w = c["m"]["winner"]
    rows = [
        ("Pipeline run", "pipeline_run", f"Output of the six-filter pipeline: {n(c['rows'])} rows read, {n(c['removed'])} duplicates removed, the 60/20/20 split, both models trained and {NAMES[w]} selected, with test recall {t['recall']:.3f} and FPR {t['fpr']:.4f}."),
        ("MLflow", "mlflow", f"Live {'Docker ' if c.get('live_docker') or docker_output(c['nb']) else ''}registry page of fraud-model. Version {c['docker_version'] or c['version']} is registered and has the alias champion."),
        ("FastAPI", "fastapi_docs", f"The live {'Docker API ' if c.get('live_docker') or docker_output(c['nb']) else 'service '}/docs page with an executed GET /health response showing model version {c['docker_version'] or c['version']}, plus POST /predict request and response fields. The Swagger header's API release number is not the MLflow model version."),
        ("Fraud test call", "test_fraud", f"A saved native notebook request, serving model version {f['model_version']}. The fraud probability is {f['probability']:.4f}, is_fraud is {str(f['is_fraud']).lower()} and the answer came in {f['latency_ms']:.0f} ms."),
        ("Genuine test call", "test_genuine", f"A saved native notebook request, serving model version {g['model_version']}. The probability is {g['probability']:.3f}, {'below' if g['probability'] < g['threshold'] else 'above'} the threshold {g['threshold']:.4f}, so is_fraud is {str(g['is_fraud']).lower()}."),
        ("Prediction log", "prediction_log", f"The prediction log after the test calls and the 100 latency requests: {c['rows_logged']} rows, each with timestamp, request id, probability, label, latency and model version."),
    ]
    dock = c.get("live_docker") or docker_output(c["nb"])
    if dock:
        rows.append(("Docker Compose", "docker_compose",
                     f"Live Compose status and {'live registry verification' if c.get('live_docker') else 'the saved Docker notebook checks'}: registry (container port 5000) and "
                     "prediction-api (container port 8000), published on dynamically assigned localhost ports. "
                     f"The Docker champion is version {c['docker_version']}; services remain running until manually stopped."))
        rows.append(("Docker database", "docker_database",
                     "Read-only SQLite inspection executed inside the registry container, showing the database file, "
                     f"tables, model-version count, and champion model version {c['docker_version']}. /app/docker-mlflow.db persists in the bind-mounted project folder; "
                     "SQLite is embedded in the registry, not a separate database-server container."))
    return "\n\n".join(f"**Screenshot {i} - {a}.** {e}\n\n![{a}](docs/screenshots/{b}.png)" for i, (a, b, e) in enumerate(rows, 1))


def docker_output(nb):
    """Output of the Docker Compose cell, or "" when Docker was not available in the last run."""
    try:
        out = cell_text(nb, ("# Cell: build and start both containers", "# Cell: train and serve inside Docker",
                     "# Cell: train and start Docker services"))
    except SystemExit:
        return ""
    return out if re.search(r"(?:Docker champion version:|champion -> version)\s*\d+", out) else ""


def g_deployment(c):
    for cell in c["nb"]["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown" and source.startswith("## 8. Containerised deployment"):
            return source.partition("\n")[2].strip()
    raise SystemExit("Docker deployment documentation is missing from the notebook.")


def g_conclusion(c):
    m = c["m"]; t = m["test"]; med, p95, mx = c["lat"]
    ok_r, ok_f = m["targets_met"]["recall_ok"], m["targets_met"]["fpr_ok"]
    goals = (f"false-positive rate ({pct(t['fpr'], 2)} against 2%) is {'met' if ok_f else 'not met'}, "
             f"latency (p95 {p95} ms against 200 ms) is {'met' if float(p95) < 200 else 'not met'}, and "
             f"the test recall is {pct(t['recall'])} against the 90% target, so that goal is {'met' if ok_r else 'not met'}"
             + (f". The estimated fraud loss reduction is {pct(t['loss_reduction'])} against the 30% target ({'met' if m['targets_met'].get('loss_ok') else 'not met'})" if "loss_reduction" in t else ""))
    return ("In this assignment we built a fraud detection system with a pipe-and-filter training pipeline, an MLflow registry and a FastAPI "
            f"microservice that serves the champion model, with prediction logging and {c['tests']} automated tests."
            f"{' With Docker Compose the registry and the prediction API also run as two separate containers.' if docker_output(c['nb']) else ''}"
            f" Against our goals: {goals}. "
            "From this work we learned that with so few frauds in each split the recall on the test set can differ from the validation recall, "
            "that the model for serving should be chosen on validation results only, and that a registry alias keeps the service independent "
            "of the model version. If we had more time we would try more tuning on validation, cross-validation for the threshold, and more fraud examples.")


def g_goals(c):
    m = c["m"]; t = m["test"]; med, p95, mx = c["lat"]
    yn = lambda ok: "Yes" if ok else "No"
    return ("| ID | Goal | GR4ML concept | Metric | Target | Result (latest run) | Met? |\n|---|---|---|---|---|---|---|\n"
            f"| G1 | Catch fraud | Indicator of the strategic goal; softgoal High recall | Recall on the test set | at least 90% | {pct(t['recall'])} | {yn(m['targets_met']['recall_ok'])} |\n"
            f"| G2 | Avoid blocking genuine customers | Indicator of the strategic goal; softgoal Few false alarms | False-positive rate on the test set | below 2% | {pct(t['fpr'], 2)} | {yn(m['targets_met']['fpr_ok'])} |\n"
            f"| G3 | Decide while the payment is processed | Softgoal Low latency | p95 latency of the prediction API | below 200 ms | {p95} ms | {yn(float(p95) < 200)} |\n"
            + g_loss_row(m))


def g_loss_row(m):
    t = m["test"]
    if "loss_reduction" not in t:
        return "| G4 | Reduce losses | Strategic goal | Fraud loss | 30% reduction (business target) | run the notebook again | - |"
    ok = m["targets_met"].get("loss_ok", False)
    return f"| G4 | Reduce losses | Strategic goal | Share of fraud money caught (test set) | at least 30% | {pct(t['loss_reduction'])} | {'Yes' if ok else 'No'} |"


# ---------------------------------------------------------------- figures ----------------------
# The charts are not saved as files: each one is taken from the notebook output and embedded in 13.md.
def fig(c, marker, num, title, text):
    return f"**Figure {num} - {title}.** {text}\n\n![{title}](data:image/png;base64,{cell_image(c['nb'], marker)})"


def g_fig_eda(c):
    return fig(c, "# Cell: plot the class balance", 1, "Class balance and amount",
               f"Only {n(c['raw_fraud'])} of the {n(c['rows'])} transactions ({c['raw_pct']}%) are fraud, and most "
               "amounts are small with a long tail of large payments.")


def g_fig_hour(c):
    g, f = re.search(r"Hours 0-6: ([\d.]+)% of genuine, ([\d.]+)% of fraud",
                     cell_text(c["nb"], "# Cell: share of genuine")).groups()
    return fig(c, "# Cell: share of genuine", 2, "Time of the transactions",
               "Each bar is the share of that class's transactions in one hour (Time is counted from the first "
               f"transaction, so hour 0 is not midnight). {f}% of the frauds fall in hours 0 to 6, where only {g}% of "
               "the genuine transactions happen, so fraud is relatively more common when there are few genuine "
               "payments. This is why Time is kept as a model input.")


def g_fig_amount(c):
    t = cell_text(c["nb"], "# Cell: Amount of genuine")
    gm, fm = (re.search(rf"{k}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", t).groups() for k in ("genuine", "fraud"))
    top = max(float(gm[3]), float(fm[3]))
    return fig(c, "# Cell: Amount of genuine", 3, "Amount",
               f"The median fraud is {fm[1]} against {gm[1]} for a genuine payment, and the 95th percentile is "
               f"{fm[2]} against {gm[2]}, so frauds are more spread out. Amount has a long tail, with a maximum of "
               f"{top:,.2f}, which is why it is scaled with a RobustScaler in the Data Preparation View (Section 3.3).")


def g_fig_features(c):
    t = cell_text(c["nb"], "# Cell: the V features whose average")
    gap = ast.literal_eval(t.split("Gap in standard deviations:")[1].strip())
    listing = ", ".join(f"{k} {v:.1f}" for k, v in gap.items())
    return fig(c, "# Cell: the V features whose average", 4, "Features that separate fraud best",
               f"These are the {len(gap)} V features whose average differs most between the classes, measured in "
               f"standard deviations ({listing}). The genuine averages are close to 0 and the fraud averages are far "
               "from it. A simple linear model can use such differences, which supports Logistic Regression as a "
               "candidate in the Analytics Design View (Section 3.2).")


def g_fig_cm(c):
    t = c["m"]["test"]
    return fig(c, "# Cell: draw the confusion matrix", 5, "Confusion matrix on the test set",
               f"It shows the four counts given above: {t['tp']} of the {t['tp'] + t['fn']} frauds are caught and "
               f"{n(t['fp'])} genuine transactions are flagged.")


def g_fig_models(c):
    m = c["m"]
    fprs = ", ".join(f"{NAMES.get(k, k)} {pct(x['val']['fpr'])} (PR-AUC {x['val']['pr_auc']:.3f})"
                     for k, x in m["candidates"].items())
    return fig(c, "# Cell: compare the two candidates", 6, "Model comparison on the validation set",
               "Each threshold was set to reach at least 90% recall on validation, so the false-positive rate "
               f"decides: {fprs}, against the 2% target line. {NAMES.get(m['winner'], m['winner'])} has the lowest "
               "false-positive rate and was selected.")


def g_fig_pr(c):
    p, k = re.search(r"precision ([\d.]+), (\d+) genuine transactions flagged",
                     cell_text(c["nb"], "# Cell: draw the precision-recall curve")).groups()
    return fig(c, "# Cell: draw the precision-recall curve", 7, "Precision-recall curve on the test set",
               "This is the saved notebook's frozen native-model curve, not a re-evaluation of the Docker model. "
               "The red point uses its validation-selected threshold; the dashed line is the 90% recall target. "
               f"Post-hoc analysis of this saved test set gives precision {pct(float(p))} at the recall target, "
               f"with {n(int(k))} genuine transactions flagged. This analysis is not used to tune, serve or promote a model.")


GEN = {"q1": g_q1, "results": g_results, "analytics": g_analytics, "registry": g_registry,
       "latency": g_latency, "goals": g_goals, "tests": g_tests, "shots": g_shots, "conclusion": g_conclusion,
    "deployment": g_deployment,
       "fig_eda": g_fig_eda, "fig_hour": g_fig_hour, "fig_amount": g_fig_amount, "fig_features": g_fig_features,
       "fig_cm": g_fig_cm, "fig_models": g_fig_models, "fig_pr": g_fig_pr}


def update_md(c):
    s = MD.read_text(encoding="utf-8")
    for key, fn in GEN.items():
        pat = re.compile(rf"(<!--A:{key}-->).*?(<!--/A:{key}-->)", re.S)
        if not pat.search(s):
            print(f"  warning: block A:{key} not found in 13.md")
            continue
        s = pat.sub(lambda mo: mo.group(1) + fn(c) + mo.group(2), s)
    MD.write_text(s, encoding="utf-8")
    print("13.md updated from the latest run")


# ---------------------------------------------------------------- screenshots ------------------
CSS = ("body{margin:0;padding:24px;background:#fff;font:15px/1.5 Menlo,Consolas,monospace;color:#222} "
       "h3{font:600 17px sans-serif;margin:0 0 12px;color:#444} pre{background:#f6f8fa;border:1px solid #ddd;padding:16px;"
       "border-radius:6px;white-space:pre-wrap} table{border-collapse:collapse;font:13px sans-serif} td,th{border:1px solid #ccc;padding:4px 10px}")


def page(title, body): return f"<html><style>{CSS}</style><body><h3>{title}</h3>{body}</body></html>"


def pipeline_page(c):
    stages = [line.strip() for line in c["pipe"].splitlines()
              if re.match(r"^\s*\[[1-6]/6\s+\w+\]", line)]
    if {int(line[1]) for line in stages} != set(range(1, 7)):
        raise ValueError("Screenshot 1 requires saved output from all six pipeline stages.")
    metrics = c["m"]
    stage_output = html.escape("\n".join(stages))
    registry = html.escape(c["metrics_registry"])
    version = html.escape(str(metrics["model_version"]))
    rows = "".join(
        f"<tr><td>{label}</td><td>{metrics['val'][key]:.4f}</td>"
        f"<td>{metrics['test'][key]:.4f}</td></tr>"
        for label, key in (("Recall", "recall"), ("Precision", "precision"),
                           ("PR-AUC", "pr_auc"), ("False-positive rate", "fpr")))
    note = ("Current metrics verified against live Docker; saved stage output is historical."
            if c.get("live_docker") else "Metrics match the saved registry evidence.")
    return page("Pipeline run - six-filter output",
                f"<pre>{stage_output}</pre><h3>Evaluation: {registry} registry, version {version}</h3>"
                f"<p>Champion: {html.escape(NAMES[metrics['winner']])} | Threshold: {metrics['threshold']:.6f}</p>"
                f"<table><thead><tr><th>Metric</th><th>Validation</th><th>Test</th></tr></thead>"
                f"<tbody>{rows}</tbody></table>"
                f"<p>Test recall target &gt;= 90%: {'PASS' if metrics['targets_met']['recall_ok'] else 'NOT MET'}. "
                f"False-positive rate target &lt; 2%: {'PASS' if metrics['targets_met']['fpr_ok'] else 'NOT MET'}.</p>"
                f"<p>{note}</p>")


def wait_port(port, secs=60):
    for _ in range(secs):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(1)
    raise SystemExit(f"nothing started on port {port}")


def compose(*arguments):
    return subprocess.run(["docker", "compose", "--ansi", "never", *arguments], cwd=ROOT,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", check=True).stdout


def docker_url(service, port):
    binding = compose("port", service, str(port)).strip()
    if not re.fullmatch(r"127\.0\.0\.1:\d+", binding):
        raise SystemExit(f"Unexpected Docker address for {service}: {binding!r}")
    return f"http://{binding}"


def read_json(url):
    with urlopen(url, timeout=15) as response:
        return json.load(response)


def available_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def chromium(p):
    exe = os.environ.get("PW_CHROME")
    return p.chromium.launch(executable_path=exe, args=["--no-sandbox"]) if exe else p.chromium.launch()


def take_shots(c):
    from playwright.sync_api import sync_playwright
    SHOTS.mkdir(parents=True, exist_ok=True)
    nb = c["nb"]
    calls = cell_text(nb, "# Cell: send one fraud row").split("--- ")
    pages = {
        "pipeline_run": pipeline_page(c),
        "test_fraud": page("POST /predict - fraud transaction", f"<pre>{html.escape('--- ' + calls[1])}</pre>"),
        "test_genuine": page("POST /predict - genuine transaction", f"<pre>{html.escape('--- ' + calls[2])}</pre>"),
        "prediction_log": page("Prediction log (predictions_log.csv)", f"<pre>{html.escape(cell_text(nb, 'Rows logged'))}</pre>{cell_html(nb, 'Rows logged')}"),
    }
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"}
    processes = []
    try:
        if c.get("live_docker") or docker_output(nb):
            api_url = docker_url("prediction-api", 8000)
            registry_url = docker_url("registry", 5000)
            health = read_json(f"{api_url}/health")
            champion = read_json(f"{registry_url}/api/2.0/mlflow/registered-models/alias?name=fraud-model&alias=champion")
            if str(health["model_version"]) != c["docker_version"] or str(champion["model_version"]["version"]) != c["docker_version"]:
                raise SystemExit("Live Docker and saved notebook versions differ. Save the latest notebook before rebuilding the report.")
            expected_threshold = c["docker_health"]["threshold"] if c["docker_health"] else c["m"]["threshold"]
            if health["threshold"] != expected_threshold:
                raise SystemExit("Live Docker and evaluation thresholds differ. Regenerate matching training evidence before building the report.")
            status = compose("ps")
            verification = "\n".join(line for line in docker_output(nb).splitlines() if line.startswith(
                ("Docker API docs:", "Docker registry:", "Health:", "Prediction:", "Docker champion version:",
                 "Registry SQLite database:", "Services remain running", "champion -> version")))
            verification = c.get("live_docker") or verification
            pages["docker_compose"] = page("Docker Compose: live status and registry verification" if c.get("live_docker") else "Docker Compose: live status and saved notebook verification",
                                            f"<pre>{html.escape(status)}</pre><pre>{html.escape(verification)}</pre>")
            inspection = ("import json, sqlite3; from pathlib import Path; "
                          "database=Path('/app/docker-mlflow.db'); "
                          "connection=sqlite3.connect('file:/app/docker-mlflow.db?mode=ro', uri=True); "
                          "print(json.dumps({'database':str(database), 'size_bytes':database.stat().st_size, "
                          "'tables':[row[0] for row in connection.execute(\"SELECT name FROM sqlite_master WHERE type='table' "
                          "AND name IN ('experiments','runs','registered_models','model_versions','registered_model_aliases') ORDER BY name\")], "
                          "'model_versions':connection.execute('SELECT COUNT(*) FROM model_versions').fetchone()[0], "
                          "'champion_model_version':connection.execute(\"SELECT version FROM registered_model_aliases "
                          "WHERE name='fraud-model' AND alias='champion'\").fetchone()[0]}, indent=2))")
            database = json.loads(compose("exec", "-T", "registry", "python", "-c", inspection))
            if str(database["champion_model_version"]) != c["docker_version"]:
                raise SystemExit("The SQLite champion changed during screenshot capture; rerun the report.")
            pages["docker_database"] = page("Docker registry: persistent SQLite database (read-only inspection)",
                                             f"<pre>{html.escape(json.dumps(database, indent=2))}</pre>")
        else:
            registry_port, api_port = available_port(), available_port()
            processes.append(subprocess.Popen([sys.executable, "-m", "mlflow", "server", "--backend-store-uri",
                                                f"sqlite:///{ROOT / 'mlflow.db'}", "--host", "127.0.0.1", "--port", str(registry_port)],
                                               cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            processes.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(api_port)],
                                               cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            wait_port(registry_port); wait_port(api_port)
            api_url, registry_url = f"http://127.0.0.1:{api_port}", f"http://127.0.0.1:{registry_port}"
        with sync_playwright() as p:
            b = chromium(p)
            context = b.new_context(viewport={"width": 1100, "height": 400}, device_scale_factor=2)
            pg = context.new_page()
            pg.set_default_timeout(30000)
            pg.set_default_navigation_timeout(30000)
            for name, h in pages.items():
                print(f"Capturing screenshot: {name}", flush=True)
                pg.set_content(h); pg.screenshot(path=str(SHOTS / f"{name}.png"), full_page=True)
            print("Capturing screenshot: FastAPI docs", flush=True)
            pg.set_viewport_size({"width": 1200, "height": 800})
            sw = os.environ.get("SWAGGER_DIST")            # only needed when the CDN is blocked
            if sw:
                def route(r):
                    u = r.request.url
                    if u.endswith("swagger-ui.css"): r.fulfill(path=sw + "/swagger-ui.css", content_type="text/css")
                    elif u.endswith("swagger-ui-bundle.js"): r.fulfill(path=sw + "/swagger-ui-bundle.js", content_type="application/javascript")
                    elif "favicon" in u: r.abort()
                    else: r.continue_()
                pg.route("**/*", route)
            pg.goto(f"{api_url}/docs"); pg.wait_for_selector(".opblock", timeout=30000)
            for blk in pg.locator(".opblock-summary").all(): blk.click()
            health_block = pg.locator(".opblock-get:has(.opblock-summary-path[data-path='/health'])")
            health_block.get_by_role("button", name="Try it out", exact=True).click()
            with pg.expect_response(f"{api_url}/health") as executed_health:
                health_block.get_by_role("button", name="Execute", exact=True).click()
            response = executed_health.value
            served = response.json()
            if (response.status != 200 or served["status"] != "ok"
                    or str(served["model_version"]) != str(c["version"])
                    or served["threshold"] != c["m"]["threshold"]):
                raise SystemExit("Swagger's live health response does not match the verified report model.")
            health_block.locator(".live-responses-table").wait_for()
            pg.screenshot(path=str(SHOTS / "fastapi_docs.png"), full_page=True)
            print("Capturing screenshot: MLflow registry", flush=True)
            pg.set_viewport_size({"width": 1280, "height": 800})
            # The MLflow page is a JavaScript app: wait until the model name is really on the screen
            # (a fixed sleep gave a blank picture on a slower start), and retry with a reload if needed.
            ok = False
            for _ in range(3):
                pg.goto(f"{registry_url}/#/models/fraud-model")
                try:
                    pg.wait_for_selector("text=Registered Models", timeout=30000)
                    pg.wait_for_selector("text=champion", timeout=30000)
                    ok = True
                    break
                except Exception:
                    pg.reload()
            try: pg.get_by_role("button", name="Close").first.click(timeout=2000)   # close the assistant panel if it is open
            except Exception: pass
            pg.screenshot(path=str(SHOTS / "mlflow.png"))
            if not ok:
                raise RuntimeError("MLflow did not render its registry and champion; refusing to publish an incomplete report.")
            b.close()
        print("screenshots saved in docs/screenshots/")
    finally:
        for process in processes:
            process.terminate()
            process.wait(timeout=15)


# ---------------------------------------------------------------- PDF --------------------------
def make_pdf():
    import markdown
    from playwright.sync_api import sync_playwright
    text = MD.read_text(encoding="utf-8")
    text = re.sub(r"<!--(?!A:|/A:).*?-->", "", text, flags=re.S)                 # hide author comments
    text = re.sub(r"```mermaid\n(.*?)```", lambda m: f'<pre class="mermaid">{html.escape(m.group(1))}</pre>', text, flags=re.S)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    css = ("body{font:11pt/1.45 'Helvetica Neue',Arial,sans-serif;max-width:none;color:#111} h1{font-size:20pt} h2{font-size:15pt;"
           "border-bottom:1px solid #ccc;padding-bottom:3px;margin-top:22px} h3{font-size:12.5pt} table{border-collapse:collapse;"
           "margin:8px 0;font-size:9.5pt} td,th{border:1px solid #bbb;padding:4px 8px;vertical-align:top} th{background:#f0f0f0} "
           "pre{background:#f6f8fa;border:1px solid #ddd;padding:8px;font-size:8.5pt;white-space:pre-wrap;page-break-inside:avoid} "
           "pre.mermaid{background:#fff;border:none;text-align:center;page-break-inside:avoid;break-inside:avoid} h1,h2,h3,h4{break-after:avoid;page-break-after:avoid} img{break-inside:avoid} img{max-width:100%;max-height:215mm;width:auto;display:block;margin:0 auto} p:has(+ p > img:only-child){break-after:avoid;page-break-after:avoid} code{font-size:9pt}")
    out = ROOT / "13.html"
    if os.environ.get("MERMAID_JS"):                       # optional: use a local copy instead of the CDN
        mermaid = f"<script>{Path(os.environ['MERMAID_JS']).read_text(encoding='utf-8')}</script>"
    else:
        mermaid = "<script src='https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js'></script>"
    init = "<script>mermaid.initialize({startOnLoad:true,theme:'default',flowchart:{useMaxWidth:true}});</script>"
    out.write_text(f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}{mermaid}{init}</body></html>",
                   encoding="utf-8")
    with sync_playwright() as p:
        b = chromium(p); pg = b.new_page()
        pg.goto(out.as_uri()); pg.wait_for_load_state("networkidle")
        pg.wait_for_function("Array.from(document.querySelectorAll('pre.mermaid')).every(node => node.querySelector('svg'))", timeout=30000)
        pg.wait_for_function("Array.from(document.images).every(image => image.complete && image.naturalWidth > 0)", timeout=30000)
        pg.evaluate("document.fonts.ready")
        pg.pdf(path=str(ROOT / "13.pdf"), format="A4", margin={"top": "15mm", "bottom": "15mm", "left": "14mm", "right": "14mm"},
               print_background=True)
        b.close()
    out.unlink()
    print("13.pdf created")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-shots", action="store_true"); ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()
    ctx = collect()
    if not a.no_shots: take_shots(ctx)
    update_md(ctx)
    if not a.no_pdf: make_pdf()
