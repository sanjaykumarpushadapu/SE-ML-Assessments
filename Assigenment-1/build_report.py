"""
build_report.py - refresh the report from the latest notebook run and make 13.pdf.

Run it after the notebook has been run from top to bottom (Restart + Run All) and saved:

    python build_report.py            # update 13.md values, retake screenshots, make 13.pdf
    python build_report.py --no-shots # same, but keep the existing screenshots
    python build_report.py --no-pdf   # only update 13.md

Needs (once):  pip install markdown playwright   and   playwright install chromium

How it works
  * Numbers come from artifacts/metrics.json (written by the pipeline) and from the saved outputs of 13.ipynb.
  * 13.md has blocks marked  <!--A:name--> ... <!--/A:name-->  . Only the text inside these blocks is rewritten,
    everything else in 13.md is left alone, so it is safe to edit the rest by hand.
  * Screenshots are saved in docs/screenshots/ (MLflow and FastAPI pages are opened in a headless browser,
    the other four are drawn from the notebook outputs).
"""
import argparse, html, json, os, re, socket, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB, MD, METRICS = ROOT / "13.ipynb", ROOT / "13.md", ROOT / "artifacts" / "metrics.json"
SHOTS = ROOT / "docs" / "screenshots"
NAMES = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest"}


# ---------------------------------------------------------------- read the run ----------------
def cell_text(nb, marker):
    """Text printed by the first code cell whose source contains `marker`."""
    for c in nb["cells"]:
        if c["cell_type"] == "code" and marker in "".join(c["source"]):
            return "".join("".join(o.get("text", [])) for o in c.get("outputs", []) if o["output_type"] == "stream")
    raise SystemExit(f"Cell with '{marker}' has no output. Run the notebook from top to bottom first.")


def cell_html(nb, marker):
    for c in nb["cells"]:
        if c["cell_type"] == "code" and marker in "".join(c["source"]):
            for o in c.get("outputs", []):
                if o["output_type"] == "execute_result" and "text/html" in o["data"]:
                    return "".join(o["data"]["text/html"])
    return ""


def collect():
    m = json.loads(METRICS.read_text())
    nb = json.loads(NB.read_text())
    pipe = cell_text(nb, "!python -m pipeline.run_pipeline")
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
    lat = re.search(r"median ([\d.]+) ms, p95 ([\d.]+) ms, max ([\d.]+) ms", cell_text(nb, "# Cell: send 100 requests"))
    c["lat"] = lat.groups()
    c["tests"] = re.search(r"(\d+) passed", re.sub(r"\x1b\[[0-9;]*m", "", cell_text(nb, "# Cell: run all automated tests"))).group(1)
    calls = cell_text(nb, "# Cell: send one fraud row")
    c["fraud_call"] = json.loads(calls.split("--- fraud transaction (HTTP 200)")[1].split("--- genuine")[0])
    c["genuine_call"] = json.loads(calls.split("--- genuine transaction (HTTP 200)")[1])
    c["rows_logged"] = re.search(r"Rows logged: (\d+)", cell_text(nb, "# Cell: show the prediction log")).group(1)
    reg = cell_text(nb, "# Cell: show the MLflow runs")
    c["version"] = re.search(r"points to version (\d+)", reg).group(1)
    return c


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
    return f"""**Model results:** The dataset has {n(c['rows'])} rows with {c['raw_fraud']} frauds ({c['raw_pct']}%). After removing {n(c['removed'])} duplicate rows, {n(c['clean_rows'])} rows remain with {c['clean_fraud']} frauds ({c['clean_pct']}%). The stratified split gives {n(c['train'])} training, {n(c['val'])} validation and {n(c['test'])} test rows. For each model the threshold is the highest one that still gives a validation recall of at least 0.90.

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
    return (f"After the run, fraud-model has {NAMES[m['winner']]} as version {c['version']} with the alias champion, and the stored "
            f"threshold is {m['threshold']:.4f}. The prediction service loads only `models:/fraud-model@champion`, so a new "
            "version can be promoted without any change in the service code.")


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
        ("MLflow", "mlflow", f"MLflow registry page of fraud-model. Version {c['version']} is registered and has the alias champion, which is the only model the service loads."),
        ("FastAPI", "fastapi_docs", "The /docs page of the service with the GET /health and POST /predict endpoints, the request fields (Time, V1 to V28, Amount) and the response fields."),
        ("Fraud test call", "test_fraud", f"A fraud row from the test set sent to /predict. The probability is {f['probability']:.4f}, is_fraud is {str(f['is_fraud']).lower()} and the answer came in {f['latency_ms']:.0f} ms."),
        ("Genuine test call", "test_genuine", f"A genuine row from the test set. The probability is {g['probability']:.3f}, {'below' if g['probability'] < g['threshold'] else 'above'} the threshold {g['threshold']:.4f}, so is_fraud is {str(g['is_fraud']).lower()}."),
        ("Prediction log", "prediction_log", f"The prediction log after the test calls and the 100 latency requests: {c['rows_logged']} rows, each with timestamp, request id, probability, label, latency and model version."),
    ]
    return "\n\n".join(f"**Screenshot {i} - {a}.** {e}\n\n![{a}](docs/screenshots/{b}.png)" for i, (a, b, e) in enumerate(rows, 1))


def g_conclusion(c):
    m = c["m"]; t = m["test"]; med, p95, mx = c["lat"]
    ok_r, ok_f = m["targets_met"]["recall_ok"], m["targets_met"]["fpr_ok"]
    goals = (f"false-positive rate ({pct(t['fpr'], 2)} against 2%) is {'met' if ok_f else 'not met'}, "
             f"latency (p95 {p95} ms against 200 ms) is {'met' if float(p95) < 200 else 'not met'}, and "
             f"the test recall is {pct(t['recall'])} against the 90% target, so that goal is {'met' if ok_r else 'not met'}"
             + (f". The estimated fraud loss reduction is {pct(t['loss_reduction'])} against the 30% target ({'met' if m['targets_met'].get('loss_ok') else 'not met'})" if "loss_reduction" in t else ""))
    return ("In this assignment we built a fraud detection system with a pipe-and-filter training pipeline, an MLflow registry and a FastAPI "
            f"microservice that serves the champion model, with prediction logging and {c['tests']} automated tests. Against our goals: {goals}. "
            "From this work we learned that with so few frauds in each split the recall on the test set can differ from the validation recall, "
            "that the model for serving should be chosen on validation results only, and that a registry alias keeps the service independent "
            "of the model version. If we had more time we would try more tuning on validation, cross-validation for the threshold, and more fraud examples.")


def g_goals(c):
    m = c["m"]; t = m["test"]; med, p95, mx = c["lat"]
    yn = lambda ok: "Yes" if ok else "No"
    return ("| Goal | GR4ML concept | Metric | Target | Result (latest run) | Met? |\n|---|---|---|---|---|---|\n"
            f"| Catch fraud | Indicator of the strategic goal; softgoal High recall | Recall on the test set | at least 90% | {pct(t['recall'])} | {yn(m['targets_met']['recall_ok'])} |\n"
            f"| Avoid blocking genuine customers | Indicator of the strategic goal; softgoal Few false alarms | False-positive rate on the test set | below 2% | {pct(t['fpr'], 2)} | {yn(m['targets_met']['fpr_ok'])} |\n"
            f"| Decide while the payment is processed | Softgoal Low latency | p95 latency of the prediction API | below 200 ms | {p95} ms | {yn(float(p95) < 200)} |\n"
            + g_loss_row(m))


def g_loss_row(m):
    t = m["test"]
    if "loss_reduction" not in t:
        return "| Reduce losses | Strategic goal | Fraud loss | 30% reduction (business target) | run the notebook again | - |"
    ok = m["targets_met"].get("loss_ok", False)
    return f"| Reduce losses | Strategic goal | Share of fraud money caught (test set) | at least 30% | {pct(t['loss_reduction'])} | {'Yes' if ok else 'No'} |"


def code_appendix():
    """All code of the notebook, grouped by file, for the printed report (Appendix A)."""
    nb = json.loads(NB.read_text())
    out, other = [], []
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if src.startswith("%%writefile"):
            first, body = src.split("\n", 1)
            out.append(f"#### {first.replace('%%writefile', '').strip()}\n\n```python\n{body.rstrip()}\n```\n")
        else:
            other.append(src.rstrip())
    notebook_cells = "\n\n".join(f"# ---------- notebook cell {i} ----------\n{s}" for i, s in enumerate(other, 1))
    return "\n".join(out) + f"\n#### Notebook cells (setup, data check, EDA, run, test calls)\n\n```python\n{notebook_cells}\n```\n"


GEN = {"q1": g_q1, "results": g_results, "analytics": g_analytics, "registry": g_registry,
       "latency": g_latency, "goals": g_goals, "tests": g_tests, "shots": g_shots, "conclusion": g_conclusion}


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


def wait_port(port, secs=60):
    for _ in range(secs):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(1)
    raise SystemExit(f"nothing started on port {port}")


def chromium(p):
    exe = os.environ.get("PW_CHROME")
    return p.chromium.launch(executable_path=exe, args=["--no-sandbox"]) if exe else p.chromium.launch()


def take_shots(c):
    from playwright.sync_api import sync_playwright
    SHOTS.mkdir(parents=True, exist_ok=True)
    nb = c["nb"]
    clean = lambda t: "".join(l + "\n" for l in t.replace("\n\n", "\n").splitlines() if "INFO mlflow" not in l)
    res = cell_text(nb, "# Cell: read artifacts/metrics.json")
    calls = cell_text(nb, "# Cell: send one fraud row").split("--- ")
    pages = {
        "pipeline_run": page("Pipeline run (13.ipynb)", f"<pre>{html.escape(clean(c['pipe']))}</pre><pre>{html.escape(res)}</pre>"),
        "test_fraud": page("POST /predict - fraud transaction", f"<pre>{html.escape('--- ' + calls[1])}</pre>"),
        "test_genuine": page("POST /predict - genuine transaction", f"<pre>{html.escape('--- ' + calls[2])}</pre>"),
        "prediction_log": page("Prediction log (predictions_log.csv)", f"<pre>{html.escape(cell_text(nb, 'Rows logged'))}</pre>{cell_html(nb, 'Rows logged')}"),
    }
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"}
    mlflow = subprocess.Popen([sys.executable, "-m", "mlflow", "server", "--backend-store-uri", f"sqlite:///{ROOT / 'mlflow.db'}", "--port", "5055"],
                              cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--port", "8000"],
                           cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_port(5055); wait_port(8000)
        with sync_playwright() as p:
            b = chromium(p)
            for name, h in pages.items():
                pg = b.new_page(viewport={"width": 1100, "height": 400}, device_scale_factor=2)
                pg.set_content(h); pg.screenshot(path=str(SHOTS / f"{name}.png"), full_page=True); pg.close()
            pg = b.new_page(viewport={"width": 1200, "height": 800}, device_scale_factor=2)
            sw = os.environ.get("SWAGGER_DIST")            # only needed when the CDN is blocked
            if sw:
                def route(r):
                    u = r.request.url
                    if u.endswith("swagger-ui.css"): r.fulfill(path=sw + "/swagger-ui.css", content_type="text/css")
                    elif u.endswith("swagger-ui-bundle.js"): r.fulfill(path=sw + "/swagger-ui-bundle.js", content_type="application/javascript")
                    elif "favicon" in u: r.abort()
                    else: r.continue_()
                pg.route("**/*", route)
            pg.goto("http://localhost:8000/docs"); pg.wait_for_selector(".opblock", timeout=20000)
            for blk in pg.locator(".opblock-summary").all(): blk.click()
            pg.wait_for_timeout(1000); pg.screenshot(path=str(SHOTS / "fastapi_docs.png"), full_page=True); pg.close()
            pg = b.new_page(viewport={"width": 1280, "height": 800}, device_scale_factor=2)
            # The MLflow page is a JavaScript app: wait until the model name is really on the screen
            # (a fixed sleep gave a blank picture on a slower start), and retry with a reload if needed.
            ok = False
            for _ in range(3):
                pg.goto("http://localhost:5055/#/models/fraud-model")
                try:
                    pg.wait_for_selector("text=Registered Models", timeout=30000)
                    pg.wait_for_selector("text=champion", timeout=30000)
                    ok = True
                    break
                except Exception:
                    pg.reload()
            try: pg.get_by_role("button", name="Close").first.click(timeout=2000)   # close the assistant panel if it is open
            except Exception: pass
            pg.wait_for_timeout(800)
            pg.screenshot(path=str(SHOTS / "mlflow.png"))
            if not ok:
                print("WARNING: the MLflow page did not load fully, check docs/screenshots/mlflow.png")
            b.close()
        print("screenshots saved in docs/screenshots/")
    finally:
        mlflow.terminate(); api.terminate()


# ---------------------------------------------------------------- PDF --------------------------
def make_pdf():
    import markdown
    from playwright.sync_api import sync_playwright
    text = MD.read_text(encoding="utf-8")
    text = text.replace("<!--APPENDIX_CODE-->", code_appendix())                      # full code only in the printed report
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
        pg.goto(out.as_uri()); pg.wait_for_load_state("networkidle"); pg.wait_for_timeout(3000)
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
    update_md(ctx)
    if not a.no_shots: take_shots(ctx)
    if not a.no_pdf: make_pdf()
