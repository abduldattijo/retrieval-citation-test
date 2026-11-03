"""Run the Sonnet judge over data/audit_items.json. usage: python audit_run.py --out results/audit_sonnet.jsonl [--limit N]
Resumes by item index. Refusals and unparseable replies are stored as errors, not hidden."""
import argparse, json, re, threading
from concurrent.futures import ThreadPoolExecutor
from clients import Ledger, Chooser
from audit_prompts import AUDIT_VERSION, R_SYSTEM, C_SYSTEM, r_user, c_user
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--cap", type=float, default=5.0)
ap.add_argument("--model", default="claude-sonnet-5-5"); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--kind", default=None)
a = ap.parse_args()
A = json.load(open("data/audit_items.json")); items = [(i, x) for i, x in enumerate(A["R"] + A["C"]) if a.kind in (None, x["kind"])][: a.limit]
done = set()
try:
    for l in open(a.out): done.add(json.loads(l)["item_index"])
except FileNotFoundError: pass
jobs = [(i, x) for i, x in items if i not in done]
ledger = Ledger(a.cap, "results/ledger.json"); chooser = Chooser(a.model, ledger, max_tokens=250); lock = threading.Lock()
print(f"{len(jobs)} judge calls, project spend so far {ledger.total:.4f} USD, cap {a.cap:.2f}", flush=True)
LAB = {"R": ("SUPPORTS", "CONTRADICTS", "RELATED NOT EVIDENCE", "UNRELATED"), "C": ("SUPPORTS", "PARTLY", "DOES NOT SUPPORT")}
def work(job):
    i, x = job
    system, user = (R_SYSTEM, r_user(x["claim"], x["title"], x["text"])) if x["kind"] == "R" else (C_SYSTEM, c_user(x["sentence"], x["title"], x["text"]))
    text, rec = chooser.choose(system, user, max_tokens=250)
    m = re.search(r"Label:\s*(SUPPORTS|CONTRADICTS|RELATED NOT EVIDENCE|UNRELATED|PARTLY|DOES NOT SUPPORT)\b", text, re.I)
    label = m.group(1).upper() if m and m.group(1).upper() in LAB[x["kind"]] else None
    row = dict(item_index=i, kind=x["kind"], group=x["group"], claim_id=x["claim_id"], doc_id=x["doc_id"], label=label, reply=text, refusal=(text.strip() == "" and rec["out"] == 0),
               truncated=rec["out"] >= 250, audit_version=AUDIT_VERSION, model=a.model, tokens_in=rec["in"], tokens_out=rec["out"], usd=rec["usd"])
    with lock:
        with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
    return row
with ThreadPoolExecutor(a.workers) as ex: rows = list(ex.map(work, jobs))
print(f"done {len(rows)} calls, this run {sum(r['usd'] for r in rows):.4f} USD, project total {ledger.total:.4f} USD, no label {sum(r['label'] is None for r in rows)}, refusals {sum(r['refusal'] for r in rows)}")
