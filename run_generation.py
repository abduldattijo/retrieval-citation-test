"""Run the generator over frozen pools. usage: python run_generation.py --split pilot|main --model claude-haiku-5-5 --orders 10 --out results/pilot.jsonl [--limit-claims N] [--limit-orders K]
Resumes: (claim, order) pairs already in the output file are skipped. Replies that hit the token limit are stored with truncated=true and counted as errors by analysis."""
import argparse, json, threading, numpy as np
from concurrent.futures import ThreadPoolExecutor
from clients import Ledger, Chooser
from prompt import PROMPT_VERSION, SYSTEM, user_message
from parse_citations import parse
ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True); ap.add_argument("--model", default="claude-haiku-5-5")
ap.add_argument("--orders", type=int, default=10); ap.add_argument("--out", required=True); ap.add_argument("--cap", type=float, default=5.0)
ap.add_argument("--claims-file", default=None); ap.add_argument("--limit-claims", type=int, default=None); ap.add_argument("--limit-orders", type=int, default=None); ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()
MAX_TOKENS = 450
P = json.load(open("data/pools.json")); claims = [c for c in P["claims"] if c["split"] == a.split]
if a.claims_file: ids = set(json.load(open(a.claims_file))["claim_ids"]); claims = [c for c in claims if c["claim_id"] in ids]
claims = claims[: a.limit_claims]
done = set()
try:
    for l in open(a.out): r = json.loads(l); done.add((r["claim_id"], r["order_index"]))
except FileNotFoundError: pass
jobs = [(c, k) for c in claims for k in range(a.limit_orders or a.orders) if (c["claim_id"], k) not in done]
ledger = Ledger(a.cap, "results/ledger.json"); chooser = Chooser(a.model, ledger, max_tokens=MAX_TOKENS); lock = threading.Lock()
print(f"{len(jobs)} calls to run, project spend so far {ledger.total:.4f} USD, cap {a.cap:.2f}", flush=True)
def work(job):
    c, k = job
    perm = np.random.default_rng([P["seed"], int(c["claim_id"]), k]).permutation(len(c["pool"]))
    shown = [c["pool"][i] for i in perm]
    text, rec = chooser.choose(SYSTEM, user_message(c["claim"], [(p["title"], p["text"]) for p in shown]), max_tokens=MAX_TOKENS)
    parsed = parse(text, len(shown))
    row = dict(claim_id=c["claim_id"], order_index=k, model=a.model, prompt_version=PROMPT_VERSION, shown_doc_ids=[p["doc_id"] for p in shown],
               shown_relevant=[p["relevant"] for p in shown], reply=text, truncated=rec["out"] >= MAX_TOKENS, tokens_in=rec["in"], tokens_out=rec["out"], usd=rec["usd"], **parsed)
    with lock:
        with open(a.out, "a") as f: f.write(json.dumps(row) + "\n")
    return row
with ThreadPoolExecutor(a.workers) as ex: rows = list(ex.map(work, jobs))
print(f"done {len(rows)} calls, this run {sum(r['usd'] for r in rows):.4f} USD, project total {ledger.total:.4f} USD, truncated {sum(r['truncated'] for r in rows)}, no verdict {sum(r['verdict'] is None for r in rows)}")
