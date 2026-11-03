"""Summarize the Sonnet audit. usage: python audit_analyze.py results/audit_sonnet.jsonl [--summary results/summary_audit.json]
Stops if any row is a refusal, truncated or has no label (those must be looked at, not hidden)."""
import sys, json, numpy as np
from collections import Counter, defaultdict
rows = [json.loads(l) for l in open(sys.argv[1])]
bad = [r for r in rows if r["refusal"] or r["truncated"] or r["label"] is None or r["usd"] <= 0]
if bad: sys.exit(f"STOP: {len(bad)} judge error rows (refusal {sum(r['refusal'] for r in bad)}, truncated {sum(r['truncated'] for r in bad)}, no label {sum(r['label'] is None for r in bad)})")
rng = np.random.default_rng(12345); S = {}
def boot_share(flags, n=10000):
    f = np.array(flags, float); i = rng.integers(0, len(f), size=(n, len(f))); return [float(x) for x in np.percentile(f[i].mean(1), [2.5, 97.5])]
print(f"{len(rows)} judge calls, {sum(r['usd'] for r in rows):.4f} USD")
for kind, labs, good in (("R", ("SUPPORTS", "CONTRADICTS", "RELATED NOT EVIDENCE", "UNRELATED"), ("SUPPORTS", "CONTRADICTS")), ("C", ("SUPPORTS", "PARTLY", "DOES NOT SUPPORT"), ("SUPPORTS",))):
    print(f"\nAudit {kind}: " + ("does the passage bear on the claim (evidence = supports or contradicts)" if kind == "R" else "does the cited passage support its sentence (clean support = supports)"))
    for g in sorted({r["group"] for r in rows if r["kind"] == kind}):
        rs = [r for r in rows if r["kind"] == kind and r["group"] == g]; c = Counter(r["label"] for r in rs); flags = [r["label"] in good for r in rs]
        lo, hi = boot_share(flags); S[f"{kind}_{g}"] = dict(n=len(rs), counts={k: c.get(k, 0) for k in labs}, share_good=float(np.mean(flags)), ci95=[lo, hi])
        print(f"  {g:22s} n={len(rs):3d} " + " ".join(f"{k}={c.get(k,0)}" for k in labs) + f" | {'evidence' if kind=='R' else 'clean support'} {np.mean(flags)*100:.1f} percent (95 percent range {lo*100:.1f} to {hi*100:.1f})")
if "--summary" in sys.argv: json.dump(S, open(sys.argv[sys.argv.index("--summary") + 1], "w"), indent=1)
