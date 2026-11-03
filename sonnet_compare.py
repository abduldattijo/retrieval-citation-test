"""Compare Sonnet 5.5 with Haiku 5.5 on the same 40 claims and the same 4 shuffles (orders 0 to 3). Descriptive, no pass mark. No model calls."""
import json, numpy as np
from collections import defaultdict
from scipy.stats import spearmanr
SUB = json.load(open("data/sonnet_subset.json"))["claim_ids"]; ORD = range(4)
P = {c["claim_id"]: c for c in json.load(open("data/pools.json"))["claims"]}
def load(path):
    return [r for r in map(json.loads, open(path)) if r["claim_id"] in SUB and r["order_index"] in ORD and r["reply"].strip() != "" and not r["truncated"]]
def summarize(rows, name, nboot=10000):
    by = defaultdict(list)
    for r in rows: by[r["claim_id"]].append(r)
    claims = sorted(by); never, relf, unlf, pos, nans, rho = [], [], [], np.zeros((len(claims), 10)), np.zeros(len(claims)), []
    for ci, c in enumerate(claims):
        cnt = defaultdict(int)
        for r in by[c]:
            for p in r["cited"]: cnt[r["shown_doc_ids"][p - 1]] += 1; pos[ci, p - 1] += 1
        nans[ci] = len(by[c]); pool = P[c]["pool"]; f = np.array([cnt[p["doc_id"]] / len(by[c]) for p in pool]); rel = np.array([p["relevant"] for p in pool])
        never.append(bool((f[rel] == 0).any())); relf.append(f[rel].mean()); unlf.append(f[~rel].mean()); rho.append(spearmanr(rel.astype(float), f)[0] if f.std() > 0 else np.nan)
    never = np.array(never); idx = np.random.default_rng(12345).integers(0, len(claims), size=(nboot, len(claims)))
    lo, hi = np.percentile(never[idx].mean(1), [2.5, 97.5]); rate = pos.sum(0) / nans.sum()
    v = {k: sum(r["verdict"] == k for r in rows) for k in ("SUPPORTED", "REFUTED", "NOT SETTLED")}
    out = dict(model=name, claims=len(claims), answers=len(rows), never_share=float(never.mean()), never_ci=[float(lo), float(hi)], cite_relevant=float(np.mean(relf)), cite_unlabeled=float(np.mean(unlf)),
               spearman=float(np.nanmean(rho)), position_spread_pts=float((rate.max() - rate.min()) * 100), position_pct=[round(float(x) * 100, 1) for x in rate], verdicts=v, cited_per_answer=float(np.mean([len(r["cited"]) for r in rows])),
               usd=float(sum(r["usd"] for r in rows)), never_claims=[c for c, n in zip(claims, never) if n])
    return out
H = summarize(load("results/main_haiku.jsonl"), "haiku-5.5 (same 40 claims, orders 0 to 3)"); S = summarize(load("results/sonnet_gen.jsonl"), "sonnet-5.5")
for o in (H, S): print(json.dumps(o, indent=1))
json.dump(dict(haiku_subset=H, sonnet=S), open("results/summary_sonnet_check.json", "w"), indent=1)
# paired: per answer, do the two models cite the same passages for the same claim and shuffle?
hh = {(r["claim_id"], r["order_index"]): r for r in load("results/main_haiku.jsonl")}; ss = {(r["claim_id"], r["order_index"]): r for r in load("results/sonnet_gen.jsonl")}
common = sorted(set(hh) & set(ss)); same = sum(set(hh[k]["cited"]) == set(ss[k]["cited"]) for k in common)
jac = np.mean([len(set(hh[k]["cited"]) & set(ss[k]["cited"])) / max(1, len(set(hh[k]["cited"]) | set(ss[k]["cited"]))) for k in common])
vsame = sum(hh[k]["verdict"] == ss[k]["verdict"] for k in common)
print(f"\npaired answers {len(common)}: identical cited sets {same} ({same/len(common)*100:.1f} percent), mean overlap (Jaccard) {jac:.2f}, same verdict {vsame} ({vsame/len(common)*100:.1f} percent)")
