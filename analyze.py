"""Analysis of a generation file. usage: python analyze.py results/main_haiku.jsonl [--summary results/summary_haiku.json]
Error handling (rule in PLAN.md): refusals (empty reply, 0 output tokens) and truncated replies are errors and are excluded and reported. A claim needs at least MIN_ANSWERED usable orders.
Any other error row (non empty reply with no verdict, invalid citation numbers, zero cost) stops the script."""
import sys, json, numpy as np
from collections import defaultdict
from scipy.stats import spearmanr
MIN_ANSWERED, NBOOT, BSEED = 8, 10000, 12345
rows = [json.loads(l) for l in open(sys.argv[1])]
refused = [r for r in rows if r["reply"].strip() == "" and r["tokens_out"] == 0]
trunc = [r for r in rows if r["truncated"]]
other = [r for r in rows if r not in refused and r not in trunc and (r["verdict"] is None or r["invalid"] or r["usd"] <= 0)]
if other: sys.exit(f"STOP: {len(other)} unexplained error rows")
good = [r for r in rows if r not in refused and r not in trunc]
P = {c["claim_id"]: c for c in json.load(open("data/pools.json"))["claims"]}
by = defaultdict(list)
for r in good: by[r["claim_id"]].append(r)
all_claims = sorted({r["claim_id"] for r in rows}); kept = [c for c in all_claims if len(by[c]) >= MIN_ANSWERED]; dropped = [c for c in all_claims if c not in kept]
print(f"{len(rows)} calls, {sum(r['usd'] for r in rows):.4f} USD | refused {len(refused)}, truncated {len(trunc)} | claims {len(all_claims)}, analyzed {len(kept)}, excluded {len(dropped)} {dropped}")
# per claim arrays
rel_freq, unl_freq, never, pos_cnt, n_ans, rhos_rel, rhos_rank = {}, {}, [], np.zeros((len(kept), 10)), np.zeros(len(kept)), [], []
for ci, c in enumerate(kept):
    rs = by[c]; cnt = defaultdict(int)
    for r in rs:
        for pos in r["cited"]: cnt[r["shown_doc_ids"][pos - 1]] += 1; pos_cnt[ci, pos - 1] += 1
    n_ans[ci] = len(rs); pool = P[c]["pool"]; f = np.array([cnt[p["doc_id"]] / len(rs) for p in pool]); rel = np.array([p["relevant"] for p in pool]); rk = np.array([p["retrieval_rank"] for p in pool])
    rel_freq[c] = f[rel]; unl_freq[c] = f[~rel]; never.append(bool((f[rel] == 0).any()))
    rhos_rel.append(spearmanr(rel.astype(float), f)[0] if f.std() > 0 else np.nan); rhos_rank.append(spearmanr(-rk, f)[0] if f.std() > 0 else np.nan)
never = np.array(never); rhos_rel = np.array(rhos_rel); rhos_rank = np.array(rhos_rank)
rng = np.random.default_rng(BSEED); idx = rng.integers(0, len(kept), size=(NBOOT, len(kept)))
ci95 = lambda v: np.percentile(v, [2.5, 97.5])
share_never = never.mean(); lo, hi = ci95(never[idx].mean(1))
mean_rel = np.mean([f.mean() for f in rel_freq.values()]); mean_unl = np.mean([f.mean() for f in unl_freq.values()])
rate = pos_cnt.sum(0) / n_ans.sum(); boot_rate = pos_cnt[idx].sum(1) / n_ans[idx].sum(1)[:, None]; spread = rate.max() - rate.min(); bspread = boot_rate.max(1) - boot_rate.min(1)
lo_s, hi_s = ci95(bspread); nn = ~np.isnan(rhos_rel)
all_rel = np.concatenate(list(rel_freq.values())); all_unl = np.concatenate(list(unl_freq.values()))
verd = {v: sum(r["verdict"] == v for r in good if r["claim_id"] in kept) for v in ("SUPPORTED", "REFUTED", "NOT SETTLED")}
S = dict(calls=len(rows), usd=round(sum(r["usd"] for r in rows), 4), refused=len(refused), truncated=len(trunc), claims_total=len(all_claims), claims_analyzed=len(kept), excluded=dropped,
         answers_analyzed=int(n_ans.sum()), share_claims_relevant_never_cited=float(share_never), ci95_never=[float(lo), float(hi)],
         rule_relevance_reliable=bool(share_never <= 0.05 and hi < 0.10), mean_cite_freq_relevant=float(mean_rel), mean_cite_freq_unlabeled=float(mean_unl),
         relevant_cited_in_fewer_than_half=int((all_rel < 0.5).sum()), n_relevant=int(len(all_rel)), unlabeled_cited_in_at_least_half=int((all_unl >= 0.5).sum()), n_unlabeled=int(len(all_unl)),
         position_rate_pct=[round(float(x) * 100, 1) for x in rate], position_spread_points=float(spread * 100), ci95_spread=[float(lo_s * 100), float(hi_s * 100)], rule_position_material=bool(spread >= 0.10),
         spearman_relevance_mean=float(np.nanmean(rhos_rel)), ci95_spearman_relevance=[float(x) for x in ci95(np.array([np.nanmean(r[i]) for r in [rhos_rel] for i in idx]))],
         spearman_retrieval_rank_mean=float(np.nanmean(rhos_rank)), verdicts=verd, passages_cited_per_answer=float(np.mean([len(r["cited"]) for r in good if r["claim_id"] in kept])))
print(json.dumps(S, indent=1))
if "--summary" in sys.argv: json.dump(S, open(sys.argv[sys.argv.index("--summary") + 1], "w"), indent=1)
