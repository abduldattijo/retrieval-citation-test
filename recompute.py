"""Independent recomputation of every headline number from the raw saved files. Does NOT import analyze.py or scipy. Prints a dict used by check_draft.py."""
import json, csv, numpy as np
from collections import Counter, defaultdict
def avg_rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]: j += 1
        for k in range(i, j + 1): r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    return r
def spearman(a, b):
    ra, rb = avg_rank(a), avg_rank(b); ma, mb = np.mean(ra), np.mean(rb); da, db = np.array(ra) - ma, np.array(rb) - mb
    d = np.sqrt((da ** 2).sum() * (db ** 2).sum()); return float((da * db).sum() / d) if d > 0 else float("nan")
P = {c["claim_id"]: c for c in json.load(open("data/pools.json"))["claims"]}
def load(path): return [json.loads(l) for l in open(path)]
def usable(rows): return [r for r in rows if r["reply"].strip() != "" and not r["truncated"]]
def claim_stats(rows, claims):
    out = {}
    for c in claims:
        rs = [r for r in rows if r["claim_id"] == c]; docs = [p["doc_id"] for p in P[c]["pool"]]; rel = [p["relevant"] for p in P[c]["pool"]]; rank = [p["retrieval_rank"] for p in P[c]["pool"]]
        cnt = Counter(r["shown_doc_ids"][k - 1] for r in rs for k in set(r["cited"])); f = [cnt[d] / len(rs) for d in docs]
        out[c] = dict(f=f, rel=rel, rank=rank, n=len(rs))
    return out
def headline(rows, claims, nboot=10000):
    cs = claim_stats(rows, claims); never = np.array([any(f == 0 for f, r in zip(v["f"], v["rel"]) if r) for v in cs.values()])
    idx = np.random.default_rng(12345).integers(0, len(claims), size=(nboot, len(claims)))
    relf = [f for v in cs.values() for f, r in zip(v["f"], v["rel"]) if r]; unlf = [f for v in cs.values() for f, r in zip(v["f"], v["rel"]) if not r]
    pos = np.zeros(10); n = 0
    for r in rows:
        if r["claim_id"] in cs:
            n += 1
            for k in set(r["cited"]): pos[k - 1] += 1
    per_claim_pos = np.array([[sum(k in r["cited"] for r in rows if r["claim_id"] == c) if False else 0 for k in range(1, 11)] for c in claims])  # placeholder, replaced below
    pc = np.zeros((len(claims), 10)); na = np.zeros(len(claims))
    for ci, c in enumerate(claims):
        for r in rows:
            if r["claim_id"] == c:
                na[ci] += 1
                for k in set(r["cited"]): pc[ci, k - 1] += 1
    rate = pc.sum(0) / na.sum(); br = pc[idx].sum(1) / na[idx].sum(1)[:, None]; sp = br.max(1) - br.min(1)
    rho_rel = [spearman([float(x) for x in v["rel"]], v["f"]) for v in cs.values()]; rho_rank = [spearman([-x for x in v["rank"]], v["f"]) for v in cs.values()]
    return dict(claims=len(claims), answers=int(na.sum()), never_n=int(never.sum()), never_share=float(never.mean()), never_ci=[float(x) for x in np.percentile(never[idx].mean(1), [2.5, 97.5])],
                cite_rel=float(np.mean([np.mean([f for f, r in zip(v["f"], v["rel"]) if r]) for v in cs.values()])), cite_unl=float(np.mean([np.mean([f for f, r in zip(v["f"], v["rel"]) if not r]) for v in cs.values()])),
                rel_under_half=int(sum(f < 0.5 for f in relf)), n_rel=len(relf), unl_half_or_more=int(sum(f >= 0.5 for f in unlf)), n_unl=len(unlf),
                pos_rate=[round(float(x) * 100, 1) for x in rate], spread=float((rate.max() - rate.min()) * 100), spread_ci=[float(x * 100) for x in np.percentile(sp, [2.5, 97.5])],
                rho_rel=float(np.nanmean(rho_rel)), rho_rank=float(np.nanmean(rho_rank)), rho_rel_ci=[float(x) for x in np.percentile([np.nanmean(np.array(rho_rel)[i]) for i in idx], [2.5, 97.5])],
                verdicts={k: sum(r["verdict"] == k for r in rows if r["claim_id"] in cs) for k in ("SUPPORTED", "REFUTED", "NOT SETTLED")}, per_answer=float(np.mean([len(set(r["cited"])) for r in rows if r["claim_id"] in cs])))
H = load("results/main_haiku.jsonl"); G = usable(H); cnt = Counter(r["claim_id"] for r in G); kept = sorted(c for c in cnt if cnt[c] >= 8)
R = dict(calls=len(H), refused=sum(r["reply"].strip() == "" and r["tokens_out"] == 0 for r in H), truncated=sum(r["truncated"] for r in H), cost_haiku=round(sum(r["usd"] for r in H), 3),
         pilot_cost=round(sum(r["usd"] for r in load("results/pilot.jsonl")), 3), excluded=sorted(set(r["claim_id"] for r in H) - set(kept)), main=headline([r for r in G if r["claim_id"] in kept], kept))
# audit
A = load("results/audit_sonnet.jsonl"); R["audit_calls"] = len(A); R["audit_cost"] = round(sum(r["usd"] for r in A), 2)
R["audit"] = {f"{k}_{g}": dict(n=len(rs), counts=dict(Counter(r["label"] for r in rs))) for k, g in sorted({(r["kind"], r["group"]) for r in A}) for rs in [[r for r in A if r["kind"] == k and r["group"] == g]]}
# hand review
sheet = {r["review_id"]: r for r in csv.DictReader(open("results/hand_review_sheet.csv"))}; key = {r["review_id"]: r for r in csv.DictReader(open("results/hand_review_key.csv"))}
hr = {}
for kind in ("R", "C"):
    ids = [i for i in sheet if key[i]["kind"] == kind]; h = [sheet[i]["YOUR_LABEL"].strip().upper() for i in ids]; j = [key[i]["judge_label"] for i in ids]
    yes = ("SUPPORTS", "CONTRADICTS") if kind == "R" else ("SUPPORTS",)
    hr[kind] = dict(n=len(ids), exact=sum(a == b for a, b in zip(h, j)), yesno=sum((a in yes) == (b in yes) for a, b in zip(h, j)), human_yes=sum(a in yes for a in h), judge_yes=sum(b in yes for b in j))
    po = hr[kind]["exact"] / len(ids); ch, cj = Counter(h), Counter(j); pe = sum(ch[l] * cj[l] for l in set(ch) | set(cj)) / len(ids) ** 2; hr[kind]["kappa"] = round((po - pe) / (1 - pe), 2)
R["hand"] = hr
# sonnet
sub = json.load(open("data/sonnet_subset.json"))["claim_ids"]; S = usable(load("results/sonnet_gen.jsonl"))
Hs = [r for r in G if r["claim_id"] in sub and r["order_index"] < 4]
R["sonnet"] = dict(cost=round(sum(r["usd"] for r in S), 2), haiku=headline(Hs, sorted(sub)), sonnet=headline(S, sorted(sub)))
hh = {(r["claim_id"], r["order_index"]): r for r in Hs}; ss = {(r["claim_id"], r["order_index"]): r for r in S}; com = sorted(set(hh) & set(ss))
R["sonnet"]["paired"] = dict(n=len(com), same_set=sum(set(hh[k]["cited"]) == set(ss[k]["cited"]) for k in com), same_verdict=sum(hh[k]["verdict"] == ss[k]["verdict"] for k in com))
R["total_cost"] = round(json.load(open("results/ledger.json"))["total_usd"], 2)
json.dump(R, open("results/recomputed.json", "w"), indent=1); print(json.dumps(R, indent=1)[:3500])
