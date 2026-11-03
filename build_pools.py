"""Build the frozen candidate pools. No model calls. Writes data/pools.json and prints its hash.
Pool per claim: every labeled relevant abstract, filled to POOL_SIZE with the best ranked unlabeled abstracts (bge-base ranks 1 to 30)."""
import csv, json, hashlib, numpy as np
SEED, N_PILOT, N_MAIN, POOL_SIZE, FILL_DEPTH = 20261103, 10, 120, 10, 30
docs = [json.loads(l) for l in open("data/corpus.jsonl")]; qs = [json.loads(l) for l in open("data/queries.jsonl")]
ids = [d["id"] for d in docs]; idx = {d: i for i, d in enumerate(ids)}
qrels = {}
for r in csv.DictReader(open("data/qrels_test.tsv"), delimiter="\t"):
    if int(r["score"]) > 0: qrels.setdefault(r["query-id"], []).append(r["corpus-id"])
D = np.load("data/docs_base.npy"); Q = np.load("data/queries_base.npy"); assert D.shape[0] == len(docs) and Q.shape[0] == len(qs)
S = Q @ D.T
order = np.random.default_rng(SEED).permutation(len(qs))[: N_PILOT + N_MAIN]
pools = []
for pos, qi in enumerate(order):
    q = qs[qi]; rel = [d for d in qrels[q["id"]]]; assert 1 <= len(rel) <= POOL_SIZE
    rank_order = np.argsort(-S[qi], kind="stable"); rank_of = {int(d): r + 1 for r, d in enumerate(rank_order)}
    pool = [dict(doc_id=d, relevant=True) for d in rel]
    for d in rank_order[:FILL_DEPTH]:
        if len(pool) >= POOL_SIZE: break
        if ids[d] not in rel: pool.append(dict(doc_id=ids[d], relevant=False))
    assert len(pool) == POOL_SIZE, (q["id"], len(pool))
    for p in pool:
        i = idx[p["doc_id"]]; p.update(title=docs[i]["title"], text=docs[i]["text"], retrieval_rank=rank_of[i], cosine=round(float(S[qi, i]), 4))
    pools.append(dict(claim_id=q["id"], claim=q["text"], split="pilot" if pos < N_PILOT else "main", pool=pool))
json.dump(dict(seed=SEED, pool_size=POOL_SIZE, fill_depth=FILL_DEPTH, claims=pools), open("data/pools.json", "w"), indent=1)
print("sha256", hashlib.sha256(open("data/pools.json", "rb").read()).hexdigest()[:16])
nrel = [sum(p["relevant"] for p in c["pool"]) for c in pools]; chars = [sum(len(p["title"]) + len(p["text"]) for p in c["pool"]) for c in pools]
print("claims", len(pools), "pilot", sum(c["split"] == "pilot" for c in pools), "main", sum(c["split"] == "main" for c in pools))
print("relevant per claim: mean", round(np.mean(nrel), 2), "max", max(nrel), "| claims with 1 relevant", sum(n == 1 for n in nrel))
print("relevant passages ranked top 10 by retrieval:", sum(p["relevant"] and p["retrieval_rank"] <= 10 for c in pools for p in c["pool"]), "of", sum(nrel))
print("pool characters: mean", int(np.mean(chars)), "max", max(chars), "-> about", int(np.mean(chars) / 4), "tokens per call")
