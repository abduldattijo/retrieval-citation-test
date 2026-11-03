"""Build audit items from the Haiku run. No model calls. Writes data/audit_items.json."""
import json, re, hashlib, numpy as np
from collections import defaultdict
SEED = 20261104
rows = [json.loads(l) for l in open("results/main_haiku.jsonl")]
good = [r for r in rows if r["reply"].strip() != "" and not r["truncated"]]
P = {c["claim_id"]: c for c in json.load(open("data/pools.json"))["claims"]}
by = defaultdict(list)
for r in good: by[r["claim_id"]].append(r)
kept = sorted(c for c in by if len(by[c]) >= 8)
freq = {}
for c in kept:
    cnt = defaultdict(int)
    for r in by[c]:
        for pos in r["cited"]: cnt[r["shown_doc_ids"][pos - 1]] += 1
    for p in P[c]["pool"]: freq[(c, p["doc_id"])] = cnt[p["doc_id"]] / len(by[c])
pas = {(c, p["doc_id"]): p for c in kept for p in P[c]["pool"]}
rng = np.random.default_rng(SEED)
rel = [k for k in pas if pas[k]["relevant"]]
high = [k for k in pas if not pas[k]["relevant"] and freq[k] >= 0.5]
low = sorted(k for k in pas if not pas[k]["relevant"] and freq[k] < 0.5)
low_s = [low[i] for i in rng.choice(len(low), size=129, replace=False)]
R = [dict(kind="R", group=g, claim_id=k[0], doc_id=k[1], claim=P[k[0]]["claim"], title=pas[k]["title"], text=pas[k]["text"], freq=freq[k], retrieval_rank=pas[k]["retrieval_rank"])
     for g, ks in (("relevant", rel), ("unlabeled_high", high), ("unlabeled_low_sample", low_s)) for k in ks]
def sentences(reply):
    body = "\n".join(reply.strip().split("\n")[1:]).strip()
    parts = [s for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
    merged = []
    for s in parts:                      # a sentence that is only citation marks belongs to the previous sentence
        if re.fullmatch(r"(\s*\[[\d,\s\-–]+\]\s*)+", s) and merged: merged[-1] += " " + s
        else: merged.append(s)
    return merged
inst = defaultdict(list)
for c in kept:
    for r in by[c]:
        for s in sentences(r["reply"]):
            for m in re.finditer(r"\[([\d,\s\-–]+)\]", s):
                for part in re.split(r"[,\s]+", m.group(1).strip()):
                    if part.isdigit() and 1 <= int(part) <= 10:
                        d = r["shown_doc_ids"][int(part) - 1]; k = (c, d); g = "relevant" if pas[k]["relevant"] else ("unlabeled_high" if freq[k] >= 0.5 else "unlabeled_low")
                        inst[g].append(dict(claim_id=c, order_index=r["order_index"], doc_id=d, sentence=s.strip(), title=pas[k]["title"], text=pas[k]["text"]))
C = []
for g in ("relevant", "unlabeled_high", "unlabeled_low"):
    pool = sorted({json.dumps(x, sort_keys=True) for x in inst[g]}); pick = rng.choice(len(pool), size=min(100, len(pool)), replace=False)
    C += [dict(kind="C", group=g, **json.loads(pool[i])) for i in pick]
json.dump(dict(seed=SEED, R=R, C=C), open("data/audit_items.json", "w"), indent=1)
print("sha256", hashlib.sha256(open("data/audit_items.json", "rb").read()).hexdigest()[:16])
print("R items:", {g: sum(x["group"] == g for x in R) for g in ("relevant", "unlabeled_high", "unlabeled_low_sample")}, "| C items:", {g: sum(x["group"] == g for x in C) for g in ("relevant", "unlabeled_high", "unlabeled_low")})
print("citation instances available:", {g: len(v) for g, v in inst.items()})
chars = [len(x["title"]) + len(x["text"]) + len(x.get("claim", x.get("sentence", ""))) for x in R + C]; print("mean item chars", int(np.mean(chars)), "-> est input tokens per call", int(np.mean(chars) / 3.5) + 150)
