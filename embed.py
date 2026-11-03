"""Embed SciFact abstracts and claims with BAAI/bge-base-en-v1.5 (frozen). Writes data/docs_base.npy and data/queries_base.npy, used only to pick the unlabeled abstracts for each pool."""
import json, numpy as np, torch
from sentence_transformers import SentenceTransformer
QP = "Represent this sentence for searching relevant passages: "
docs = [json.loads(l) for l in open("data/corpus.jsonl")]; qs = [json.loads(l) for l in open("data/queries.jsonl")]
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
m = SentenceTransformer("BAAI/bge-base-en-v1.5", device=dev); m.max_seq_length = 512
texts = [(d["title"] + " " + d["text"]).strip() for d in docs]; order = np.argsort([len(t) for t in texts])
E = m.encode([texts[i] for i in order], batch_size=32, normalize_embeddings=True, convert_to_numpy=True); D = np.empty_like(E); D[order] = E
Q = m.encode([QP + q["text"] for q in qs], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
np.save("data/docs_base.npy", D.astype(np.float32)); np.save("data/queries_base.npy", Q.astype(np.float32)); print(D.shape, Q.shape, dev)
