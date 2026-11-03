"""Download the SciFact test split from the BeIR repositories on Hugging Face into data/. SciFact: claims and labels CC BY 4.0, abstracts ODC-By 1.0."""
import csv, json, os, pandas as pd
from huggingface_hub import hf_hub_download
os.makedirs("data", exist_ok=True)
corpus = pd.read_parquet(hf_hub_download("BeIR/scifact", "corpus/corpus-00000-of-00001.parquet", repo_type="dataset"))
queries = pd.read_parquet(hf_hub_download("BeIR/scifact", "queries/queries-00000-of-00001.parquet", repo_type="dataset"))
qrels = pd.read_csv(hf_hub_download("BeIR/scifact-qrels", "test.tsv", repo_type="dataset"), sep="\t"); qrels.columns = ["query-id", "corpus-id", "score"]
qrels["query-id"] = qrels["query-id"].astype(str); qrels["corpus-id"] = qrels["corpus-id"].astype(str)
queries["_id"] = queries["_id"].astype(str); queries = queries[queries["_id"].isin(set(qrels["query-id"]))]
with open("data/corpus.jsonl", "w") as f:
    for r in corpus.itertuples(): f.write(json.dumps({"id": str(r._1), "title": r.title, "text": r.text}) + "\n")
with open("data/queries.jsonl", "w") as f:
    for r in queries.itertuples(): f.write(json.dumps({"id": r._1, "text": r.text}) + "\n")
qrels.to_csv("data/qrels_test.tsv", sep="\t", index=False); print("docs", len(corpus), "test claims", len(queries), "labeled pairs", len(qrels))
