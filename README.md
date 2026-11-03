# Does a relevant source actually get cited?

Code, frozen pools, saved answers, judge labels and a hand review for a small test of which passages an AI model cites when it answers from a fixed set of passages in random order. The claims come from SciFact, and the answering models are Claude Haiku 5.5 (main run) and Claude Sonnet 5.5 (second model check).

## The pass mark, set before the first scored run

For each claim the model sees ten passages (every passage that annotators labeled relevant, filled up with the closest unlabeled abstracts) in ten different random orders. Citation frequency is the share of a passage's orders in which it is cited.

- Relevance counts as a reliable stand in for use only if the share of claims with a labeled relevant passage that is never cited is at most 5 percent and the upper end of its 95 percent bootstrap range (10,000 resamples over claims) is below 10 percent.
- A position effect is material if the best position beats the worst by at least 10 points in citation rate.

The full design and a dated log of every change are in `PROTOCOL.md`.

## Results in short

| Measure (Claude Haiku 5.5, 117 claims, 1,168 answers) | Result |
|---|---|
| Claims with a labeled relevant passage never cited | 3 of 117 (2.6 percent, range 0.0 to 6.0), so the pass mark is met |
| Citation frequency, labeled relevant vs unlabeled passages | 0.86 vs 0.14 |
| Labeled relevant passages cited in fewer than half the orders | 18 of 129 |
| Unlabeled passages cited in at least half the orders | 117 of 1,041 |
| Position effect (best minus worst position) | 5.7 points (range 4.0 to 9.4), so not material |
| Rank agreement with relevance (Spearman), within claim | 0.61 (range 0.55 to 0.66) |

An AI judge (Claude Sonnet 5.5) found evidence in 66.7 percent of labeled relevant passages and 38.5 percent of heavily cited unlabeled ones. The author read 46 judged items blind to the judge's labels. Agreement on evidence or not was 26 of 28 for the relevance audit, and exact agreement was 20 of 28 (kappa 0.60). For citation support, exact agreement was 12 of 18 (kappa 0.34), so those numbers are rough. Sonnet 5.5 as the answering model, on 40 claims and 4 orders, showed the same pattern.

Thirty answers (3 claims, all about viruses) were refused by the API with a biology safety category and were excluded, as was one answer that hit the length limit.

## Files

| Path | What it is |
|---|---|
| `prepare_data.py`, `embed.py` | Download SciFact test data and embed it with BGE base (used only to pick the unlabeled abstracts) |
| `build_pools.py` | Builds the frozen pools. The public copy of `data/pools.json` keeps ids, labels and ranks, with no abstract text |
| `prompt.py`, `parse_citations.py`, `test_parse.py` | The frozen prompt and the citation parser with tests |
| `run_generation.py`, `clients.py` | Runs the answering model under a spend cap (trimmed `clients.py`, key from `ANTHROPIC_API_KEY`) |
| `analyze.py`, `recompute.py` | Main analysis, and an independent recomputation with separate code |
| `audit_prompts.py`, `audit_build.py`, `audit_run.py`, `audit_analyze.py` | The two judge audits (does the passage bear on the claim, and does the cited passage support its sentence) |
| `make_hand_review.py`, `hand_review_analyze.py` | The blind hand review sheet and the comparison with the judge |
| `sonnet_compare.py`, `make_charts.py` | The second model comparison and the figure |
| `results/` | Saved answers (`main_haiku.jsonl`, `pilot.jsonl`, `sonnet_gen.jsonl`), judge labels (`audit_sonnet.jsonl`), the hand review, summaries and the spend ledger |

## Running it

```
pip install -r requirements.txt
python prepare_data.py && python embed.py
python build_pools.py        # sha256 of the full data/pools.json starts with f5614f33372312b5 if your embeddings match mine; small differences in embeddings can change the filler abstracts
export ANTHROPIC_API_KEY=...
python run_generation.py --split pilot --out results/pilot.jsonl
python run_generation.py --split main --out results/main_haiku.jsonl
python analyze.py results/main_haiku.jsonl
python audit_build.py        # sha256 of data/audit_items.json should start with 854415ab1c7f0978
python audit_run.py --out results/audit_sonnet.jsonl
```

API calls cost money. The whole project cost about 4.8 US dollars: 0.07 for the pilot, 0.83 for the main run, 1.81 for the judge, 2.11 for the Sonnet check.

## Rights and license notes

SciFact claims and annotations are released under CC BY 4.0 and its abstracts under ODC-By 1.0 (https://github.com/allenai/scifact/blob/master/LICENSE.md). Commercial use is allowed with credit. This public copy leaves out the abstract text. Run `prepare_data.py` and `build_pools.py` to get it back. The Hugging Face copy of BEIR labels SciFact CC BY SA 4.0, which differs from the original terms. BGE base is MIT licensed. No license file is chosen for the code yet.

## Limits

One prompt, two models from one company, one set of scientific claims and pools of ten passages built from embedding search. Labeled relevant passages are the ones annotators judged to hold evidence, so high citation of them is partly expected. The judge is another model and the author read only 46 of 675 judged items. The Sonnet check is small and used 4 orders and not 5 because of the budget cap. Nothing here says how a live web search engine picks sources.
