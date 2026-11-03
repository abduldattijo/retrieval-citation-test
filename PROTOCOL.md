# Protocol and change log

Written before any model call, then updated with dated entries. Dates are 2026-11-03.

## Question
When a RAG generator is given a fixed set of passages in random order, does a passage that annotators marked relevant get cited, and which passages get cited instead? Scope is one prompt, Claude Haiku 5.5 (plus a small Sonnet 5.5 check), SciFact claims, pools of 10.

## Design (frozen before the pilot)
- 130 claims drawn at random with seed 20261103 from the 300 SciFact test claims, 10 for a pilot and 120 for the main run. Pools of 10 abstracts: every labeled relevant abstract, filled up with the best ranked unlabeled abstracts from BGE base retrieval (ranks 1 to 30). Pools written to `data/pools.json` before any generation.
- Prompt v1 in `prompt.py`: verdict supported, refuted or not settled, at most three sentences, cite passages by number. No temperature argument.
- 10 random orders per claim, seeded by (seed, claim id, order index).
- Measures: claims with a labeled relevant passage never cited, citation frequency, position effect, within claim Spearman correlations, with bootstrap over claims (10,000 resamples, seed 12345).
- Pass mark: share never cited at most 5 percent and the upper end of its 95 percent range below 10 percent. Position effect material if the best position beats the worst by at least 10 points.
- Judge audits with Claude Sonnet 5.5: audit R (does the passage bear on the claim) on every labeled relevant passage, every unlabeled passage cited in at least half the orders, and a seeded sample of 129 rarely cited unlabeled ones. Audit C (does the cited passage support its sentence) on 300 seeded citation instances, 100 per group.
- Results true by construction, named in advance: labeled relevant passages are the ones with evidence, so a high citation rate for them is partly expected. Unlabeled does not mean irrelevant.

## Log
- Prices checked on the Anthropic pricing page: Haiku 5.5 0.10 and 0.50 US dollars per million input and output tokens, Sonnet 5.5 2 and 10.
- Pools built: 130 claims, 1.1 relevant passages per claim on average, 143 relevant passages in all pools, 127 of them in the retrieval top 10.
- Smoke call worked. Output limit raised from 300 to 450 tokens before the pilot, because one reply used 245 tokens.
- Pilot, 10 claims by 10 orders, 100 calls, 0.067 US dollars. The effect under test showed up, so the full run went ahead. Pilot numbers were shown to the author before the full run.
- Main run, 1,200 calls, 0.829 US dollars. The analysis script stopped on 31 error rows before any citation result was read. 30 were empty replies with 0 output tokens, and a direct re-test showed the API refusing with a biology safety category. One reply hit the 450 token limit.
- Rule added after seeing only those error counts: refused and truncated answers are errors and are excluded and reported. A claim is analyzed only if at least 8 of its 10 orders gave a usable answer. Claims 380, 823 and 1021 were excluded, leaving 117 claims and 1,168 answers.
- Judge run. 657 of 675 calls finished before the API account balance ran out. The remaining 18 were run after a top up, with the same prompts and settings.
- Hand review: 46 items (28 relevance, 18 citation support) labeled blind to the judge by the author, who confirmed the labels were their own.
- Sonnet check: 40 claims drawn with seed 20261105 from the 117, orders 0 to 3 only. The plan was 5 orders, but 5 would have passed the 5.00 US dollar cap, so it ran 4. Same shuffles as Haiku, so the comparison is paired.
- Total spend 4.83 US dollars of the 5.00 cap.
