"""Builds a blind hand review sheet (judge labels hidden) and a separate key. Seeded, no model calls."""
import json, csv, numpy as np
A = json.load(open("data/audit_items.json")); items = A["R"] + A["C"]
J = {r["item_index"]: r for r in map(json.loads, open("results/audit_sonnet.jsonl"))}
rng = np.random.default_rng(777)
def pick(kind, group, n):
    c = [i for i, x in enumerate(items) if x["kind"] == kind and x["group"] == group and i in J]
    return [c[j] for j in rng.choice(len(c), size=min(n, len(c)), replace=False)]
sel = pick("R", "unlabeled_high", 14) + pick("R", "relevant", 10) + pick("R", "unlabeled_low_sample", 4) + pick("C", "relevant", 6) + pick("C", "unlabeled_high", 6) + pick("C", "unlabeled_low", 6)
rng.shuffle(sel)
with open("results/hand_review_sheet.csv", "w", newline="") as f, open("results/hand_review_key.csv", "w", newline="") as g:
    w = csv.writer(f); k = csv.writer(g)
    w.writerow(["review_id", "question_type", "claim_or_sentence", "passage", "YOUR_LABEL", "note"]); k.writerow(["review_id", "item_index", "kind", "group", "judge_label", "judge_reason"])
    for n, i in enumerate(sel, 1):
        x = items[i]
        q = "R: does the passage bear on the claim? Labels: SUPPORTS, CONTRADICTS, RELATED NOT EVIDENCE, UNRELATED" if x["kind"] == "R" else "C: does the cited passage support the sentence? Labels: SUPPORTS, PARTLY, DOES NOT SUPPORT"
        w.writerow([n, q, x.get("claim", x.get("sentence")), x["title"] + ". " + x["text"], "", ""]); k.writerow([n, i, x["kind"], x["group"], J[i]["label"], J[i]["reply"].replace("\n", " ")])
print(len(sel), "items; R", sum(items[i]["kind"] == "R" for i in sel), "C", sum(items[i]["kind"] == "C" for i in sel))
