"""Compare the author's blind hand labels with the Sonnet judge. No model calls. usage: python hand_review_analyze.py"""
import csv, json
from collections import Counter, defaultdict
sheet = {r["review_id"]: r for r in csv.DictReader(open("results/hand_review_sheet.csv"))}
key = {r["review_id"]: r for r in csv.DictReader(open("results/hand_review_key.csv"))}
VALID = {"R": {"SUPPORTS", "CONTRADICTS", "RELATED NOT EVIDENCE", "UNRELATED"}, "C": {"SUPPORTS", "PARTLY", "DOES NOT SUPPORT"}}
rows, bad = [], []
for rid, r in sheet.items():
    k = key[rid]; lab = r["YOUR_LABEL"].strip().upper().replace("  ", " ")
    if lab not in VALID[k["kind"]]: bad.append((rid, k["kind"], r["YOUR_LABEL"])); continue
    rows.append(dict(rid=rid, kind=k["kind"], group=k["group"], human=lab, judge=k["judge_label"], note=r["note"].strip()))
print(f"{len(sheet)} rows in sheet, {len(rows)} valid labels, {len(bad)} invalid {bad}")
def kappa(pairs):
    n = len(pairs); po = sum(a == b for a, b in pairs) / n; ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[l] * cb[l] for l in set(ca) | set(cb)) / n ** 2; return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")
for kind in ("R", "C"):
    rs = [r for r in rows if r["kind"] == kind]; pairs = [(r["human"], r["judge"]) for r in rs]; po, kp = kappa(pairs)
    print(f"\n== Type {kind}: n={len(rs)} exact agreement {po*100:.1f} percent, Cohen kappa {kp:.2f}")
    labs = sorted(VALID[kind]); print("  rows = human, columns = judge:", labs)
    for h in labs: print("  ", f"{h:22s}", [sum(1 for a, b in pairs if a == h and b == j) for j in labs])
    if kind == "R":
        ev = lambda l: l in ("SUPPORTS", "CONTRADICTS"); p2 = [(ev(a), ev(b)) for a, b in pairs]
        print(f"  collapsed to evidence yes or no: agreement {sum(a==b for a,b in p2)/len(p2)*100:.1f} percent; human says evidence {sum(a for a,_ in p2)}, judge says evidence {sum(b for _,b in p2)} of {len(p2)}")
    else:
        p2 = [(a == "SUPPORTS", b == "SUPPORTS") for a, b in pairs]
        print(f"  collapsed to clean support yes or no: agreement {sum(a==b for a,b in p2)/len(p2)*100:.1f} percent; human clean support {sum(a for a,_ in p2)}, judge {sum(b for _,b in p2)} of {len(p2)}")
    by = defaultdict(list)
    for r in rs: by[r["group"]].append(r)
    for g, v in sorted(by.items()): print(f"  group {g:22s} n={len(v):2d} agree {sum(r['human']==r['judge'] for r in v)}  human labels {dict(Counter(r['human'] for r in v))}")
print("\nrows where human and judge differ:")
for r in rows:
    if r["human"] != r["judge"]: print(f"  #{r['rid']} {r['kind']} {r['group']}: human {r['human']} | judge {r['judge']} | note: {r['note'][:150]}")
print("\nnotes written:", sum(bool(r["note"]) for r in rows), "of", len(rows))
