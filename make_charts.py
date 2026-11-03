import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import Counter, defaultdict
P = {c["claim_id"]: c for c in json.load(open("data/pools.json"))["claims"]}
H = [r for r in map(json.loads, open("results/main_haiku.jsonl")) if r["reply"].strip() != "" and not r["truncated"]]
by = defaultdict(list)
for r in H: by[r["claim_id"]].append(r)
rel, unl = [], []
for c, rs in by.items():
    if len(rs) < 8: continue
    cnt = Counter(r["shown_doc_ids"][k - 1] for r in rs for k in set(r["cited"]))
    for p in P[c]["pool"]: (rel if p["relevant"] else unl).append(cnt[p["doc_id"]] / len(rs))
bins = [(0, 0), (0.0001, 0.2), (0.2001, 0.4), (0.4001, 0.6), (0.6001, 0.8), (0.8001, 1.0)]
labels = ["never", "up to 0.2", "0.2 to 0.4", "0.4 to 0.6", "0.6 to 0.8", "0.8 to 1.0"]
def share(v): return [100 * sum(lo <= x <= hi for x in v) / len(v) for lo, hi in bins]
fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=200); x = np.arange(len(bins)); w = 0.38
b1 = ax.bar(x - w / 2, share(rel), w, color="#1f77b4", label=f"Labeled relevant ({len(rel)} passages)"); b2 = ax.bar(x + w / 2, share(unl), w, color="#d55e00", label=f"Unlabeled ({len(unl):,} passages)")
for b in list(b1) + list(b2): ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.0f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9); ax.set_xlabel("How often the passage was cited across its random orders (share of orders)"); ax.set_ylabel("Share of passages in the group (percent)")
ax.set_ylim(0, 85); ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig("figures/figure1_citation_frequency.png"); print("fig1", [round(v) for v in share(rel)], [round(v) for v in share(unl)])
# text free featured image: ten cards, one blue (labeled relevant), orange top bar on cited ones
fig, ax = plt.subplots(figsize=(10, 5.625), dpi=200); ax.set_xlim(0, 32); ax.set_ylim(0, 18); ax.axis("off")
cited = {2, 6}; relc = 2
for i in range(10):
    x0 = 1.2 + i * 3.0; col = "#1f77b4" if i == relc else "#d9dde3"
    ax.add_patch(Rectangle((x0, 4.2), 2.3, 9.0, color=col, lw=0))
    for k in range(4): ax.add_patch(Rectangle((x0 + 0.35, 6.2 + k * 1.6), 1.6, 0.35, color="white" if i == relc else "#b9c0ca", lw=0))
    if i in cited: ax.add_patch(Rectangle((x0, 13.4), 2.3, 1.1, color="#d55e00", lw=0))
fig.subplots_adjust(0, 0, 1, 1); fig.savefig("figures/featured_image_text_free.png"); print("featured saved")
