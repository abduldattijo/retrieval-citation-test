from parse_citations import parse
def t(reply, **want):
    got = parse(reply)
    for k, v in want.items(): assert got[k] == v, (reply, k, got[k], v)
t("Verdict: SUPPORTED\nThe claim holds [3]. It is shown again [3][5].", verdict="SUPPORTED", cited=[3, 5])
t("Verdict: refuted\nContradicted [2, 7].", verdict="REFUTED", cited=[2, 7])
t("Verdict: NOT SETTLED\nNo passage settles it.", verdict="NOT SETTLED", cited=[])
t("Verdict: SUPPORTED\nSeveral passages [2-4] agree.", cited=[2, 3, 4])
t("Verdict: SUPPORTED\nOut of range [11] and [0].", cited=[], invalid=[0, 11])
t("The claim holds [3].", verdict=None, cited=[3])
t("Verdict: SUPPORTED\nSee [1] and also [10].", cited=[1, 10])
t("Verdict: SUPPORTED\nNo brackets at all.", cited=[], n_marks=0)
print("parse tests ok")
