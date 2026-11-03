import re
VERDICT = re.compile(r"^\s*Verdict:\s*(SUPPORTED|REFUTED|NOT SETTLED)\b", re.I | re.M)
BRACKET = re.compile(r"\[([\d,\s–\-]+)\]")
def parse(reply: str, n_passages: int = 10):
    """Returns dict(verdict, cited (sorted unique positions 1..n), invalid (positions outside 1..n), n_citation_marks). verdict None if the format is broken."""
    m = VERDICT.search(reply); verdict = m.group(1).upper() if m else None
    cited, invalid, marks = set(), set(), 0
    for b in BRACKET.finditer(reply):
        marks += 1
        for part in re.split(r"[,\s]+", b.group(1).strip()):
            if not part: continue
            rng = re.split(r"[–\-]", part)
            try: nums = range(int(rng[0]), int(rng[-1]) + 1) if len(rng) == 2 and rng[1] else [int(rng[0])]
            except ValueError: continue
            for k in nums: (cited if 1 <= k <= n_passages else invalid).add(k)
    return dict(verdict=verdict, cited=sorted(cited), invalid=sorted(invalid), n_marks=marks)
