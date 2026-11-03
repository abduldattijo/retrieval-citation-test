"""Frozen prompt. Do not edit after the pilot starts. PROMPT_VERSION is recorded with every call."""
PROMPT_VERSION = "v1"
SYSTEM = ("You check scientific claims against a numbered list of passages. Use only the passages. "
          "Reply in exactly this format. First line: Verdict: SUPPORTED, REFUTED or NOT SETTLED. "
          "Then at most three sentences explaining the verdict. After each statement, cite the passages you rely on by number in square brackets, "
          "like [3] or [2][5]. Cite only passages you actually use.")
def user_message(claim: str, passages: list[tuple[str, str]]) -> str:
    body = "\n\n".join(f"[{i}] {t}. {x}" for i, (t, x) in enumerate(passages, 1))
    return f"Claim: {claim}\n\nPassages:\n\n{body}"
