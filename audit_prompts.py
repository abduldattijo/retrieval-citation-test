"""Frozen judge prompts. AUDIT_VERSION is stored with every judge call."""
AUDIT_VERSION = "v1"
R_SYSTEM = ("You judge whether a passage bears on a scientific claim. Read only the claim and the passage. Reply with one line: Label: followed by exactly one of "
            "SUPPORTS, CONTRADICTS, RELATED NOT EVIDENCE, UNRELATED. SUPPORTS means the passage contains evidence that the claim is true. CONTRADICTS means it contains evidence that the claim is false. "
            "RELATED NOT EVIDENCE means it covers the same topic but gives no evidence for or against the claim. UNRELATED means a different topic. Then, on a second line, give one short reason.")
def r_user(claim, title, text): return f"Claim: {claim}\n\nPassage: {title}. {text}"
C_SYSTEM = ("You check whether a cited passage supports a sentence. Read only the sentence and the passage. Reply with one line: Label: followed by exactly one of SUPPORTS, PARTLY, DOES NOT SUPPORT. "
            "SUPPORTS means the passage clearly backs everything the sentence states. PARTLY means it backs part of the sentence or only loosely. DOES NOT SUPPORT means it does not back the sentence. Then, on a second line, give one short reason.")
def c_user(sentence, title, text): return f"Sentence: {sentence}\n\nCited passage: {title}. {text}"
