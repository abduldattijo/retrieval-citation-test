"""Model calls with a spend ledger. Trimmed copy of the file used for the runs: it keeps the price table, the ledger and the Anthropic call, and reads the key only from ANTHROPIC_API_KEY.
No temperature is sent, because these models reject non default values. Prices are US dollars per million tokens, checked on the Anthropic pricing page on 2026-11-03 (prompts up to 100,000 tokens)."""
from __future__ import annotations
import json, os, threading, time
from pathlib import Path
PRICES = {"claude-haiku-5-5": {"in": 0.10, "out": 0.50}, "claude-sonnet-5-5": {"in": 2.00, "out": 10.00}}
THINKING = {"claude-haiku-5-5": {"type": "disabled"}, "claude-sonnet-5-5": {"type": "between_tools"}}
class BudgetExceeded(RuntimeError): pass
class Ledger:
    """Project spend, saved to a JSON file. A call reserves its worst case cost before it is sent, so calls in flight cannot carry the total past the cap."""
    def __init__(self, cap_usd: float, path=None):
        self.cap, self.calls, self.tin, self.tout, self.reserved = cap_usd, 0, 0, 0, 0.0
        self.path = Path(path) if path else None; self.base = 0.0
        if self.path and self.path.exists(): self.base = json.loads(self.path.read_text())["total_usd"]
        self.usd = 0.0; self.lock = threading.Lock()
    @property
    def total(self): return self.base + self.usd
    def reserve(self, est):
        with self.lock:
            if self.total + self.reserved + est > self.cap: raise BudgetExceeded(f"project total {self.total:.4f} plus reserve {self.reserved + est:.4f} would pass the cap {self.cap:.2f}")
            self.reserved += est
    def release(self, est):
        with self.lock: self.reserved -= est
    def add(self, usd, tin, tout, est=0.0):
        with self.lock:
            self.reserved -= est; self.usd += usd; self.calls += 1; self.tin += tin; self.tout += tout
            if self.path: self.path.write_text(json.dumps({"total_usd": self.base + self.usd}))
class Chooser:
    """choose(system, user) returns (reply_text, record) for one Claude model."""
    def __init__(self, model: str, ledger: Ledger, max_tokens: int = 12):
        if model not in PRICES: raise RuntimeError(f"no price for {model}")
        import anthropic
        key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not key: raise RuntimeError("Set ANTHROPIC_API_KEY")
        self.model, self.ledger, self.max_tokens, self.client = model, ledger, max_tokens, anthropic.Anthropic(api_key=key)
    def choose(self, system: str, user: str, max_tokens: int | None = None):
        mt = max_tokens or self.max_tokens; p = PRICES[self.model]; last = None
        est = ((len(system) + len(user)) / 2.0 * p["in"] + mt * p["out"]) / 1e6      # worst case: 2 characters per token, full output
        self.ledger.reserve(est)
        for attempt in range(5):
            t0 = time.monotonic()
            try:
                r = self.client.messages.create(model=self.model, max_tokens=mt, system=system, messages=[{"role": "user", "content": user}], thinking=THINKING[self.model])
                text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text"); tin, tout = r.usage.input_tokens, r.usage.output_tokens
                usd = (tin * p["in"] + tout * p["out"]) / 1e6; self.ledger.add(usd, tin, tout, est)
                return text, {"in": tin, "out": tout, "usd": usd, "s": round(time.monotonic() - t0, 3)}
            except BudgetExceeded: self.ledger.release(est); raise
            except Exception as e: last = type(e).__name__; time.sleep(2 ** attempt)
        self.ledger.release(est); raise RuntimeError(f"model call failed after 5 tries ({last})")
