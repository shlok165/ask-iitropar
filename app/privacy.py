"""Redaction of every answer, including while it streams. Reuses scraper/common.py's redact()."""
import re

from scraper.common import redact

HOLD = 32  # characters held back so a number split across two stream pieces is still caught

# A run of characters that could be (part of) an entry number or mobile number: letters/digits/+,
# with short space/hyphen gaps allowed when a digit follows ("+91 98765-43210").
_RUN = re.compile(r"[+\dA-Za-z](?:[\dA-Za-z+]|[\s-]{1,3}(?=[\d+]))*")


class StreamRedactor:
    """feed() pieces as they arrive and get back text that is safe to show; flush() at the end."""

    def __init__(self):
        self.buf = ""

    def feed(self, piece: str) -> str:
        self.buf += piece
        cut = len(self.buf) - HOLD
        if cut <= 0:
            return ""
        for m in _RUN.finditer(self.buf):
            if m.start() < cut < m.end():  # never cut through something that might be a number
                cut = m.start()
                break
            if m.start() >= cut:
                break
        out, self.buf = self.buf[:cut], self.buf[cut:]
        return redact(out)

    def flush(self) -> str:
        out, self.buf = self.buf, ""
        return redact(out)


__all__ = ["StreamRedactor", "redact"]
