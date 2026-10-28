"""Fake provider for tests and UI work: free, instant, deterministic, streamed in small pieces."""
import re

from app.llm.base import Provider, Usage, estimate_tokens

SOURCE_LINE = re.compile(r"^\[(\d+)\] (.+?) \| (\S+) \| source: (\w+)", re.M)


class FakeProvider(Provider):
    name = "fake"
    paid = False

    def __init__(self, model: str = "fake", script: str | None = None, piece: int = 7):
        super().__init__(model)
        self.script = script  # exact text to stream (tests); otherwise an answer built from the sources
        self.piece = piece

    def generate(self, messages, stream=True):
        prompt = messages[-1]["content"]
        text = self.script if self.script is not None else self._answer(prompt)
        step = self.piece if stream else len(text) or 1
        for i in range(0, len(text), step):
            yield text[i:i + step]
        yield Usage(sum(estimate_tokens(m["content"]) for m in messages), estimate_tokens(text))

    @staticmethod
    def _answer(prompt: str) -> str:
        sources = SOURCE_LINE.findall(prompt)
        if not sources:
            return "I couldn't find that in IIT Ropar's official information."
        bullets = "\n".join(f"- {title} [{n}]" for n, title, _, _ in sources[:5])
        links = "\n".join(f"- {url}" for _, _, url, _ in sources[:5])
        return f"(fake answer) The most relevant official sources are:\n{bullets}\n\nSources:\n{links}"
