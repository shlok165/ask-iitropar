"""Provider interface: generate(messages, stream=True) yields text pieces, then one final Usage."""
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    cost: float = 0.0          # in the provider's currency: pollen (Pollinations), 0 (local)
    estimated: bool = False    # True when the provider didn't report usage and we estimated it


class LLMError(Exception):
    def __init__(self, message: str, status: int | None = None, out_of_credit: bool = False):
        super().__init__(message)
        self.status = status
        self.out_of_credit = out_of_credit


def estimate_tokens(text: str) -> int:
    """Deliberately generous (~3.5 chars/token) so budget checks err on the safe side."""
    return int(len(text) / 3.5) + 1


def messages_tokens(messages: list[dict]) -> int:
    return sum(estimate_tokens(m["content"]) + 4 for m in messages)


class Provider:
    name = "base"
    paid = False

    def __init__(self, model: str):
        self.model = model

    def available(self) -> bool:
        return True

    def generate(self, messages: list[dict], stream: bool = True) -> Iterator[str | Usage]:
        raise NotImplementedError

    def worst_case_cost(self, messages: list[dict], max_tokens: int) -> float:
        """Upper bound on the cost of one call, checked against the daily cap before calling."""
        return 0.0
