"""A source passage handed to the LLM, in the numbered format prompts/system_prompt.md describes."""
from dataclasses import dataclass


@dataclass
class Source:
    title: str
    url: str
    source_type: str  # site | pdf | api | live | web
    fetched: str      # YYYY-MM-DD
    text: str


def format_sources(sources: list[Source]) -> str:
    return "\n\n".join(f"[{i}] {s.title} | {s.url} | source: {s.source_type} | fetched: {s.fetched}\n{s.text}"
                       for i, s in enumerate(sources, 1))
