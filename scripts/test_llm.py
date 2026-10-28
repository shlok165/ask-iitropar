"""Quick check that the Pollinations key, endpoint and model in .env work.

Usage:  python scripts/test_llm.py
Never prints the API key.
"""
import sys
from pathlib import Path

import requests

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def load_env(path: Path) -> dict:
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.split(" #", 1)[0].strip().strip('"').strip("'")
    return env


def main():
    if not ENV_PATH.exists():
        sys.exit(f"No .env found at {ENV_PATH}. Copy .env.example to .env and fill it in.")
    env = load_env(ENV_PATH)
    key, base, model = (env.get(k, "") for k in
                        ("POLLINATIONS_API_KEY", "POLLINATIONS_BASE_URL", "POLLINATIONS_MODEL"))
    missing = [n for n, v in [("POLLINATIONS_API_KEY", key), ("POLLINATIONS_BASE_URL", base),
                              ("POLLINATIONS_MODEL", model)] if not v]
    if missing:
        sys.exit(f"Fill these in .env first: {', '.join(missing)}")

    url = base.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    print(f"Endpoint: {url}\nModel:    {model}\nKey:      {key[:6]}… (hidden)\n")

    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "In one sentence: where is IIT Ropar located?"},
            ],
            # GPT-5.4 models take reasoning_effort instead of temperature; hidden reasoning
            # tokens also count against the limit, so leave headroom.
            "max_completion_tokens": 400,
            "reasoning_effort": "low",
        },
        timeout=60,
    )
    print(f"HTTP {resp.status_code}")
    try:
        data = resp.json()
    except ValueError:
        sys.exit(f"Response was not JSON:\n{resp.text[:500]}")
    if resp.status_code != 200:
        sys.exit(f"Error from API:\n{str(data)[:800]}")

    print("Answer:", data["choices"][0]["message"]["content"].strip())
    usage = data.get("usage") or {}
    print(f"Tokens: {usage.get('prompt_tokens', '?')} in, {usage.get('completion_tokens', '?')} out")
    print("\nOK: key, endpoint and model all work.")


if __name__ == "__main__":
    main()
