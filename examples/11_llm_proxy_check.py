"""Example 11: Check an OpenAI-compatible proxy endpoint safely.

This example sends one small chat completion request to the endpoint configured
by environment variables. It never stores API keys in source code.

Required:
    OPENAI_API_KEY

Optional:
    OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    OPENAI_MODEL=qwen-plus

Usage:
    python examples/11_llm_proxy_check.py --message "Say hello"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test an OpenAI-compatible chat completions endpoint.")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")))
    parser.add_argument("--message", default="Say hello in one short sentence.")
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def build_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def main() -> int:
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is required; refusing to run without an environment key.", file=sys.stderr)
        return 2

    payload = {
        "model": args.model,
        "max_tokens": 40,
        "messages": [{"role": "user", "content": args.message}],
    }
    request = urllib.request.Request(
        build_url(args.base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        print(f"Request failed with HTTP {exc.code}", file=sys.stderr)
        print(exc.read().decode("utf-8", errors="replace"), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Request failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
