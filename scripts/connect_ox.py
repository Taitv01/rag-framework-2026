"""Connect this project to Ox Alpha with OpenRouter OAuth PKCE.

The script opens OpenRouter in the user's browser, receives the authorization
code on a random localhost port, exchanges it for a user-controlled API key,
and stores the key in the ignored ``.env.local`` file. The key is never printed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import secrets
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ENV_FILE = PROJECT_ROOT / ".env.local"
OPENROUTER_AUTH_URL = "https://openrouter.ai/auth"
OPENROUTER_EXCHANGE_URL = "https://openrouter.ai/api/v1/auth/keys"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OX_MODEL = "stealth/ox-alpha"


class OxRateLimitError(RuntimeError):
    """Raised when the free Ox route remains rate-limited after retries."""


def create_pkce_pair() -> tuple[str, str]:
    """Create an RFC 7636 verifier and S256 challenge."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def update_env_file(path: Path, key: str) -> None:
    """Atomically add or replace OX_API_KEY while preserving other settings."""
    key = key.strip()
    if not key or any(character in key for character in "\r\n"):
        raise ValueError("OpenRouter returned an invalid API key")

    existing_lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    output_lines = []
    replaced = False
    for line in existing_lines:
        if line.startswith("OX_API_KEY="):
            if not replaced:
                output_lines.append(f"OX_API_KEY={key}")
                replaced = True
            continue
        output_lines.append(line)

    if not replaced:
        if output_lines and output_lines[-1].strip():
            output_lines.append("")
        output_lines.append(f"OX_API_KEY={key}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f"{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_name = temporary_file.name
            temporary_file.write("\n".join(output_lines) + "\n")
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def read_ox_key(path: Path) -> Optional[str]:
    """Read only OX_API_KEY from an env file without exposing other secrets."""
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("OX_API_KEY="):
            key = line.partition("=")[2].strip()
            return key or None
    return None


def exchange_code(code: str, verifier: str) -> str:
    """Exchange an OAuth authorization code for an OpenRouter API key."""
    payload = json.dumps(
        {
            "code": code,
            "code_verifier": verifier,
            "code_challenge_method": "S256",
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        OPENROUTER_EXCHANGE_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = json.load(response)

    key = body.get("key")
    if not isinstance(key, str) or not key.strip():
        raise RuntimeError("OpenRouter did not return an API key")
    return key


def authorize(timeout_seconds: int = 300) -> str:
    """Run the localhost OAuth flow and return the generated key."""
    verifier, challenge = create_pkce_pair()
    callback_token = secrets.token_urlsafe(24)
    result: dict[str, str] = {}
    completed = threading.Event()

    class CallbackHandler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != f"/callback/{callback_token}":
                self.send_error(404)
                return

            params = urllib.parse.parse_qs(parsed.query)
            if params.get("code"):
                result["code"] = params["code"][0]
                title = "Ox Alpha connected"
                message = "Authorization received. You can close this tab."
            else:
                result["error"] = params.get("error", ["Authorization was not completed"])[0]
                title = "Ox Alpha authorization failed"
                message = "Return to the terminal for details."

            body = (
                "<!doctype html><meta charset='utf-8'>"
                f"<title>{title}</title>"
                "<style>body{font:16px system-ui;max-width:680px;margin:15vh auto;"
                "padding:2rem;line-height:1.5}</style>"
                f"<h1>{title}</h1><p>{message}</p>"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            completed.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), CallbackHandler)
    server.timeout = 1
    callback_url = (
        f"http://127.0.0.1:{server.server_address[1]}/callback/{callback_token}"
    )
    auth_url = OPENROUTER_AUTH_URL + "?" + urllib.parse.urlencode(
        {
            "callback_url": callback_url,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
    )

    print("Opening OpenRouter authorization in your browser...")
    if not webbrowser.open(auth_url):
        print(f"Open this URL manually:\n{auth_url}")

    deadline = time.monotonic() + timeout_seconds
    try:
        while not completed.is_set() and time.monotonic() < deadline:
            server.handle_request()
    finally:
        server.server_close()

    if not completed.is_set():
        raise TimeoutError("Timed out waiting for OpenRouter authorization")
    if "error" in result:
        raise RuntimeError(f"OpenRouter authorization failed: {result['error']}")
    return exchange_code(result["code"], verifier)


def smoke_test(key: str, max_attempts: int = 5) -> str:
    """Make one small Ox Alpha request and return its displayable answer."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    payload = json.dumps(
        {
            "model": OX_MODEL,
            "messages": [{"role": "user", "content": "Reply exactly: Ox Alpha connected"}],
            "temperature": 0,
            "max_tokens": 256,
        }
    ).encode("utf-8")
    for attempt in range(1, max_attempts + 1):
        request = urllib.request.Request(
            OPENROUTER_CHAT_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "X-OpenRouter-Title": "Ultimate RAG Ox Setup",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.load(response)
            break
        except urllib.error.HTTPError as error:
            if error.code != 429:
                raise
            if attempt == max_attempts:
                raise OxRateLimitError(
                    "Ox Alpha's free route is temporarily rate-limited; "
                    "the stored key is ready, so retry the check later."
                ) from error
            retry_after = error.headers.get("Retry-After")
            try:
                delay = max(1, min(int(retry_after), 30)) if retry_after else 2 ** attempt
            except ValueError:
                delay = 2 ** attempt
            print(f"Free route is busy; retrying in {delay}s ({attempt}/{max_attempts})...")
            time.sleep(delay)

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Ox Alpha returned no completion choices")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Ox Alpha returned no displayable content")
    return content.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Authorize OpenRouter, store OX_API_KEY, and test Ox Alpha."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Seconds to wait for browser authorization (default: 300).",
    )
    parser.add_argument(
        "--no-smoke-test",
        action="store_true",
        help="Create and store the key without sending a model request.",
    )
    parser.add_argument(
        "--reauthorize",
        action="store_true",
        help="Create a new key even when OX_API_KEY already exists in .env.local.",
    )
    args = parser.parse_args()

    try:
        key = None if args.reauthorize else read_ox_key(LOCAL_ENV_FILE)
        if key:
            print(f"Using the existing Ox key in {LOCAL_ENV_FILE.name}.")
        else:
            key = authorize(timeout_seconds=args.timeout)
            update_env_file(LOCAL_ENV_FILE, key)
            print(f"OpenRouter key stored securely in {LOCAL_ENV_FILE.name}.")
        if not args.no_smoke_test:
            print("Testing Ox Alpha (this can take a minute on the free route)...")
            print(smoke_test(key))
        print("Ox Alpha is ready for this RAG project.")
        return 0
    except OxRateLimitError as error:
        print(error)
        return 2
    except (OSError, TimeoutError, RuntimeError, ValueError, urllib.error.URLError) as error:
        print(f"Setup failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
