"""
Optional local LLM helper for "Interpret these results using an LLM" (e.g. Ollama).
No API keys or paid services. Uses base URL only.
Can auto-start Ollama via `ollama serve` if not running.
"""
from __future__ import annotations

import json
import subprocess
import time
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"  # Good for analytical tasks; configurable via session_state


def _ping_ollama(base_url: str, timeout: int = 3) -> bool:
    """Return True if Ollama responds at base_url."""
    url = (base_url or "").rstrip("/") + "/api/tags"
    try:
        req = Request(url, method="GET")
        with urlopen(req, timeout=timeout) as _:
            return True
    except (URLError, HTTPError, OSError):
        return False


def ensure_ollama_running(base_url: str = DEFAULT_OLLAMA_URL, wait_seconds: int = 5) -> bool:
    """
    Ping Ollama at base_url. If not responding, try to start `ollama serve`, then retry.
    Returns True if Ollama responds, False otherwise. Handles Ollama not on PATH.
    """
    if _ping_ollama(base_url):
        return True
    try:
        proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError:
        return False
    for _ in range(wait_seconds):
        time.sleep(1)
        if _ping_ollama(base_url):
            return True
    try:
        proc.terminate()
    except Exception:
        pass
    return False


def enhance_with_ollama(
    base_url: str,
    context: str,
    model: Optional[str] = None,
    timeout: int = 15,
) -> str:
    """
    Ask local Ollama to interpret results from the given context.
    Professor-like: draw new conclusions, mention caveats or follow-up, optionally clinical implications.
    Returns generated text or empty string on error.
    """
    if not (base_url or "").strip() or not (context or "").strip():
        return ""
    url = base_url.rstrip("/") + "/api/generate"
    model = model or DEFAULT_OLLAMA_MODEL
    prompt = (
        "You are a scientific interpreter (like a thoughtful professor). Use ONLY the context below. "
        "Interpret the results in your own words: draw conclusions, note caveats or follow-up analyses, "
        "and if the domain is clinical, briefly mention clinical implications (e.g. reference ranges, "
        "measurement error, actionable thresholds) where relevant. "
        "Do NOT simply restate or paraphrase any existing summary; offer what you infer from the numbers and setup. "
        "Reply in 2–4 sentences, scientific-inquiry style. No preamble, no bullet points.\n\n"
        "Context:\n"
        f"{context.strip()}"
    )
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        req = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data.get("response") or "").strip()
    except (URLError, HTTPError, json.JSONDecodeError, KeyError, OSError):
        return ""
