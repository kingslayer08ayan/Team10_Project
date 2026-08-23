"""
Three-tier pluggable backend, checked in this order:
  1. Anthropic API      (if ANTHROPIC_API_KEY is set)
  2. Hugging Face router (if HF_TOKEN is set) -- hosted inference, no local
     model download, no GPU/CPU load on your machine. Uses HF's
     OpenAI-compatible chat-completions router, so it's a plain HTTPS call,
     not a `transformers` model loaded into your laptop's memory.
  3. Offline heuristic  (always available, zero network, zero API cost)

Every GraphRAG stage that "reasons" (entity extraction, HyDE generation)
goes through reason() below, so nothing in the pipeline cares which tier
actually served the request.

Set HF_MODEL to override the default hosted model, e.g.:
    export HF_TOKEN="hf_..."
    export HF_MODEL="Qwen/Qwen3-8B"   # optional
":fastest" (or ":cheapest") lets HF auto-pick a provider for that model --
check https://huggingface.co/inference for what's currently available if
the default below isn't.
"""
import os
import json
import re
import urllib.request
import urllib.error

ANTHROPIC_MODEL = "claude-sonnet-4-6"
HF_MODEL_DEFAULT = "Qwen/Qwen3-8B"
HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
_hf_credits_exhausted = False


def _active_backend():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY"):
        return "huggingface"
    return None


def llm_available() -> bool:
    return _active_backend() is not None


def _call_anthropic(prompt, system, max_tokens):
    api_key = os.environ["ANTHROPIC_API_KEY"]
    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return "\n".join(b["text"] for b in data.get("content", []) if b.get("type") == "text")


def _call_huggingface(prompt, system, max_tokens):
    """Hits HF's OpenAI-compatible chat-completions router. Runs on HF's
    hosted infrastructure -- nothing is downloaded or executed locally."""
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
    model = os.environ.get("HF_MODEL", HF_MODEL_DEFAULT)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if model.startswith("Qwen/Qwen3"):
        # Qwen3 providers do not all honor chat_template_kwargs; /no_think is
        # the model's prompt-level switch and prevents content=null on short
        # completion budgets.
        prompt = f"/no_think\n{prompt}"
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        # Qwen3 otherwise spends the small completion budget on hidden
        # reasoning and may return content=null with finish_reason=length.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        HF_ROUTER_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            # Some HF-routed providers sit behind Cloudflare and block the
            # default urllib User-Agent as bot traffic (Cloudflare error
            # 1010). A normal-looking UA avoids that.
            "User-Agent": "Mozilla/5.0 (compatible; graphrag-streamlit/1.0)",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        if "<html" in detail.lower() or "<!doctype" in detail.lower():
            # A Cloudflare/proxy HTML error page slipped through instead of
            # JSON -- pull out just the title so the UI shows one line, not
            # the whole page.
            m = re.search(r"<title>(.*?)</title>", detail, re.IGNORECASE | re.DOTALL)
            detail = m.group(1).strip() if m else "(non-JSON HTML error page returned)"
        raise RuntimeError(f"HF router HTTP {e.code} for model '{model}': {detail}") from None
    message = data.get("choices", [{}])[0].get("message", {})
    content = message.get("content")
    if content is None:
        raise RuntimeError(f"HF router returned no content for model '{model}' "
                           f"(full response: {json.dumps(data)[:300]})")
    return content


def call_llm(prompt: str, system: str = "", max_tokens: int = 500) -> str:
    backend = _active_backend()
    if backend == "anthropic":
        return _call_anthropic(prompt, system, max_tokens)
    if backend == "huggingface":
        return _call_huggingface(prompt, system, max_tokens)
    raise RuntimeError("No LLM backend configured (set ANTHROPIC_API_KEY or HF_TOKEN).")


def reason(prompt: str, system: str = "", max_tokens: int = 500, offline_fn=None):
    """
    Returns (text, source) where source is 'llm' or 'offline'.

    Falls back to offline_fn not just on an exception, but also when the
    LLM technically succeeded (200 OK) but returned empty/whitespace-only
    content -- this happens with some HF-routed providers on certain
    inputs, and every caller downstream (graph_builder's community
    summaries, hyde's query expansion, etc.) assumes it's getting a real
    non-empty string back. Catching it here once means no individual
    caller needs its own None-guard.
    """
    global _hf_credits_exhausted

    if llm_available() and not _hf_credits_exhausted:
        try:
            result = call_llm(prompt, system=system, max_tokens=max_tokens)
            if result is None or not result.strip():
                raise RuntimeError("LLM call succeeded but returned empty/None content")
            return result, "llm"
        except Exception as e:
            if _active_backend() == "huggingface" and "HF router HTTP 402" in str(e):
                _hf_credits_exhausted = True
                print("[llm_backend] HF credits are exhausted; using offline fallback for the rest of this run.")
            else:
                print(f"[llm_backend] {_active_backend()} call failed ({e}); using offline fallback.")
    if offline_fn is None:
        raise RuntimeError("No LLM available and no offline_fn provided.")
    return offline_fn(), "offline"