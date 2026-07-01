#!/usr/bin/env python3
"""Line-oriented MLX-LM helper for the Gemma4D native shim.

The parent process owns this helper through pipes.  It intentionally exposes only
token prefill/decode commands so Rust still talks to the narrow C ABI.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm.models import cache as cache_mod
from mlx_lm.utils import load_model


def emit(payload: dict[str, Any]) -> None:
    try:
        print(json.dumps(payload, separators=(",", ":")), flush=True)
    except BrokenPipeError:
        raise SystemExit(0) from None


def fail(message: str) -> None:
    emit({"ok": False, "error": message})


if len(sys.argv) != 2:
    fail("helper requires exactly one model path argument")
    raise SystemExit(2)

model_path = Path(sys.argv[1])
try:
    started = time.perf_counter()
    model, config = load_model(
        model_path,
        lazy=False,
        strict=False,
        model_config={"model_type": "gemma4"},
    )
    load_s = time.perf_counter() - started
except Exception as exc:  # noqa: BLE001 - propagate to native parent as data.
    fail(f"failed to load MLX-LM Gemma 4 text model: {exc}")
    raise SystemExit(1)

def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        parsed = int(raw)
    except ValueError as exc:
        fail(f"{name} must be an integer, got {raw!r}")
        raise SystemExit(1) from exc
    if parsed <= 0:
        fail(f"{name} must be greater than zero, got {parsed}")
        raise SystemExit(1)
    return parsed


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw not in {"0", "false", "FALSE", "off", "OFF"}


prompt_cache = None
sequence_len = 0
prefill_chunk_tokens = env_int("GEMMA4D_MLX_LM_PREFILL_CHUNK_TOKENS", 2048)
prefill_clear_cache = env_flag("GEMMA4D_MLX_LM_PREFILL_CLEAR_CACHE", True)

emit(
    {
        "ok": True,
        "backend": "mlx_lm_gemma4_text_helper",
        "model_type": config.get("model_type"),
        "load_s": load_s,
        "prefill_chunk_tokens": prefill_chunk_tokens,
        "prefill_clear_cache": prefill_clear_cache,
    }
)


def cache_state():
    return [entry.state for entry in prompt_cache]


def model_step(tokens: list[int]) -> tuple[int, float]:
    if not tokens:
        raise ValueError("model_step requires at least one token")
    input_tokens = mx.array(tokens, dtype=mx.uint32)
    logits = model(input_tokens[None], cache=prompt_cache)
    logits = logits[:, -1, :]
    greedy = mx.argmax(logits, axis=-1)
    mx.eval(greedy, logits, cache_state())
    token = int(greedy.item())
    logit = float(logits[0, token].item())
    return token, logit


def memory_payload() -> dict[str, float]:
    try:
        peak_memory_gb = float(mx.get_peak_memory() / 1e9)
    except Exception:
        peak_memory_gb = 0.0

    peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        peak_rss_mb = peak_rss / (1024 * 1024)
    else:
        peak_rss_mb = peak_rss / 1024

    return {
        "peak_memory_gb": peak_memory_gb,
        "peak_rss_mb": peak_rss_mb,
    }


def prefill(tokens: list[int]) -> tuple[int, float, int]:
    global prompt_cache, sequence_len
    if not tokens:
        raise ValueError("prefill requires at least one token")

    prompt_cache = cache_mod.make_prompt_cache(model)
    sequence_len = 0
    remaining = list(tokens)

    while len(remaining) > prefill_chunk_tokens:
        chunk = remaining[:prefill_chunk_tokens]
        input_tokens = mx.array(chunk, dtype=mx.uint32)
        model(input_tokens[None], cache=prompt_cache)
        mx.eval(cache_state())
        sequence_len += len(chunk)
        remaining = remaining[prefill_chunk_tokens:]
        if prefill_clear_cache:
            mx.clear_cache()

    token, logit = model_step(remaining)
    sequence_len += len(remaining)
    return token, logit, sequence_len


def decode_one(token: int) -> tuple[int, float, int]:
    global sequence_len
    if prompt_cache is None:
        raise ValueError("decode_one requires a prior prefill")
    token, logit = model_step([token])
    sequence_len += 1
    return token, logit, sequence_len


for line in sys.stdin:
    try:
        request = json.loads(line)
        cmd = request.get("cmd")

        if cmd == "prefill":
            token, logit, seq = prefill([int(t) for t in request.get("tokens", [])])
            emit(
                {
                    "ok": True,
                    "greedy_token": token,
                    "greedy_logit": logit,
                    "sequence_len": seq,
                    **memory_payload(),
                }
            )
        elif cmd == "decode_one":
            token, logit, seq = decode_one(int(request["token"]))
            emit(
                {
                    "ok": True,
                    "greedy_token": token,
                    "greedy_logit": logit,
                    "sequence_len": seq,
                    **memory_payload(),
                }
            )
        elif cmd == "reset":
            prompt_cache = None
            sequence_len = 0
            emit({"ok": True})
        elif cmd == "shutdown":
            emit({"ok": True})
            break
        else:
            fail(f"unknown helper command: {cmd}")
    except Exception as exc:  # noqa: BLE001 - command errors return as data.
        fail(str(exc))
