#!/usr/bin/env python3
"""XR54 drafter-only parity against the vendored Transformers assistant.

This script intentionally does not download code or weights. It consumes a local
native parity payload with `hidden.last`, shared KV tensors, and ordered target
token embeddings exported by `xr54_drafter_pytorch_parity.rs`.

The PyTorch reference path expects a dense assistant checkpoint. Generate one
from the local MLX affine-q4 artifact with `scripts/xr54_dequant_assistant.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    args = parse_args()
    try:
        request = json.loads(args.request.read_text())
    except Exception as exc:
        write_blocked(
            args.out,
            "failed to load PyTorch parity request",
            request_path=str(args.request),
            detail=str(exc),
        )
        return 2

    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised by local env.
        write_blocked(
            args.out,
            "PyTorch is not installed in the selected Python environment",
            request=request,
            detail=str(exc),
        )
        return 2

    torch.set_default_dtype(torch.bfloat16)

    try:
        from safetensors import safe_open
        from safetensors.torch import load_file
        from transformers.models.gemma4_assistant.configuration_gemma4_assistant import (
            Gemma4AssistantConfig,
        )
        from transformers.models.gemma4_assistant.modeling_gemma4_assistant import (
            Gemma4AssistantForCausalLM,
        )
    except Exception as exc:
        write_blocked(
            args.out,
            "required PyTorch/Transformers/safetensors reference imports failed",
            request=request,
            detail=str(exc),
        )
        return 2

    try:
        quantized_keys = assistant_quantized_keys(args.assistant_model_path)
    except FileNotFoundError as exc:
        write_blocked(
            args.out,
            "assistant checkpoint is missing model.safetensors",
            request=request,
            detail=str(exc),
        )
        return 2
    except Exception as exc:
        write_blocked(
            args.out,
            "failed to inspect assistant checkpoint format",
            request=request,
            detail=str(exc),
        )
        return 2
    if quantized_keys:
        write_blocked(
            args.out,
            "assistant checkpoint is MLX affine q4; run scripts/xr54_dequant_assistant.py and pass the dense checkpoint path",
            request=request,
            quantized_key_examples=quantized_keys[:12],
        )
        return 2

    try:
        payload, metadata = load_payload(args.payload, safe_open)
        config = assistant_config(args.assistant_model_path, Gemma4AssistantConfig)
        model = Gemma4AssistantForCausalLM(config).to(args.device).eval()
        state = load_file(str(args.assistant_model_path / "model.safetensors"), device=args.device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing_allowed = {"lm_head.weight"} if config.tie_word_embeddings else set()
        hard_missing = [key for key in missing if key not in missing_allowed]
        if hard_missing or unexpected:
            raise RuntimeError(
                f"assistant state_dict mismatch: missing={hard_missing[:20]} unexpected={unexpected[:20]}"
            )
        tie_lm_head(model)

        model_dtype = next(model.parameters()).dtype
        token_embeddings = token_embedding_map(payload, metadata, args.device, model_dtype)
        shared_kv_states = {
            "full_attention": (
                payload["hidden.full_attention_key"].to(device=args.device, dtype=model_dtype),
                payload["hidden.full_attention_value"].to(device=args.device, dtype=model_dtype),
            ),
            "sliding_attention": (
                payload["hidden.sliding_attention_key"].to(device=args.device, dtype=model_dtype),
                payload["hidden.sliding_attention_value"].to(device=args.device, dtype=model_dtype),
            ),
        }
        last_hidden = payload["hidden.last"].to(device=args.device, dtype=model_dtype)
        native_tokens = [int(token) for token in request["native_draft_tokens"]]
        first_position = int(request["first_position"])
        block_size = int(request["block_size"])
        last_context_token = int(request["last_context_token"])

        pinned = run_variant(
            torch=torch,
            model=model,
            token_embeddings=token_embeddings,
            shared_kv_states=shared_kv_states,
            last_hidden=last_hidden,
            last_token_id=last_context_token,
            first_position=first_position,
            block_size=block_size,
            increment_positions=False,
        )
        incremented = run_variant(
            torch=torch,
            model=model,
            token_embeddings=token_embeddings,
            shared_kv_states=shared_kv_states,
            last_hidden=last_hidden,
            last_token_id=last_context_token,
            first_position=first_position,
            block_size=block_size,
            increment_positions=True,
        )
    except Exception as exc:
        write_blocked(args.out, "PyTorch parity execution failed", request=request, detail=str(exc))
        return 2

    result = {
        "schema_version": 1,
        "status": "completed",
        "request": request,
        "pinned": pinned,
        "incremented": incremented,
        "native_draft_tokens": native_tokens,
        "matches_native": {
            "pinned": pinned["draft_tokens"] == native_tokens,
            "incremented": incremented["draft_tokens"] == native_tokens,
        },
        "missing_state_keys": missing,
        "unexpected_state_keys": unexpected,
        "assistant_model_path": str(args.assistant_model_path),
        "torch_default_dtype": str(torch.get_default_dtype()),
        "model_dtype": str(model_dtype),
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assistant-model-path", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def write_blocked(path: Path, blocker: str, **extra: object) -> None:
    result = {
        "schema_version": 1,
        "status": "blocked",
        "blocker": blocker,
    }
    result.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(blocker, file=sys.stderr)


def assistant_quantized_keys(model_path: Path) -> list[str]:
    from safetensors import safe_open

    weights = model_path / "model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(str(weights))
    with safe_open(str(weights), framework="pt") as handle:
        return [
            key
            for key in handle.keys()
            if key.endswith(".scales") or key.endswith(".biases")
        ]


def assistant_config(model_path: Path, config_cls):
    raw = json.loads((model_path / "config.json").read_text())
    text_config = dict(raw["text_config"])
    text_config["model_type"] = "gemma4_text"
    return config_cls(
        text_config=text_config,
        backbone_hidden_size=raw["backbone_hidden_size"],
        use_ordered_embeddings=raw.get("use_ordered_embeddings", False),
        num_centroids=raw.get("num_centroids", 2048),
        centroid_intermediate_top_k=raw.get("centroid_intermediate_top_k", 32),
        tie_word_embeddings=raw.get("tie_word_embeddings", True),
    )


def tie_lm_head(model) -> None:
    if not hasattr(model, "lm_head"):
        return
    embed_tokens = getattr(getattr(model, "model", None), "embed_tokens", None)
    if embed_tokens is None:
        return
    model.lm_head.weight = embed_tokens.weight


def load_payload(path: Path, safe_open):
    tensors = {}
    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata() or {}
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)
    required = [
        "hidden.last",
        "hidden.full_attention_key",
        "hidden.full_attention_value",
        "hidden.sliding_attention_key",
        "hidden.sliding_attention_value",
        "target.token_embeddings",
    ]
    missing = [key for key in required if key not in tensors]
    if missing:
        raise RuntimeError(f"payload missing tensors: {missing}")
    return tensors, metadata


def token_embedding_map(
    payload: dict, metadata: dict[str, str], device: str, dtype
) -> dict[int, object]:
    token_ids = [
        int(token_id)
        for token_id in metadata.get("target.token_embeddings.token_ids", "").split(",")
        if token_id
    ]
    embeddings = payload["target.token_embeddings"].to(device=device, dtype=dtype)
    if embeddings.shape[1] != len(token_ids):
        raise RuntimeError(
            f"target.token_embeddings shape/token-id mismatch: shape={tuple(embeddings.shape)} token_ids={token_ids}"
        )
    return {token_id: embeddings[:, index : index + 1, :] for index, token_id in enumerate(token_ids)}


def run_variant(
    *,
    torch,
    model,
    token_embeddings: dict[int, object],
    shared_kv_states: dict[str, tuple[object, object]],
    last_hidden,
    last_token_id: int,
    first_position: int,
    block_size: int,
    increment_positions: bool,
) -> dict[str, object]:
    current_hidden = last_hidden
    token_id = last_token_id
    draft_tokens: list[int] = []
    positions: list[int] = []
    for step in range(block_size):
        if token_id not in token_embeddings:
            raise RuntimeError(f"payload does not include target embedding for token {token_id}")
        position = first_position + step if increment_positions else first_position
        positions.append(position)
        inputs_embeds = torch.cat([token_embeddings[token_id], current_hidden], dim=-1)
        with torch.no_grad():
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=None,
                position_ids=torch.tensor([[position]], dtype=torch.long, device=inputs_embeds.device),
                shared_kv_states=shared_kv_states,
                use_cache=False,
            )
        token_id = int(outputs.logits.argmax(dim=-1).item())
        draft_tokens.append(token_id)
        current_hidden = outputs.last_hidden_state
    return {
        "position_mode": "incremented" if increment_positions else "pinned",
        "positions": positions,
        "draft_tokens": draft_tokens,
    }


if __name__ == "__main__":
    raise SystemExit(main())
