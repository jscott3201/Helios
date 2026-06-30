#!/usr/bin/env python3
"""Reference layer trace for the local MLX-LM Gemma 4 text implementation."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.gemma4_text import geglu as gemma4_geglu
from mlx_lm.utils import load_model


dump_dir: Path | None = None
hidden_dump_dir: Path | None = None


def parse_token_ids(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def dump(label: str, x: mx.array) -> None:
    if dump_dir is None:
        return
    dump_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(dump_dir / f"{label}.safetensors"), {"tensor": x})


def dump_hidden(label: str, x: mx.array) -> None:
    if hidden_dump_dir is None:
        return
    hidden_dump_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(hidden_dump_dir / f"{label}.safetensors"), {"tensor": x})


def stats3(label: str, x: mx.array) -> None:
    last = x[0, -1]
    rms = mx.sqrt(mx.mean(mx.square(last))).astype(mx.float32)
    sample = last[:4].astype(mx.float32)
    mx.eval(rms, sample)
    values = ",".join(str(float(v)) for v in sample.tolist())
    print(f"gemma4d_ref_trace {label} last_rms={float(rms.item())} first4=[{values}]")


def stats4(label: str, x: mx.array) -> None:
    last = x[0, 0, -1]
    rms = mx.sqrt(mx.mean(mx.square(last))).astype(mx.float32)
    sample = last[:4].astype(mx.float32)
    mx.eval(rms, sample)
    values = ",".join(str(float(v)) for v in sample.tolist())
    print(f"gemma4d_ref_trace {label} head0_last_rms={float(rms.item())} head0_first4=[{values}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--token-ids", default="9259,236772,236772")
    parser.add_argument("--dump-dir")
    parser.add_argument("--dump-hidden-dir")
    parser.add_argument("--layer-index", type=int, default=0)
    args = parser.parse_args()

    global dump_dir, hidden_dump_dir
    dump_dir = Path(args.dump_dir) if args.dump_dir else None
    hidden_dump_dir = Path(args.dump_hidden_dir) if args.dump_hidden_dir else None

    model, _ = load_model(
        Path(args.model_path),
        lazy=False,
        strict=False,
        model_config={"model_type": "gemma4"},
    )
    language_model = getattr(model, "language_model", model)
    text_model = language_model.model
    layer_index = args.layer_index
    if layer_index < 0 or layer_index >= len(text_model.layers):
        raise ValueError(f"layer-index out of range: {layer_index}")
    layer = text_model.layers[layer_index]
    attention = layer.self_attn

    token_ids = parse_token_ids(args.token_ids)
    inputs = mx.array(token_ids, dtype=mx.uint32)[None]
    h = text_model.embed_tokens(inputs) * text_model.embed_scale
    cache = [None] * len(text_model.layers)
    masks = text_model._make_masks(h, cache)
    intermediates = [(None, None)] * len(text_model.layers)
    for idx in range(layer_index):
        decoder_layer = text_model.layers[idx]
        shared_kv, offset = intermediates[text_model.previous_kvs[idx]]
        h, kvs, offset = decoder_layer(
            h,
            masks[idx],
            cache[idx],
            per_layer_input=None,
            shared_kv=shared_kv,
            offset=offset,
        )
        intermediates[idx] = (kvs, offset)

    layer_input = h
    h = layer.input_layernorm(layer_input)
    dump("input_norm", h)
    stats3("layer0.input_norm", h)

    batch, sequence_len, _ = h.shape
    queries = attention.q_proj(h)
    dump("q_proj", queries)
    stats3("layer0.q_proj", queries)
    queries = queries.reshape(batch, sequence_len, attention.n_heads, attention.head_dim)
    queries = attention.q_norm(queries)

    keys = attention.k_proj(h)
    dump("k_proj", keys)
    stats3("layer0.k_proj", keys)
    keys = keys.reshape(batch, sequence_len, attention.n_kv_heads, attention.head_dim)
    values = keys
    if not attention.use_k_eq_v:
        values = attention.v_proj(h)
        dump("v_proj", values)
        stats3("layer0.v_proj", values)
        values = values.reshape(batch, sequence_len, attention.n_kv_heads, attention.head_dim)

    offset = 0
    keys = attention.k_norm(keys)
    keys = keys.transpose(0, 2, 1, 3)
    dump("k_norm", keys)
    stats4("layer0.k_norm", keys)
    keys = attention.rope(keys, offset=offset)
    dump("k_rope", keys)
    stats4("layer0.k_rope", keys)

    values = attention.v_norm(values)
    values = values.transpose(0, 2, 1, 3)
    dump("v_norm", values)
    stats4("layer0.v_norm", values)

    queries = queries.transpose(0, 2, 1, 3)
    dump("q_norm", queries)
    stats4("layer0.q_norm", queries)
    queries = attention.rope(queries, offset=offset)
    dump("q_rope", queries)
    stats4("layer0.q_rope", queries)

    mask = masks[layer_index]
    output = scaled_dot_product_attention(
        queries,
        keys,
        values,
        cache=None,
        scale=attention.scale,
        mask=mask,
    )
    dump("sdpa", output)
    stats4("layer0.sdpa", output)
    output = output.transpose(0, 2, 1, 3).reshape(batch, sequence_len, -1)
    dump("attn_merge", output)
    stats3("layer0.attn_merge", output)
    output = attention.o_proj(output)
    dump("attn_out", output)
    stats3("layer0.attn_out", output)

    residual = layer_input
    h = layer.post_attention_layernorm(output)
    dump("post_attn_norm", h)
    h = residual + h
    dump("attn_residual", h)
    mlp_residual = h
    h = layer.pre_feedforward_layernorm(h)
    dump("pre_ff_norm", h)
    gate = layer.mlp.gate_proj(h)
    dump("gate_proj", gate)
    up = layer.mlp.up_proj(h)
    dump("up_proj", up)
    h = gemma4_geglu(gate, up)
    dump("geglu", h)
    h = layer.mlp.down_proj(h)
    dump("down_proj", h)
    h = layer.post_feedforward_layernorm(h)
    dump("post_ff_norm", h)
    h = mlp_residual + h
    dump("ff_residual", h)
    h = h * layer.layer_scalar
    dump("layer_scalar", h)

    if hidden_dump_dir is None:
        return

    h = text_model.embed_tokens(inputs) * text_model.embed_scale
    dump_hidden("embed", h)
    cache = [None] * len(text_model.layers)
    masks = text_model._make_masks(h, cache)
    intermediates = [(None, None)] * len(text_model.layers)
    for idx, (decoder_layer, cache_entry, mask, prev_idx) in enumerate(
        zip(text_model.layers, cache, masks, text_model.previous_kvs)
    ):
        shared_kv, offset = intermediates[prev_idx]
        h, kvs, offset = decoder_layer(
            h,
            mask,
            cache_entry,
            per_layer_input=None,
            shared_kv=shared_kv,
            offset=offset,
        )
        intermediates[idx] = (kvs, offset)
        dump_hidden(f"layer{idx}", h)

    h = text_model.norm(h)
    dump_hidden("final_norm", h)
    last = h[:, -1:, :]
    if language_model.tie_word_embeddings:
        logits = text_model.embed_tokens.as_linear(last)
    else:
        logits = language_model.lm_head(last)
    if language_model.final_logit_softcapping is not None:
        logits = mx.tanh(logits / language_model.final_logit_softcapping) * language_model.final_logit_softcapping
    logits = logits.reshape(-1)
    dump_hidden("logits", logits)

    candidate_ids = mx.array([236761, 236772], dtype=mx.int32)
    candidate_logits = logits[candidate_ids].astype(mx.float32)
    greedy = mx.argmax(logits)
    mx.eval(candidate_logits, greedy)
    print(
        "gemma4d_ref_trace logits "
        f"greedy={int(greedy.item())} "
        f"236761={float(candidate_logits[0].item())} "
        f"236772={float(candidate_logits[1].item())}"
    )


if __name__ == "__main__":
    main()
