#!/usr/bin/env python3
"""Dequantize the XR54 MLX affine-q4 assistant checkpoint for PyTorch parity.

The vendored Transformers reference cannot consume the local MLX q4 affine
triples directly. This utility converts every `<base>.weight/scales/biases`
triple with `mx.dequantize(group_size=64, bits=4, mode="affine")`, writes dense
float32 tensors, and adds the tied `lm_head.weight` expected by a bare
`Gemma4AssistantForCausalLM(config).load_state_dict(...)`.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


DEFAULT_SRC = Path("artifacts/models/gemma-4-12B-it-qat-assistant-4bit")
DEFAULT_OUT = Path("artifacts/models/gemma-4-12B-it-qat-assistant-dense-f32")


def main() -> int:
    args = parse_args()

    import mlx.core as mx
    from safetensors import safe_open
    from safetensors.torch import save_file

    config = json.loads((args.src / "config.json").read_text())
    quant = config.get("quantization", {})
    group_size = int(quant.get("group_size", 64))
    bits = int(quant.get("bits", 4))
    mode = str(quant.get("mode", "affine"))
    if mode != "affine":
        raise SystemExit(f"unsupported quant mode: {mode}")

    tensors: dict[str, torch.Tensor] = {}
    weights_path = args.src / "model.safetensors"
    with safe_open(str(weights_path), framework="pt") as handle:
        keys = list(handle.keys())
        quant_bases = sorted(
            key[: -len(".scales")] for key in keys if key.endswith(".scales")
        )
        consumed: set[str] = set()
        for base in quant_bases:
            dense = dequantize_base(
                handle,
                base,
                mx=mx,
                group_size=group_size,
                bits=bits,
                mode=mode,
            )
            tensors[f"{base}.weight"] = dense
            consumed.update({f"{base}.weight", f"{base}.scales"})
            if f"{base}.biases" in keys:
                consumed.add(f"{base}.biases")

        for key in keys:
            if key in consumed:
                continue
            tensors[key] = handle.get_tensor(key).float().contiguous()

    if args.tie_lm_head:
        embed_key = "model.embed_tokens.weight"
        if embed_key not in tensors:
            raise SystemExit(f"cannot tie lm_head.weight: missing {embed_key}")
        tensors["lm_head.weight"] = tensors[embed_key].clone()

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(args.out / "model.safetensors"),
        metadata={
            "source": str(args.src),
            "dequantization": f"mx.dequantize group_size={group_size} bits={bits} mode={mode}",
            "lm_head": "tied_to_model.embed_tokens.weight" if args.tie_lm_head else "unchanged",
        },
    )
    copy_reference_files(args.src, args.out)
    print(
        f"dequantized {len(quant_bases)} affine-q{bits} tensors; "
        f"wrote {len(tensors)} dense tensors"
    )
    print(f"out: {args.out / 'model.safetensors'}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--no-tie-lm-head",
        dest="tie_lm_head",
        action="store_false",
        help="do not add lm_head.weight from model.embed_tokens.weight",
    )
    parser.set_defaults(tie_lm_head=True)
    return parser.parse_args()


def dequantize_base(handle, base: str, *, mx, group_size: int, bits: int, mode: str):
    import numpy as np
    import torch

    packed = mx.array(handle.get_tensor(f"{base}.weight").numpy())
    scales = mx.array(handle.get_tensor(f"{base}.scales").float().numpy())
    bias_key = f"{base}.biases"
    biases = (
        mx.array(handle.get_tensor(bias_key).float().numpy())
        if bias_key in handle.keys()
        else None
    )
    dense = mx.dequantize(
        packed,
        scales,
        biases,
        group_size=group_size,
        bits=bits,
        mode=mode,
        dtype=mx.float32,
    )
    mx.eval(dense)
    return torch.from_numpy(np.asarray(dense)).contiguous()


def copy_reference_files(src: Path, out: Path) -> None:
    for name in [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ]:
        source = src / name
        if source.exists():
            shutil.copy(source, out / name)


if __name__ == "__main__":
    raise SystemExit(main())
