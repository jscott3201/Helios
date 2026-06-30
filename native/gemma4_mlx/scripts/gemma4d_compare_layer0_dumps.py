#!/usr/bin/env python3
"""Compare native and MLX-LM layer-0 safetensors dumps."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import mlx.core as mx


def load_tensor(path: Path) -> mx.array:
    loaded = mx.load(str(path))
    if isinstance(loaded, dict):
        return loaded["tensor"]
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-dir", required=True)
    parser.add_argument("--ref-dir", required=True)
    args = parser.parse_args()

    native_dir = Path(args.native_dir)
    ref_dir = Path(args.ref_dir)
    native_labels = {path.stem for path in native_dir.glob("*.safetensors")}
    ref_labels = {path.stem for path in ref_dir.glob("*.safetensors")}

    def sort_key(label: str) -> tuple[int, int, str]:
        if label == "embed":
            return (0, -1, label)
        match = re.fullmatch(r"layer(\d+)", label)
        if match:
            return (1, int(match.group(1)), label)
        if label == "final_norm":
            return (2, 0, label)
        if label == "logits":
            return (3, 0, label)
        return (4, 0, label)

    labels = sorted(native_labels & ref_labels, key=sort_key)
    if not labels:
        raise SystemExit("no common safetensors labels to compare")

    for label in labels:
        native_path = native_dir / f"{label}.safetensors"
        ref_path = ref_dir / f"{label}.safetensors"
        if not native_path.exists() or not ref_path.exists():
            print(f"{label}: missing")
            continue

        native = load_tensor(native_path).astype(mx.float32)
        ref = load_tensor(ref_path).astype(mx.float32)
        if native.shape != ref.shape:
            print(f"{label}: shape native={native.shape} ref={ref.shape}")
            continue

        diff = mx.abs(native - ref)
        max_diff = mx.max(diff)
        mean_diff = mx.mean(diff)
        ref_rms = mx.sqrt(mx.mean(mx.square(ref)))
        native_rms = mx.sqrt(mx.mean(mx.square(native)))
        mx.eval(max_diff, mean_diff, ref_rms, native_rms)
        print(
            f"{label}: shape={native.shape} max={float(max_diff.item()):.8g} "
            f"mean={float(mean_diff.item()):.8g} native_rms={float(native_rms.item()):.8g} "
            f"ref_rms={float(ref_rms.item()):.8g}"
        )


if __name__ == "__main__":
    main()
