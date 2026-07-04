#!/usr/bin/env python3
"""Compare XR60 PyTorch reference and MLX DSpark parity JSON artifacts."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

from dspark_common import environment_summary, read_json, render_blockers, write_json


DEFAULT_OUT_DIR = Path("benchmarks/out/XR60-dspark-native-mlx/03-mlx-parity")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join(sys.argv)
    blockers: list[str] = []
    comparisons: list[dict[str, Any]] = []

    if not args.reference.exists():
        blockers.append(f"missing PyTorch reference fixture: {args.reference}")
    if not args.mlx.exists():
        blockers.append(f"missing MLX parity output: {args.mlx}")

    if not blockers:
        reference = read_json(args.reference)
        mlx = read_json(args.mlx)
        comparisons = compare(reference, mlx, args.logit_tolerance, args.confidence_tolerance)
        blockers.extend(item["message"] for item in comparisons if not item["passed"])

    result = {
        "schema_version": 1,
        "goal": "XR60-dspark-native-mlx",
        "phase": "03-mlx-parity",
        "status": "passed" if not blockers else "blocked",
        "command": command,
        "environment": environment_summary(),
        "reference": str(args.reference),
        "mlx": str(args.mlx),
        "logit_tolerance": args.logit_tolerance,
        "confidence_tolerance": args.confidence_tolerance,
        "comparisons": comparisons,
        "blockers": blockers,
    }
    write_json(args.out_dir / "parity_report.json", result)
    (args.out_dir / "blockers.md").write_text(
        render_blockers("XR60 MLX parity", blockers, command),
        encoding="utf-8",
    )
    if blockers and not args.allow_blocked:
        return 2
    return 0


def compare(reference: dict[str, Any], mlx: dict[str, Any], logit_tol: float, confidence_tol: float) -> list[dict[str, Any]]:
    checks = []
    checks.append(equal_list("greedy_draft_tokens", reference, mlx))
    checks.append(close_list("dspark_base_logits", reference, mlx, logit_tol))
    checks.append(close_list("dspark_markov_logits", reference, mlx, logit_tol))
    checks.append(close_list("confidence", reference, mlx, confidence_tol))
    return checks


def equal_list(field: str, reference: dict[str, Any], mlx: dict[str, Any]) -> dict[str, Any]:
    left = reference.get(field)
    right = mlx.get(field)
    passed = isinstance(left, list) and left == right
    return {
        "field": field,
        "kind": "exact_list",
        "passed": passed,
        "message": "passed" if passed else f"{field} mismatch or missing",
    }


def close_list(field: str, reference: dict[str, Any], mlx: dict[str, Any], tolerance: float) -> dict[str, Any]:
    left = reference.get(field)
    right = mlx.get(field)
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        return {
            "field": field,
            "kind": "float_list",
            "passed": False,
            "max_abs_error": None,
            "message": f"{field} missing or length mismatch",
        }
    errors = [abs(float(a) - float(b)) for a, b in zip(left, right)]
    max_error = max(errors, default=0.0)
    passed = math.isfinite(max_error) and max_error <= tolerance
    return {
        "field": field,
        "kind": "float_list",
        "passed": passed,
        "max_abs_error": max_error,
        "message": "passed" if passed else f"{field} max abs error {max_error} exceeds {tolerance}",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--mlx", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--logit-tolerance", type=float, default=0.5)
    parser.add_argument("--confidence-tolerance", type=float, default=0.02)
    parser.add_argument("--allow-blocked", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())

