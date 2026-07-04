#!/usr/bin/env python3
"""Prepare an XR60 DSpark MLX conversion manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dspark_common import (
    EXPECTED_DSPARK_REVISION,
    EXPECTED_MODEL_ID,
    config_summary,
    environment_summary,
    reference_revisions,
    render_blockers,
    write_json,
)


DEFAULT_DRAFT_PATH = Path("artifacts/drafts/dspark-gemma4-12b-block7")
DEFAULT_OUT_DIR = Path("benchmarks/out/XR60-dspark-native-mlx/03-mlx-parity")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join(sys.argv)

    revision = args.revision or EXPECTED_DSPARK_REVISION
    draft, blockers = config_summary(args.draft_path, args.model_id, revision)
    env = environment_summary()
    if not env["packages"]["mlx"]:
        blockers.append("missing Python package required for MLX conversion: mlx")
    if not env["packages"]["safetensors"]:
        blockers.append("missing Python package required for safetensors loading: safetensors")
    if draft.get("safetensors_file_count", 0) == 0:
        blockers.append("cannot convert DSpark weights until model.safetensors is present locally")

    manifest = {
        "schema_version": 1,
        "goal": "XR60-dspark-native-mlx",
        "phase": "03-mlx-parity",
        "status": "blocked" if blockers else "ready",
        "command": command,
        "environment": env,
        "reference_revisions": reference_revisions(),
        "source": draft,
        "output_path": str(args.output_path),
        "conversion_status": "not_started" if blockers else "ready_for_tensor_mapping",
        "required_modules": [
            "selected target hidden tap projection fc",
            "hidden norm",
            "5-layer Gemma-style draft stack",
            "lm head",
            "rank-256 Markov head",
            "confidence head",
        ],
        "blockers": blockers,
    }
    write_json(args.out_dir / "conversion_manifest.json", manifest)
    (args.out_dir / "blockers.md").write_text(
        render_blockers("XR60 MLX conversion", blockers, command),
        encoding="utf-8",
    )
    if blockers and not args.allow_blocked:
        return 2
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-path", type=Path, default=DEFAULT_DRAFT_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/drafts/dspark-gemma4-12b-block7-mlx"))
    parser.add_argument("--model-id", default=EXPECTED_MODEL_ID)
    parser.add_argument("--revision", default=EXPECTED_DSPARK_REVISION)
    parser.add_argument("--allow-blocked", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
