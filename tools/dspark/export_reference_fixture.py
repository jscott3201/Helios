#!/usr/bin/env python3
"""Create or block an XR60 DeepSpec/PyTorch DSpark reference fixture."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dspark_common import (
    EXPECTED_DSPARK_REVISION,
    EXPECTED_MODEL_ID,
    EXPECTED_TARGET_MODEL_ID,
    EXPECTED_TARGET_REVISION,
    config_summary,
    environment_summary,
    reference_revisions,
    render_blockers,
    sha256_text,
    write_json,
)


DEFAULT_DRAFT_PATH = Path("artifacts/drafts/dspark-gemma4-12b-block7")
DEFAULT_OUT_DIR = Path("benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    command = " ".join(sys.argv)
    revision = args.revision or EXPECTED_DSPARK_REVISION
    draft, blockers = config_summary(args.draft_path, args.model_id, revision)
    env = environment_summary()
    for package in ["torch", "safetensors", "transformers"]:
        if not env["packages"][package]:
            blockers.append(f"missing Python package required for reference fixture: {package}")
    if not env["packages"]["deepspec"]:
        blockers.append("DeepSpec Python reference package is not installed or importable as `deepspec`")

    prompts = [
        {
            "id": "xr60_smoke_tokens",
            "token_ids": args.prompt_token_ids,
            "sha256": sha256_text(",".join(str(token) for token in args.prompt_token_ids)),
        }
    ]

    manifest = {
        "schema_version": 1,
        "goal": "XR60-dspark-native-mlx",
        "phase": "01-reference-fixtures",
        "status": "blocked" if blockers else "ready",
        "command": command,
        "environment": env,
        "reference_revisions": reference_revisions(),
        "deepseek_dspark": draft,
        "target_model_id": args.target_model_id,
        "tokenizer_revision": args.tokenizer_revision,
        "prompts": prompts,
        "expected_fixture_fields": [
            "input_token_ids",
            "target_hidden_taps",
            "target_last_hidden",
            "dspark_base_logits",
            "dspark_markov_logits",
            "confidence",
            "greedy_draft_tokens",
        ],
        "blockers": blockers,
    }
    write_json(args.out_dir / "manifest.json", manifest)
    (args.out_dir / "blockers.md").write_text(
        render_blockers("XR60 reference fixture", blockers, command),
        encoding="utf-8",
    )
    if blockers:
        if args.allow_blocked:
            return 0
        return 2

    # The heavy PyTorch reference pass intentionally stays behind the blocker
    # checks. Once DeepSpec is importable and weights are present, this script is
    # the fixture entry point to fill the fields above.
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-path", type=Path, default=DEFAULT_DRAFT_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-id", default=EXPECTED_MODEL_ID)
    parser.add_argument("--revision", default=EXPECTED_DSPARK_REVISION)
    parser.add_argument("--target-model-id", default=EXPECTED_TARGET_MODEL_ID)
    parser.add_argument("--tokenizer-revision", default=EXPECTED_TARGET_REVISION)
    parser.add_argument(
        "--prompt-token-ids",
        default="2,106,107",
        help="comma-separated tiny smoke prompt token ids",
    )
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="write manifest/blockers and exit 0 even when fixture generation is blocked",
    )
    args = parser.parse_args()
    args.prompt_token_ids = parse_token_ids(args.prompt_token_ids)
    return args


def parse_token_ids(value: str) -> list[int]:
    tokens = [item.strip() for item in value.split(",") if item.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("prompt token list must not be empty")
    return [int(token) for token in tokens]


if __name__ == "__main__":
    raise SystemExit(main())
