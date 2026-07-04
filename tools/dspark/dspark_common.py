#!/usr/bin/env python3
"""Shared helpers for XR60 DSpark fixture and parity tooling."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

EXPECTED_ARCHITECTURE = "Gemma4DSparkModel"
EXPECTED_MODEL_ID = "deepseek-ai/dspark_gemma4_12b_block7"
EXPECTED_DSPARK_REVISION = "2fa72e765eec2965fc4d86a8663ce6769eba6218"
EXPECTED_DEEPSPEC_REPO = "deepseek-ai/DeepSpec"
EXPECTED_DEEPSPEC_COMMIT = "afdfa7c9382a3341a3e6f17756dd816da79f132c"
EXPECTED_TARGET_MODEL_ID = "google/gemma-4-12B-it"
EXPECTED_TARGET_REVISION = "5926caa4ec0cac5cbfadaf4077420520de1d5205"
EXPECTED_BLOCK_SIZE = 7
EXPECTED_TARGET_LAYER_IDS = [5, 17, 29, 41, 46]
EXPECTED_NUM_DRAFT_LAYERS = 5
EXPECTED_MARKOV_RANK = 256
EXPECTED_NUM_ANCHORS = 512
EXPECTED_MASK_TOKEN_ID = 4
EXPECTED_DTYPE = "bfloat16"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def command_stdout(args: list[str]) -> str:
    try:
        output = subprocess.run(
            args,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as error:  # noqa: BLE001 - recorded in manifest, not re-raised.
        return f"unavailable:{error}"
    return output.stdout.strip()


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def config_summary(draft_path: Path, model_id: str, revision: str | None) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    config_path = draft_path / "config.json"
    if not config_path.exists():
        return (
            {
                "path": str(draft_path),
                "exists": draft_path.exists(),
                "config_exists": False,
                "model_id": model_id,
                "revision": revision,
            },
            [f"missing DSpark config: {config_path}"],
        )

    config = read_json(config_path)
    architecture = first(config.get("architectures"), default="<missing>")
    checks = {
        "architecture": architecture == EXPECTED_ARCHITECTURE,
        "block_size": config.get("block_size") == EXPECTED_BLOCK_SIZE,
        "target_layer_ids": config.get("target_layer_ids") == EXPECTED_TARGET_LAYER_IDS,
        "num_hidden_layers": config.get("num_hidden_layers") == EXPECTED_NUM_DRAFT_LAYERS,
        "markov_rank": config.get("markov_rank") == EXPECTED_MARKOV_RANK,
        "num_anchors": config.get("num_anchors") == EXPECTED_NUM_ANCHORS,
        "mask_token_id": config.get("mask_token_id") == EXPECTED_MASK_TOKEN_ID,
        "dtype": config.get("dtype") == EXPECTED_DTYPE,
        "confidence_head_with_markov": config.get("confidence_head_with_markov") is True,
        "enable_confidence_head": config.get("enable_confidence_head") is True,
    }
    for name, passed in checks.items():
        if not passed:
            blockers.append(f"DSpark config check failed: {name}")

    safetensors = sorted(draft_path.glob("*.safetensors"))
    total_bytes = sum(path.stat().st_size for path in safetensors if path.exists())
    if not safetensors:
        blockers.append(f"missing DSpark safetensors weights under {draft_path}")

    summary = {
        "path": str(draft_path),
        "exists": draft_path.exists(),
        "model_id": model_id,
        "revision": revision,
        "config_exists": True,
        "config_sha256": sha256_file(config_path),
        "architecture": architecture,
        "model_type": config.get("model_type"),
        "target_model_type": config.get("target_model_type"),
        "target_text_model_type": config.get("target_text_model_type"),
        "block_size": config.get("block_size"),
        "target_layer_ids": config.get("target_layer_ids"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "hidden_size": config.get("hidden_size"),
        "intermediate_size": config.get("intermediate_size"),
        "vocab_size": config.get("vocab_size"),
        "markov_head_type": config.get("markov_head_type"),
        "markov_rank": config.get("markov_rank"),
        "num_anchors": config.get("num_anchors"),
        "mask_token_id": config.get("mask_token_id"),
        "dtype": config.get("dtype"),
        "confidence_head_with_markov": config.get("confidence_head_with_markov"),
        "enable_confidence_head": config.get("enable_confidence_head"),
        "checks": checks,
        "safetensors": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in safetensors
        ],
        "safetensors_file_count": len(safetensors),
        "safetensors_total_bytes": total_bytes,
    }
    return summary, blockers


def environment_summary() -> dict[str, Any]:
    return {
        "timestamp_unix": int(time.time()),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "git_sha": command_stdout(["git", "rev-parse", "HEAD"]),
        "git_status_short": command_stdout(["git", "status", "--short"]),
        "packages": {
            "torch": package_available("torch"),
            "safetensors": package_available("safetensors"),
            "transformers": package_available("transformers"),
            "mlx": package_available("mlx"),
            "mlx_lm": package_available("mlx_lm"),
            "deepspec": package_available("deepspec"),
        },
    }


def reference_revisions() -> dict[str, str]:
    return {
        "deepspec_repo": EXPECTED_DEEPSPEC_REPO,
        "deepspec_commit": EXPECTED_DEEPSPEC_COMMIT,
        "dspark_model_id": EXPECTED_MODEL_ID,
        "dspark_revision": EXPECTED_DSPARK_REVISION,
        "target_model_id": EXPECTED_TARGET_MODEL_ID,
        "target_revision": EXPECTED_TARGET_REVISION,
    }


def render_blockers(title: str, blockers: list[str], command: str) -> str:
    if not blockers:
        return f"# {title} blockers\n\nNo blockers recorded.\n"
    lines = [f"# {title} blockers", ""]
    for index, blocker in enumerate(blockers, start=1):
        lines.extend(
            [
                f"## Blocker {index}: {blocker}",
                "",
                f"- Time: {int(time.time())}",
                f"- Git SHA: {command_stdout(['git', 'rev-parse', 'HEAD'])}",
                "- Phase/Gate: XR60 reference/parity preparation",
                f"- Command: `{command}`",
                "- Expected: deterministic DSpark fixture/parity artifact can be produced",
                f"- Observed: {blocker}",
                "- Next input needed: install/provide the missing dependency or artifact, then rerun the command",
                "",
            ]
        )
    return "\n".join(lines)


def first(value: Any, default: str) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, str):
        return value
    return default
