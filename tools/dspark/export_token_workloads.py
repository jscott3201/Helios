#!/usr/bin/env python3
"""Export real-context workload prompt token IDs for XR60 DSpark diagnosis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dspark_common import environment_summary, render_blockers, sha256_text, write_json


DEFAULT_MANIFEST = Path("benchmarks/workloads/real-contexts/workloads.jsonl")
DEFAULT_TOKENIZER = Path("artifacts/models/gemma-4-12B-it-4bit")
DEFAULT_OUT = Path("benchmarks/out/XR60-dspark-native-mlx/real-context-token-workloads.jsonl")
DEFAULT_WORKLOADS = ["chat_short_1k_001", "mtp_candidate_1k_001"]


def main() -> int:
    args = parse_args()
    command = " ".join(sys.argv)
    blockers: list[str] = []
    records = read_jsonl(args.manifest, blockers)
    selected = select_records(records, args.workloads, blockers)
    tokenizer = load_tokenizer(args.tokenizer_path, blockers)
    exported: list[dict[str, Any]] = []
    if tokenizer is not None:
        exported = export_records(selected, tokenizer, args, blockers)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.out.parent / f"{args.out.stem}.manifest.json"
    blockers_path = args.out.parent / f"{args.out.stem}.blockers.md"
    status = "passed" if not blockers else "blocked"
    result = {
        "schema_version": 1,
        "goal": "XR60-dspark-native-mlx",
        "phase": "real-context-token-workload-export",
        "status": status,
        "command": command,
        "manifest": str(args.manifest),
        "tokenizer_path": str(args.tokenizer_path),
        "out": str(args.out),
        "requested_workloads": args.workloads,
        "exported_workloads": [record["workload_id"] for record in exported],
        "environment": environment_summary(),
        "blockers": blockers,
    }
    write_json(summary_path, result)
    blockers_path.write_text(
        render_blockers("XR60 token workload export", blockers, command),
        encoding="utf-8",
    )
    if blockers and not args.allow_mismatch:
        return 2

    write_jsonl(args.out, exported)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export tokenizer IDs from real-context prompts for dspark_fixed_block_matrix."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--workloads",
        default=",".join(DEFAULT_WORKLOADS),
        help="Comma-separated workload IDs, or 'all'. Defaults to a bounded 1K pair.",
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Write output even when SHA/token-count validation records blockers.",
    )
    return parser.parse_args()


def read_jsonl(path: Path, blockers: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        blockers.append(f"missing workload manifest: {path}")
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except Exception as error:  # noqa: BLE001 - captured in blockers artifact.
                blockers.append(f"could not parse {path}:{line_number}: {error}")
                continue
            if not isinstance(value, dict):
                blockers.append(f"{path}:{line_number} is not a JSON object")
                continue
            records.append(value)
    return records


def select_records(
    records: list[dict[str, Any]], workload_arg: str, blockers: list[str]
) -> list[dict[str, Any]]:
    if workload_arg.strip() == "all":
        return records
    requested = [item.strip() for item in workload_arg.split(",") if item.strip()]
    if not requested:
        blockers.append("--workloads selected no workload IDs")
        return []
    by_id = {str(record.get("workload_id")): record for record in records}
    missing = [workload_id for workload_id in requested if workload_id not in by_id]
    if missing:
        blockers.append(f"unknown workload IDs in manifest: {','.join(missing)}")
    return [by_id[workload_id] for workload_id in requested if workload_id in by_id]


def load_tokenizer(tokenizer_path: Path, blockers: list[str]) -> Any | None:
    if not tokenizer_path.exists():
        blockers.append(f"missing tokenizer path: {tokenizer_path}")
        return None
    try:
        from transformers import AutoTokenizer
    except Exception as error:  # noqa: BLE001 - recorded in blockers artifact.
        blockers.append(f"transformers AutoTokenizer unavailable: {error}")
        return None
    try:
        return AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    except Exception as error:  # noqa: BLE001 - recorded in blockers artifact.
        blockers.append(f"could not load tokenizer from {tokenizer_path}: {error}")
        return None


def export_records(
    records: list[dict[str, Any]],
    tokenizer: Any,
    args: argparse.Namespace,
    blockers: list[str],
) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    tokenizer_class = type(tokenizer).__name__
    for record in records:
        workload_id = str(record.get("workload_id") or "")
        prompt_path = Path(str(record.get("prompt_path") or ""))
        if not workload_id:
            blockers.append(f"manifest record has empty workload_id for prompt {prompt_path}")
            continue
        if not prompt_path.exists():
            blockers.append(f"{workload_id} missing prompt file: {prompt_path}")
            continue
        prompt = prompt_path.read_text(encoding="utf-8")
        prompt_sha256 = sha256_text(prompt)
        manifest_sha256 = str(record.get("prompt_sha256") or "")
        if prompt_sha256 != manifest_sha256:
            blockers.append(
                f"{workload_id} prompt_sha256 mismatch: manifest={manifest_sha256} actual={prompt_sha256}"
            )
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        actual_context_tokens = len(token_ids)
        manifest_actual = int(record.get("actual_context_tokens") or -1)
        if actual_context_tokens != manifest_actual:
            blockers.append(
                f"{workload_id} token count mismatch: manifest={manifest_actual} actual={actual_context_tokens}"
            )
        exported.append(
            {
                "schema_version": 1,
                "workload_id": workload_id,
                "family": record.get("family"),
                "prompt_path": str(prompt_path),
                "prompt_sha256": prompt_sha256,
                "tokenizer_path": str(args.tokenizer_path),
                "tokenizer_class": tokenizer_class,
                "actual_context_tokens": actual_context_tokens,
                "manifest_actual_context_tokens": manifest_actual,
                "max_new_tokens": record.get("max_new_tokens"),
                "token_ids": token_ids,
            }
        )
    return exported


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
