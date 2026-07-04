#!/usr/bin/env python3
"""Compare DeepSpec native-tap fixtures with Helios native DSpark trace records."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

from dspark_common import environment_summary, read_json, render_blockers, write_json


DEFAULT_REFERENCE = Path(
    "benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures/native-tap/reference_fixture.json"
)
DEFAULT_RECORDS = Path("benchmarks/out/XR60-dspark-native-mlx/warm-anchor-matrix/records.jsonl")
DEFAULT_OUT_DIR = Path("benchmarks/out/XR60-dspark-native-mlx/03-mlx-parity/native-trace")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join(sys.argv)
    blockers: list[str] = []
    comparisons: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []

    if not args.reference.exists():
        blockers.append(f"missing DeepSpec native-tap reference fixture: {args.reference}")
    if not args.records.exists():
        blockers.append(f"missing Helios native trace records: {args.records}")

    if not blockers:
        reference = read_json(args.reference)
        fixtures = fixtures_by_workload(reference, blockers)
        records = read_jsonl(args.records, blockers)
        for record_index, record in enumerate(records):
            workload_id = str(record.get("workload_id"))
            fixture = fixtures.get(workload_id)
            if fixture is None:
                skipped_records.append(
                    {
                        "record_index": record_index,
                        "workload_id": workload_id,
                        "reason": "no matching reference fixture",
                    }
                )
                continue
            comparison = compare_record(
                record_index=record_index,
                record=record,
                fixture=fixture,
                trace_index=args.trace_index,
                logit_tolerance=args.logit_tolerance,
                margin_tolerance=args.margin_tolerance,
                confidence_tolerance=args.confidence_tolerance,
            )
            comparisons.append(comparison)
        if not comparisons:
            blockers.append("no native trace records matched a DeepSpec reference fixture")
        for comparison in comparisons:
            for check in comparison["checks"]:
                if not check["passed"]:
                    blockers.append(
                        f"record {comparison['record_index']} {comparison['workload_id']} "
                        f"{check['field']} failed: {check['message']}"
                    )

    result = {
        "schema_version": 1,
        "goal": "XR60-dspark-native-mlx",
        "phase": "03-mlx-parity/native-trace",
        "status": "passed" if not blockers else "blocked",
        "command": command,
        "environment": environment_summary(),
        "reference": str(args.reference),
        "records": str(args.records),
        "trace_index": args.trace_index,
        "logit_tolerance": args.logit_tolerance,
        "margin_tolerance": args.margin_tolerance,
        "confidence_tolerance": args.confidence_tolerance,
        "comparisons": comparisons,
        "skipped_records": skipped_records,
        "blockers": blockers,
    }
    write_json(args.out_dir / "parity_report.json", result)
    (args.out_dir / "blockers.md").write_text(
        render_blockers("XR60 native trace parity", blockers, command),
        encoding="utf-8",
    )
    if blockers and not args.allow_blocked:
        return 2
    return 0


def fixtures_by_workload(reference: dict[str, Any], blockers: list[str]) -> dict[str, dict[str, Any]]:
    fixtures = reference.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        blockers.append("DeepSpec reference fixture has no fixtures")
        return {}
    result: dict[str, dict[str, Any]] = {}
    for fixture in fixtures:
        workload_id = fixture.get("workload_id")
        if not isinstance(workload_id, str):
            blockers.append("DeepSpec reference fixture is missing workload_id")
            continue
        result[workload_id] = fixture
    return result


def read_jsonl(path: Path, blockers: list[str]) -> list[dict[str, Any]]:
    records = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    record = read_json_line(line, path, line_number, blockers)
                    if record is not None:
                        records.append(record)
    except Exception as error:  # noqa: BLE001 - reported in parity artifact.
        blockers.append(f"could not read native trace records {path}: {error}")
    return records


def read_json_line(line: str, path: Path, line_number: int, blockers: list[str]) -> dict[str, Any] | None:
    try:
        import json

        return json.loads(line)
    except Exception as error:  # noqa: BLE001 - reported in parity artifact.
        blockers.append(f"could not parse {path}:{line_number}: {error}")
        return None


def compare_record(
    *,
    record_index: int,
    record: dict[str, Any],
    fixture: dict[str, Any],
    trace_index: int,
    logit_tolerance: float,
    margin_tolerance: float,
    confidence_tolerance: float,
) -> dict[str, Any]:
    trace_entries = record.get("verify_trace")
    trace = {}
    if isinstance(trace_entries, list) and trace_entries and trace_index < len(trace_entries):
        trace = trace_entries[trace_index]

    draft_tokens = list_or_empty(trace.get("draft_tokens"))
    draft_count = len(draft_tokens)
    reference_tokens = list_or_empty(fixture.get("greedy_draft_tokens"))
    reference_markov_logits = first_batch(fixture.get("dspark_markov_selected_logits"))
    reference_confidence = first_batch(fixture.get("confidence"))
    reference_margins = markov_margins(fixture.get("dspark_markov_top_k"))
    target_tokens = list_or_empty(trace.get("target_tokens"))

    checks = [
        exact_prefix_check(
            "greedy_draft_tokens",
            reference_tokens,
            draft_tokens,
            draft_count,
        ),
        float_prefix_check(
            "dspark_markov_selected_logits",
            reference_markov_logits,
            list_or_empty(trace.get("draft_logits")),
            draft_count,
            logit_tolerance,
        ),
        float_prefix_check(
            "confidence",
            reference_confidence,
            list_or_empty(trace.get("draft_confidence")),
            draft_count,
            confidence_tolerance,
        ),
        float_prefix_check(
            "dspark_markov_margin",
            reference_margins,
            list_or_empty(trace.get("draft_margins")),
            draft_count,
            margin_tolerance,
        ),
    ]
    target_prefix = target_tokens[:draft_count]
    return {
        "record_index": record_index,
        "workload_id": record.get("workload_id"),
        "scheduler": record.get("scheduler"),
        "record_scheduled_len": record.get("scheduled_len"),
        "trace_scheduled_len": trace.get("scheduled_len"),
        "warmup_target_tokens": record.get("warmup_target_tokens"),
        "trace_index": trace_index,
        "draft_count_compared": draft_count,
        "native_draft_tokens": draft_tokens,
        "reference_draft_prefix": reference_tokens[:draft_count],
        "target_token_prefix": target_prefix,
        "draft_matches_reference_prefix": draft_tokens == reference_tokens[:draft_count],
        "draft_matches_target_prefix": draft_tokens == target_prefix,
        "accepted_draft_count": trace.get("accepted_draft_count"),
        "committed_tokens": trace.get("committed_tokens"),
        "checks": checks,
    }


def exact_prefix_check(field: str, reference: list[Any], native: list[Any], count: int) -> dict[str, Any]:
    expected = reference[:count]
    observed = native[:count]
    passed = count > 0 and observed == expected
    return {
        "field": field,
        "kind": "exact_prefix",
        "passed": passed,
        "expected": expected,
        "observed": observed,
        "message": "passed" if passed else "prefix mismatch or empty native draft",
    }


def float_prefix_check(
    field: str,
    reference: list[Any],
    native: list[Any],
    count: int,
    tolerance: float,
) -> dict[str, Any]:
    expected = reference[:count]
    observed = native[:count]
    if count <= 0 or len(expected) != count or len(observed) != count:
        return {
            "field": field,
            "kind": "float_prefix",
            "passed": False,
            "tolerance": tolerance,
            "max_abs_error": None,
            "message": "missing or length-mismatched prefix",
        }
    errors = [abs(float(a) - float(b)) for a, b in zip(expected, observed)]
    max_error = max(errors, default=0.0)
    passed = math.isfinite(max_error) and max_error <= tolerance
    return {
        "field": field,
        "kind": "float_prefix",
        "passed": passed,
        "tolerance": tolerance,
        "max_abs_error": max_error,
        "expected": expected,
        "observed": observed,
        "message": "passed" if passed else f"max abs error {max_error} exceeds {tolerance}",
    }


def markov_margins(top_k: Any) -> list[float]:
    if not isinstance(top_k, dict):
        return []
    logits = top_k.get("logits")
    positions = first_batch(logits)
    margins = []
    for values in positions:
        if isinstance(values, list) and len(values) >= 2:
            margins.append(float(values[0]) - float(values[1]))
        elif isinstance(values, list) and len(values) == 1:
            margins.append(float(values[0]))
    return margins


def first_batch(value: Any) -> list[Any]:
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    if isinstance(value, list):
        return value
    return []


def list_or_empty(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--trace-index", type=int, default=0)
    parser.add_argument("--logit-tolerance", type=float, default=0.25)
    parser.add_argument("--margin-tolerance", type=float, default=0.25)
    parser.add_argument("--confidence-tolerance", type=float, default=0.01)
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args()
    if args.trace_index < 0:
        parser.error("--trace-index must be non-negative")
    if args.logit_tolerance < 0.0:
        parser.error("--logit-tolerance must be non-negative")
    if args.margin_tolerance < 0.0:
        parser.error("--margin-tolerance must be non-negative")
    if args.confidence_tolerance < 0.0:
        parser.error("--confidence-tolerance must be non-negative")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
