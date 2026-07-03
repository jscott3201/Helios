#!/usr/bin/env python3
"""Summarize XR55 n-block MTP records and optional sequential-oracle diffs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_EXPECTED_BLOCKS = (1, 2, 3, 4, 6, 8)
DEFAULT_EXPECTED_WORKLOADS = (
    "chat_short_1k_001",
    "tool_json_1k_001",
    "mtp_candidate_1k_001",
)
GUARDED_POLICY = "net_latency_guarded_5pct"
REPAIR_SUBTIMER_KEYS = (
    "repair_clone_ms",
    "repair_forward_ms",
    "repair_fallback_ms",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    if not records:
        raise SystemExit(f"{path}: no records found")
    return records


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"{path}: summary file not found") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: invalid JSON: {exc}") from exc


def parse_int_csv(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise SystemExit("expected at least one integer value")
    try:
        return [int(item) for item in values]
    except ValueError as exc:
        raise SystemExit(f"invalid integer CSV {raw!r}") from exc


def parse_str_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def record_key(record: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(record.get("workload_id")),
        str(record.get("trial_kind")),
        int(record.get("trial_index")),
        int(record.get("block_size")),
    )


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def percent(numerator: float, denominator: float) -> float:
    return 100.0 * safe_ratio(numerator, denominator)


def fmt_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def fmt_repair_subtimer(row: dict[str, Any], key: str) -> str:
    if not row["repair_subtimers_captured"]:
        return "not captured (pre-ABI-v3)"
    return fmt_float(float(row[key]), 1)


def required_float(
    record: dict[str, Any],
    section: str,
    field: str,
    failures: list[str],
    key: tuple[str, str, int, int],
) -> float | None:
    value = (record.get(section) or {}).get(field)
    if value is None:
        failures.append(f"{key}: missing {section}.{field}")
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        failures.append(f"{key}: invalid {section}.{field}={value!r}")
        return None


def block_coverage_failures(
    records: list[dict[str, Any]],
    expected_blocks: list[int],
) -> list[str]:
    measured_blocks = {
        int(record.get("block_size"))
        for record in records
        if record.get("measured")
    }
    missing = [block for block in expected_blocks if block not in measured_blocks]
    if missing:
        return [f"missing measured block sizes: {','.join(str(block) for block in missing)}"]
    return []


def workload_trial_failures(
    records: list[dict[str, Any]],
    expected_blocks: list[int],
    expected_workloads: list[str],
    min_measured_trials: int,
) -> list[str]:
    if min_measured_trials <= 0 or not expected_workloads:
        return []
    counts: dict[tuple[int, str], int] = defaultdict(int)
    for record in records:
        if not record.get("measured"):
            continue
        counts[(int(record["block_size"]), str(record["workload_id"]))] += 1

    failures: list[str] = []
    for block in expected_blocks:
        for workload in expected_workloads:
            count = counts[(block, workload)]
            if count < min_measured_trials:
                failures.append(
                    f"block {block} workload {workload} has {count} measured trials; "
                    f"expected at least {min_measured_trials}"
                )
    return failures


def exactness_failures(records: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for record in records:
        key = record_key(record)
        if record.get("status") != "passed":
            failures.append(f"{key}: status={record.get('status')}")
        if not record.get("comparison", {}).get("byte_identical"):
            failures.append(f"{key}: comparison.byte_identical is false")
        if record.get("blocker"):
            failures.append(f"{key}: blocker={record.get('blocker')}")
    return failures


def provenance_failures(records: list[dict[str, Any]], require_gemma4d_env: bool) -> list[str]:
    failures: list[str] = []
    for record in records:
        provenance = record.get("build_provenance") or {}
        key = record_key(record)
        if not provenance.get("git_sha"):
            failures.append(f"{key}: missing build_provenance.git_sha")
        if not provenance.get("dirty_diff_sha256"):
            failures.append(f"{key}: missing build_provenance.dirty_diff_sha256")
        if not provenance.get("runner_binary_path"):
            failures.append(f"{key}: missing build_provenance.runner_binary_path")
        if provenance.get("runner_binary_link_mtime_unix_seconds") in (None, 0):
            failures.append(f"{key}: missing build_provenance.runner_binary_link_mtime_unix_seconds")
        if require_gemma4d_env and not provenance.get("gemma4d_env"):
            failures.append(f"{key}: missing build_provenance.gemma4d_env")
    return failures


def memory_failures(records: list[dict[str, Any]], memory_cliff_gb: float) -> list[str]:
    failures: list[str] = []
    for record in records:
        key = record_key(record)
        mtp_peak = required_float(record, "mtp", "peak_memory_gb", failures, key)
        baseline_peak = required_float(record, "baseline", "peak_memory_gb", failures, key)
        if mtp_peak is not None and mtp_peak > memory_cliff_gb:
            failures.append(f"{key}: MTP peak {mtp_peak:.3f} GB > cliff {memory_cliff_gb:.3f} GB")
        if baseline_peak is not None and baseline_peak > memory_cliff_gb:
            failures.append(
                f"{key}: baseline peak {baseline_peak:.3f} GB > cliff {memory_cliff_gb:.3f} GB"
            )
    return failures


def trace_failures(records: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for record in records:
        workload = record.get("workload_id")
        trial = record.get("trial_index")
        block_size = record.get("block_size")
        for event in record.get("mtp", {}).get("events", []):
            draft_len = len(event.get("draft_tokens") or [])
            accepted = int(event.get("accepted_draft_count") or 0)
            if event.get("terminal_no_lookahead"):
                expected = draft_len
            elif accepted >= draft_len:
                expected = draft_len + 1
            else:
                expected = accepted + 2
            actual = int(event.get("trace_position_count") or 0)
            if actual < expected:
                failures.append(
                    f"{workload} trial={trial} block={block_size} pass={event.get('pass_index')}: "
                    f"trace positions {actual} < expected {expected}"
                )
    return failures


def full_block_event_failures(records: list[dict[str, Any]], block_size: int) -> list[str]:
    full_events = 0
    for record in records:
        if not record.get("measured") or int(record.get("block_size")) != block_size:
            continue
        for event in record.get("mtp", {}).get("events", []):
            if len(event.get("draft_tokens") or []) == block_size:
                full_events += 1
    if full_events == 0:
        return [f"no measured N={block_size} event drafted {block_size} tokens"]
    return []


def summary_failures(summary: dict[str, Any], expected_blocks: list[int]) -> list[str]:
    failures: list[str] = []
    if summary.get("blockers"):
        failures.append(f"summary.blockers is non-empty: {summary.get('blockers')}")
    summary_blocks = [int(block) for block in summary.get("block_sizes", [])]
    missing = [block for block in expected_blocks if block not in summary_blocks]
    if missing:
        failures.append(f"summary.block_sizes missing: {','.join(str(block) for block in missing)}")
    if not policy_summary(summary, GUARDED_POLICY):
        failures.append(f"summary.policy_summaries missing {GUARDED_POLICY}")
    return failures


def policy_summary(summary: dict[str, Any], policy_name: str) -> dict[str, Any] | None:
    for policy in summary.get("policy_summaries", []):
        if policy.get("policy_name") == policy_name:
            return policy
    return None


def aggregate_blocks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("measured"):
            by_block[int(record["block_size"])].append(record)

    rows: list[dict[str, Any]] = []
    for block_size in sorted(by_block):
        block_records = by_block[block_size]
        baseline_decode_ms = sum(float(record["baseline"]["decode_ms"]) for record in block_records)
        mtp_decode_ms = sum(float(record["mtp"]["decode_phase_ms"]) for record in block_records)
        draft_ms = sum(float(record["mtp"]["draft_ms"]) for record in block_records)
        verify_ms = sum(float(record["mtp"]["verify_ms"]) for record in block_records)
        verify_forward_ms = sum(float(record["mtp"]["verify_forward_ms"]) for record in block_records)
        verify_repair_ms = sum(float(record["mtp"]["verify_repair_ms"]) for record in block_records)
        repair_subtimer_presence = [
            all(key in record["mtp"] for key in REPAIR_SUBTIMER_KEYS)
            for record in block_records
        ]
        repair_subtimers_captured = all(repair_subtimer_presence)
        if any(repair_subtimer_presence) and not repair_subtimers_captured:
            raise SystemExit(
                f"block {block_size}: repair sub-timer fields are present on only some measured records"
            )
        repair_clone_ms = (
            sum(float(record["mtp"]["repair_clone_ms"]) for record in block_records)
            if repair_subtimers_captured
            else None
        )
        repair_forward_ms = (
            sum(float(record["mtp"]["repair_forward_ms"]) for record in block_records)
            if repair_subtimers_captured
            else None
        )
        repair_fallback_ms = (
            sum(float(record["mtp"]["repair_fallback_ms"]) for record in block_records)
            if repair_subtimers_captured
            else None
        )
        attempted = sum(int(record["mtp"]["attempted_draft_tokens"]) for record in block_records)
        accepted = sum(int(record["mtp"]["accepted_draft_tokens"]) for record in block_records)
        verify_passes = sum(int(record["mtp"]["target_verify_passes"]) for record in block_records)
        verifier_records = [
            record for record in block_records if not record["mtp"].get("auto_disabled")
        ]
        verifier_generated_tokens = sum(
            len(record["mtp"]["generated_tokens"]) for record in verifier_records
        )
        verifier_passes = sum(
            int(record["mtp"]["target_verify_passes"]) for record in verifier_records
        )
        peak_mlx_gb = max(float(record["mtp"]["peak_memory_gb"]) for record in block_records)
        baseline_peak_gb = max(float(record["baseline"]["peak_memory_gb"]) for record in block_records)
        exact_records = sum(
            1
            for record in block_records
            if record.get("status") == "passed" and record.get("comparison", {}).get("byte_identical")
        )
        auto_disabled = sum(1 for record in block_records if record["mtp"].get("auto_disabled"))

        slot_attempts: dict[int, int] = defaultdict(int)
        slot_accepts: dict[int, int] = defaultdict(int)
        full_len_events = 0
        for record in block_records:
            for event in record["mtp"].get("events", []):
                draft_len = len(event.get("draft_tokens") or [])
                if draft_len == block_size:
                    full_len_events += 1
                event_accepted = min(int(event.get("accepted_draft_count") or 0), draft_len)
                for slot in range(draft_len):
                    slot_attempts[slot] += 1
                    if slot < event_accepted:
                        slot_accepts[slot] += 1

        verify_ms_per_pass = safe_ratio(verify_ms, verify_passes)
        draft_ms_per_attempt = safe_ratio(draft_ms, attempted)
        rows.append(
            {
                "block_size": block_size,
                "records": len(block_records),
                "exact_records": exact_records,
                "auto_disabled_records": auto_disabled,
                "baseline_decode_ms": baseline_decode_ms,
                "mtp_decode_ms": mtp_decode_ms,
                "net_speedup_pct": percent(baseline_decode_ms - mtp_decode_ms, baseline_decode_ms),
                "acceptance_rate": safe_ratio(accepted, attempted),
                "accepted": accepted,
                "attempted": attempted,
                "tokens_per_verify_pass": safe_ratio(verifier_generated_tokens, verifier_passes),
                "accepted_tokens_per_verify_pass": safe_ratio(accepted, verify_passes),
                "verify_passes": verify_passes,
                "tokens_per_verify_records": len(verifier_records),
                "tokens_per_verify_generated_tokens": verifier_generated_tokens,
                "tokens_per_verify_passes": verifier_passes,
                "draft_ms": draft_ms,
                "verify_ms": verify_ms,
                "verify_forward_ms": verify_forward_ms,
                "verify_repair_ms": verify_repair_ms,
                "repair_clone_ms": repair_clone_ms,
                "repair_forward_ms": repair_forward_ms,
                "repair_fallback_ms": repair_fallback_ms,
                "repair_subtimers_captured": repair_subtimers_captured,
                "draft_step_verify_units": safe_ratio(draft_ms_per_attempt, verify_ms_per_pass),
                "peak_mlx_gb": peak_mlx_gb,
                "baseline_peak_gb": baseline_peak_gb,
                "full_len_events": full_len_events,
                "per_slot": [
                    {
                        "slot": slot + 1,
                        "accepted": slot_accepts[slot],
                        "attempted": slot_attempts[slot],
                        "rate": safe_ratio(slot_accepts[slot], slot_attempts[slot]),
                    }
                    for slot in sorted(slot_attempts)
                ],
            }
        )
    return rows


def compare_sequential(
    candidate_records: list[dict[str, Any]],
    sequential_records: list[dict[str, Any]],
) -> dict[str, Any]:
    sequential_by_key = {record_key(record): record for record in sequential_records}
    missing: list[str] = []
    mismatches: list[str] = []
    compared = 0
    for candidate in candidate_records:
        key = record_key(candidate)
        sequential = sequential_by_key.get(key)
        if sequential is None:
            missing.append(str(key))
            continue
        compared += 1
        if candidate.get("mtp", {}).get("generated_tokens") != sequential.get("mtp", {}).get("generated_tokens"):
            mismatches.append(str(key))
    return {
        "compared_records": compared,
        "missing_records": missing,
        "mismatches": mismatches,
        "passed": not missing and not mismatches,
    }


def render_markdown(
    candidate_path: Path,
    candidate_summary_path: Path,
    candidate_summary: dict[str, Any],
    candidate_records: list[dict[str, Any]],
    block_rows: list[dict[str, Any]],
    block_issues: list[str],
    workload_trial_issues: list[str],
    exactness_issues: list[str],
    provenance_issues: list[str],
    memory_issues: list[str],
    trace_issues: list[str],
    full_block_issues: list[str],
    summary_issues: list[str],
    memory_cliff_gb: float,
    sequential_path: Path | None,
    sequential_diff: dict[str, Any] | None,
) -> str:
    first = candidate_records[0]
    guarded = policy_summary(candidate_summary, GUARDED_POLICY)
    gemma4d_env = (first.get("build_provenance") or {}).get("gemma4d_env")
    if gemma4d_env:
        env_line = ", ".join(f"{key}={value}" for key, value in sorted(gemma4d_env.items()))
    else:
        env_line = "not captured; env stamping postdates these XR55 legs"
    lines = [
        "# N-Block MTP Evidence",
        "",
        f"- Candidate records: `{candidate_path}`",
        f"- Candidate summary: `{candidate_summary_path}`",
        f"- Run ID: `{first.get('run_id')}`",
        f"- Git SHA: `{first.get('git_sha')}`",
        f"- Dirty diff SHA-256: `{first.get('build_provenance', {}).get('dirty_diff_sha256')}`",
        f"- Runner: `{first.get('build_provenance', {}).get('runner_binary_path')}`",
        f"- Runner link mtime: `{first.get('build_provenance', {}).get('runner_binary_link_mtime_unix_seconds')}`",
        f"- GEMMA4D env: `{env_line}`",
        f"- Tiny16 memory cliff: `{fmt_float(memory_cliff_gb, 3)} GB`",
        "",
        "## Block Sweep",
        "",
        "| N | measured | exact | speedup % | acceptance | tokens/verify | accepted/verify | draft ms | verify ms | verify forward ms | verify repair ms | repair clone ms | repair forward ms | repair fallback ms | draft-step verify units | peak MLX GB | full-N events |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in block_rows:
        lines.append(
            "| {block_size} | {records} | {exact_records} | {speedup} | {acceptance} | "
            "{tokens_per_verify} | {accepted_per_verify} | {draft_ms} | {verify_ms} | "
            "{verify_forward_ms} | {verify_repair_ms} | {repair_clone_ms} | {repair_forward_ms} | "
            "{repair_fallback_ms} | {draft_units} | {peak_mlx} | {full_len_events} |".format(
                block_size=row["block_size"],
                records=row["records"],
                exact_records=row["exact_records"],
                speedup=fmt_float(row["net_speedup_pct"], 2),
                acceptance=fmt_float(row["acceptance_rate"], 3),
                tokens_per_verify=fmt_float(row["tokens_per_verify_pass"], 3),
                accepted_per_verify=fmt_float(row["accepted_tokens_per_verify_pass"], 3),
                draft_ms=fmt_float(row["draft_ms"], 1),
                verify_ms=fmt_float(row["verify_ms"], 1),
                verify_forward_ms=fmt_float(row["verify_forward_ms"], 1),
                verify_repair_ms=fmt_float(row["verify_repair_ms"], 1),
                repair_clone_ms=fmt_repair_subtimer(row, "repair_clone_ms"),
                repair_forward_ms=fmt_repair_subtimer(row, "repair_forward_ms"),
                repair_fallback_ms=fmt_repair_subtimer(row, "repair_fallback_ms"),
                draft_units=fmt_float(row["draft_step_verify_units"], 3),
                peak_mlx=fmt_float(row["peak_mlx_gb"], 3),
                full_len_events=row["full_len_events"],
            )
        )

    lines.extend(["", "## Per-Slot Acceptance", ""])
    for row in block_rows:
        slot_cells = [
            f"s{slot['slot']}={slot['accepted']}/{slot['attempted']} ({fmt_float(slot['rate'], 3)})"
            for slot in row["per_slot"]
        ]
        lines.append(f"- N={row['block_size']}: " + ", ".join(slot_cells))

    lines.extend(["", "## Guarded Policy", ""])
    if guarded:
        lines.extend(
            [
                f"- Policy: `{guarded.get('policy_name')}`",
                f"- Decision: `{guarded.get('decision')}`",
                f"- Selected workloads: `{', '.join(guarded.get('selected_workloads') or [])}`",
                f"- Aggregate speedup: `{fmt_float(float(guarded.get('aggregate_speedup_percent') or 0.0), 3)}%`",
                f"- Weighted acceptance: `{fmt_float(float(guarded.get('weighted_acceptance_rate') or 0.0), 3)}`",
                f"- Peak MLX: `{fmt_float(float(guarded.get('max_peak_memory_gb') or 0.0), 3)} GB`",
            ]
        )
    else:
        lines.append(f"- Missing `{GUARDED_POLICY}` policy summary.")

    lines.extend(["", "## Gates", ""])
    lines.append(f"- Block coverage: {'PASS' if not block_issues else 'FAIL'}")
    for issue in block_issues[:10]:
        lines.append(f"  - {issue}")
    lines.append(f"- Workload/trial coverage: {'PASS' if not workload_trial_issues else 'FAIL'}")
    for issue in workload_trial_issues[:10]:
        lines.append(f"  - {issue}")
    lines.append(f"- Greedy exactness: {'PASS' if not exactness_issues else 'FAIL'}")
    for issue in exactness_issues[:10]:
        lines.append(f"  - {issue}")
    if len(exactness_issues) > 10:
        lines.append(f"  - ... {len(exactness_issues) - 10} more")
    lines.append(f"- Provenance: {'PASS' if not provenance_issues else 'FAIL'}")
    for issue in provenance_issues[:10]:
        lines.append(f"  - {issue}")
    if len(provenance_issues) > 10:
        lines.append(f"  - ... {len(provenance_issues) - 10} more")

    lines.append(f"- Tiny16 memory: {'PASS' if not memory_issues else 'FAIL'}")
    for issue in memory_issues[:10]:
        lines.append(f"  - {issue}")
    if len(memory_issues) > 10:
        lines.append(f"  - ... {len(memory_issues) - 10} more")

    lines.append(f"- Trace completeness: {'PASS' if not trace_issues else 'FAIL'}")
    for issue in trace_issues[:10]:
        lines.append(f"  - {issue}")
    if len(trace_issues) > 10:
        lines.append(f"  - ... {len(trace_issues) - 10} more")

    lines.append(f"- Full N=8 trace exercise: {'PASS' if not full_block_issues else 'FAIL'}")
    for issue in full_block_issues[:10]:
        lines.append(f"  - {issue}")

    lines.append(f"- Summary policy/blockers: {'PASS' if not summary_issues else 'FAIL'}")
    for issue in summary_issues[:10]:
        lines.append(f"  - {issue}")

    if sequential_diff is not None:
        lines.append(
            f"- Sequential differential: {'PASS' if sequential_diff['passed'] else 'FAIL'} "
            f"({sequential_diff['compared_records']} records vs `{sequential_path}`)"
        )
        for issue in sequential_diff["missing_records"][:10]:
            lines.append(f"  - missing {issue}")
        for issue in sequential_diff["mismatches"][:10]:
            lines.append(f"  - mismatch {issue}")

    high_draft_cost = [
        row for row in block_rows if row["draft_step_verify_units"] > 0.1
    ]
    if high_draft_cost:
        lines.extend(["", "## Draft-Cost Flag", ""])
        for row in high_draft_cost:
            lines.append(
                f"- N={row['block_size']}: draft cost/step is "
                f"{fmt_float(row['draft_step_verify_units'], 3)} verify-units (>0.1)."
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-records", required=True, type=Path)
    parser.add_argument("--candidate-summary", type=Path)
    parser.add_argument("--sequential-records", type=Path)
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument("--expected-block-sizes", default=",".join(str(item) for item in DEFAULT_EXPECTED_BLOCKS))
    parser.add_argument("--expected-workload-ids", default=",".join(DEFAULT_EXPECTED_WORKLOADS))
    parser.add_argument("--min-measured-trials", type=int, default=3)
    parser.add_argument("--memory-cliff-gb", type=float)
    parser.add_argument("--require-gemma4d-env", action="store_true")
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    expected_blocks = parse_int_csv(args.expected_block_sizes)
    expected_workloads = parse_str_csv(args.expected_workload_ids)
    if args.sequential_records is None and not args.candidate_only:
        raise SystemExit("--sequential-records is required unless --candidate-only is set")

    candidate_records = load_jsonl(args.candidate_records)
    candidate_summary_path = args.candidate_summary or args.candidate_records.parent / "summary.json"
    candidate_summary = load_json(candidate_summary_path)
    sequential_records = load_jsonl(args.sequential_records) if args.sequential_records else None
    memory_cliff_gb = (
        args.memory_cliff_gb
        if args.memory_cliff_gb is not None
        else float(candidate_summary.get("memory_cliff_gb") or 14.0)
    )
    block_rows = aggregate_blocks(candidate_records)
    block_issues = block_coverage_failures(candidate_records, expected_blocks)
    workload_trial_issues = (
        []
        if args.candidate_only
        else workload_trial_failures(
            candidate_records,
            expected_blocks,
            expected_workloads,
            args.min_measured_trials,
        )
    )
    exactness_issues = exactness_failures(candidate_records)
    provenance_issues = provenance_failures(candidate_records, args.require_gemma4d_env)
    memory_issues = memory_failures(candidate_records, memory_cliff_gb)
    trace_issues = trace_failures(candidate_records)
    full_block_issues = full_block_event_failures(candidate_records, max(expected_blocks))
    summary_issues = summary_failures(candidate_summary, expected_blocks)
    sequential_diff = (
        compare_sequential(candidate_records, sequential_records)
        if sequential_records is not None
        else None
    )

    payload = {
        "candidate_records": str(args.candidate_records),
        "candidate_summary": str(candidate_summary_path),
        "sequential_records": str(args.sequential_records) if args.sequential_records else None,
        "expected_blocks": expected_blocks,
        "expected_workloads": expected_workloads,
        "min_measured_trials": args.min_measured_trials,
        "memory_cliff_gb": memory_cliff_gb,
        "require_gemma4d_env": args.require_gemma4d_env,
        "blocks": block_rows,
        "block_issues": block_issues,
        "workload_trial_issues": workload_trial_issues,
        "exactness_issues": exactness_issues,
        "provenance_issues": provenance_issues,
        "memory_issues": memory_issues,
        "trace_issues": trace_issues,
        "full_block_issues": full_block_issues,
        "summary_issues": summary_issues,
        "sequential_diff": sequential_diff,
        "guarded_policy": policy_summary(candidate_summary, GUARDED_POLICY),
    }

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(
        render_markdown(
            args.candidate_records,
            candidate_summary_path,
            candidate_summary,
            candidate_records,
            block_rows,
            block_issues,
            workload_trial_issues,
            exactness_issues,
            provenance_issues,
            memory_issues,
            trace_issues,
            full_block_issues,
            summary_issues,
            memory_cliff_gb,
            args.sequential_records,
            sequential_diff,
        ),
        encoding="utf-8",
    )
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    gate_issues = (
        block_issues
        + workload_trial_issues
        + exactness_issues
        + provenance_issues
        + memory_issues
        + trace_issues
        + full_block_issues
        + summary_issues
    )
    if gate_issues or (sequential_diff is not None and not sequential_diff["passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
