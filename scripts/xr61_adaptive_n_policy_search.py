#!/usr/bin/env python3
"""Offline Adaptive-N MTP policy search for XR61 records.

The script is intentionally read-only over benchmark artifacts. It can analyze
XR56-style fixed-N records immediately and enrich the policy feature report
when XR57 real-margin/top-k records are provided.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_BLOCKS = (1, 2, 3, 4, 6, 8)
DEFAULT_CANDIDATE_RECORDS = (
    "benchmarks/out/XR56-repair-cost/candidate-retro-prefix/records.jsonl"
)
DEFAULT_OUT_DIR = "benchmarks/out/XR61-adaptive-n-mtp/policy-search"
DEFAULT_MIN_SPEEDUP_PERCENT = 5.0
DEFAULT_MEMORY_CLIFF_GB = 14.0
DEFAULT_EXPECTED_REAL_MARGIN_WORKLOADS = (
    "chat_short_1k_001",
    "tool_json_1k_001",
    "mtp_candidate_1k_001",
)
DEFAULT_MIN_REAL_MARGIN_MEASURED_RECORDS = 54


def load_jsonl(path: Path, *, required: bool) -> list[dict[str, Any]]:
    if not path.exists():
        if required:
            raise SystemExit(f"{path}: required JSONL file does not exist")
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    if required and not records:
        raise SystemExit(f"{path}: no records found")
    return records


def parse_csv_ints(raw: str) -> list[int]:
    out: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            out.append(int(item))
        except ValueError as exc:
            raise SystemExit(f"invalid integer in CSV {raw!r}: {item!r}") from exc
    if not out:
        raise SystemExit("expected at least one integer")
    return sorted(set(out))


def parse_csv_strings(raw: str) -> list[str]:
    values = sorted({item.strip() for item in raw.split(",") if item.strip()})
    if not values:
        raise SystemExit("expected at least one string")
    return values


def median(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if finite(value)]
    if not clean:
        return 0.0
    return float(statistics.median(clean))


def mean(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if finite(value)]
    if not clean:
        return 0.0
    return float(sum(clean) / len(clean))


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def speedup_percent(baseline_ms: float, candidate_ms: float) -> float:
    return ((baseline_ms - candidate_ms) / baseline_ms * 100.0) if baseline_ms > 0 else 0.0


def record_key(record: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(record.get("workload_id")),
        str(record.get("trial_kind")),
        int(record.get("trial_index") or 0),
        int(record.get("block_size") or 0),
    )


def measured_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record.get("measured")]


def records_by_workload(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("workload_id"))].append(record)
    return dict(grouped)


def baseline_decode_ms(records: list[dict[str, Any]]) -> float:
    return median((record.get("baseline") or {}).get("decode_ms") for record in records)


def block_records(records: list[dict[str, Any]], block_size: int) -> list[dict[str, Any]]:
    return [record for record in records if int(record.get("block_size") or 0) == block_size]


def block_summary(records: list[dict[str, Any]], block_size: int) -> dict[str, Any] | None:
    rows = block_records(records, block_size)
    if not rows:
        return None
    attempted = sum(int((row.get("mtp") or {}).get("attempted_draft_tokens") or 0) for row in rows)
    accepted = sum(int((row.get("mtp") or {}).get("accepted_draft_tokens") or 0) for row in rows)
    verify_passes = sum(int((row.get("mtp") or {}).get("target_verify_passes") or 0) for row in rows)
    baseline_ms = baseline_decode_ms(rows)
    candidate_ms = median((row.get("mtp") or {}).get("decode_phase_ms") for row in rows)
    return {
        "block_size": block_size,
        "records": len(rows),
        "exact": all((row.get("comparison") or {}).get("byte_identical") for row in rows),
        "baseline_decode_ms": baseline_ms,
        "candidate_decode_phase_ms": candidate_ms,
        "speedup_percent": speedup_percent(baseline_ms, candidate_ms),
        "accepted_draft_tokens": accepted,
        "attempted_draft_tokens": attempted,
        "acceptance_rate": ratio(accepted, attempted),
        "target_verify_passes": verify_passes,
        "accepted_tokens_per_verify": ratio(accepted, verify_passes),
        "peak_memory_gb": max(
            [float((row.get("mtp") or {}).get("peak_memory_gb") or 0.0) for row in rows]
            or [0.0]
        ),
        "draft_ms_median": median((row.get("mtp") or {}).get("draft_ms") for row in rows),
        "verify_ms_median": median((row.get("mtp") or {}).get("verify_ms") for row in rows),
        "verify_forward_ms_median": median(
            (row.get("mtp") or {}).get("verify_forward_ms") for row in rows
        ),
        "verify_repair_ms_median": median(
            (row.get("mtp") or {}).get("verify_repair_ms") for row in rows
        ),
        "repair_fallback_ms_median": median(
            (row.get("mtp") or {}).get("repair_fallback_ms") for row in rows
        ),
    }


def aggregate_block_summary(
    records: list[dict[str, Any]], block_size: int
) -> dict[str, Any] | None:
    rows = block_records(records, block_size)
    if not rows:
        return None
    baseline_ms = sum(float((row.get("baseline") or {}).get("decode_ms") or 0.0) for row in rows)
    candidate_ms = sum(
        float((row.get("mtp") or {}).get("decode_phase_ms") or 0.0) for row in rows
    )
    attempted = sum(int((row.get("mtp") or {}).get("attempted_draft_tokens") or 0) for row in rows)
    accepted = sum(int((row.get("mtp") or {}).get("accepted_draft_tokens") or 0) for row in rows)
    verify_passes = sum(int((row.get("mtp") or {}).get("target_verify_passes") or 0) for row in rows)
    peak = max([float((row.get("mtp") or {}).get("peak_memory_gb") or 0.0) for row in rows] or [0.0])
    return {
        "aggregation": "sum_of_measured_records",
        "block_size": block_size,
        "records": len(rows),
        "exact": all((row.get("comparison") or {}).get("byte_identical") for row in rows),
        "baseline_decode_ms": baseline_ms,
        "candidate_decode_phase_ms": candidate_ms,
        "speedup_percent": speedup_percent(baseline_ms, candidate_ms),
        "peak_memory_gb": peak,
        "accepted_draft_tokens": accepted,
        "attempted_draft_tokens": attempted,
        "acceptance_rate": ratio(accepted, attempted),
        "target_verify_passes": verify_passes,
        "accepted_tokens_per_verify": ratio(accepted, verify_passes),
        "draft_ms": sum(float((row.get("mtp") or {}).get("draft_ms") or 0.0) for row in rows),
        "verify_ms": sum(float((row.get("mtp") or {}).get("verify_ms") or 0.0) for row in rows),
        "verify_forward_ms": sum(
            float((row.get("mtp") or {}).get("verify_forward_ms") or 0.0) for row in rows
        ),
        "verify_repair_ms": sum(
            float((row.get("mtp") or {}).get("verify_repair_ms") or 0.0) for row in rows
        ),
        "repair_clone_ms": sum(
            float((row.get("mtp") or {}).get("repair_clone_ms") or 0.0) for row in rows
        ),
        "repair_forward_ms": sum(
            float((row.get("mtp") or {}).get("repair_forward_ms") or 0.0) for row in rows
        ),
        "repair_fallback_ms": sum(
            float((row.get("mtp") or {}).get("repair_fallback_ms") or 0.0) for row in rows
        ),
    }


def fixed_n_table(records: list[dict[str, Any]], blocks: list[int]) -> dict[str, Any]:
    measured = measured_records(records)
    by_workload = records_by_workload(measured)
    workloads: dict[str, Any] = {}
    for workload_id, rows in by_workload.items():
        workloads[workload_id] = {
            "baseline_decode_ms": baseline_decode_ms(rows),
            "blocks": {
                str(block): summary
                for block in blocks
                if (summary := block_summary(rows, block)) is not None
            },
        }
    aggregate = {
        str(block): summary
        for block in blocks
        if (summary := aggregate_block_summary(measured, block)) is not None
    }
    return {"aggregate": aggregate, "workloads": workloads}


def guarded_policy(
    records: list[dict[str, Any]],
    blocks: list[int],
    min_speedup_percent: float,
    memory_cliff_gb: float,
) -> dict[str, Any]:
    measured = measured_records(records)
    by_workload = records_by_workload(measured)
    selected: dict[str, Any] = {}
    total_baseline = 0.0
    total_candidate = 0.0
    accepted = 0
    attempted = 0
    peak = 0.0
    regressions: list[str] = []

    for workload_id, rows in by_workload.items():
        baseline = baseline_decode_ms(rows)
        total_baseline += baseline
        viable: list[dict[str, Any]] = []
        for block in blocks:
            summary = block_summary(rows, block)
            if summary is None or not summary["exact"]:
                continue
            if float(summary["peak_memory_gb"]) > memory_cliff_gb:
                continue
            if float(summary["speedup_percent"]) >= min_speedup_percent:
                viable.append(summary)
        if not viable:
            selected[workload_id] = {
                "mtp_enabled": False,
                "reason": "no exact block met speed and memory gates",
                "baseline_decode_ms": baseline,
                "selected_decode_phase_ms": baseline,
            }
            total_candidate += baseline
            continue
        winner = min(viable, key=lambda row: float(row["candidate_decode_phase_ms"]))
        selected[workload_id] = {
            "mtp_enabled": True,
            "block_size": winner["block_size"],
            "reason": "fastest exact block meeting speed and memory gates",
            "baseline_decode_ms": baseline,
            "selected_decode_phase_ms": winner["candidate_decode_phase_ms"],
            "speedup_percent": winner["speedup_percent"],
            "accepted_draft_tokens": winner["accepted_draft_tokens"],
            "attempted_draft_tokens": winner["attempted_draft_tokens"],
            "acceptance_rate": winner["acceptance_rate"],
            "peak_memory_gb": winner["peak_memory_gb"],
        }
        total_candidate += float(winner["candidate_decode_phase_ms"])
        accepted += int(winner["accepted_draft_tokens"])
        attempted += int(winner["attempted_draft_tokens"])
        peak = max(peak, float(winner["peak_memory_gb"]))
        if float(winner["candidate_decode_phase_ms"]) > baseline * 1.05:
            regressions.append(workload_id)

    return {
        "policy_name": "net_latency_guarded_5pct_recomputed",
        "selected_workloads": selected,
        "total_baseline_decode_ms": total_baseline,
        "total_selected_decode_phase_ms": total_candidate,
        "aggregate_speedup_percent": speedup_percent(total_baseline, total_candidate),
        "weighted_acceptance_rate": ratio(accepted, attempted),
        "accepted_draft_tokens": accepted,
        "attempted_draft_tokens": attempted,
        "max_peak_memory_gb": peak,
        "regressed_workloads": regressions,
    }


def event_rows(records: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if not record.get("measured"):
            continue
        previous: dict[str, Any] | None = None
        events = (record.get("mtp") or {}).get("events") or []
        for event_index, event in enumerate(events):
            draft_tokens = event.get("draft_tokens") or []
            accepted_count = int(event.get("accepted_draft_count") or 0)
            margins = [float(value) for value in event.get("logit_margins") or [] if finite(value)]
            in_top_k = [bool(value) for value in event.get("draft_in_target_top_k") or []]
            target_top_token_ids = event.get("target_top_token_ids") or []
            target_ranks: list[int | None] = []
            for position, token in enumerate(draft_tokens):
                row = target_top_token_ids[position] if position < len(target_top_token_ids) else []
                try:
                    target_ranks.append([int(value) for value in row].index(int(token)) + 1)
                except ValueError:
                    target_ranks.append(None)
            draft_len = len(draft_tokens)
            current = {
                "source": source,
                "workload_id": record.get("workload_id"),
                "family": record.get("family"),
                "block_size": int(record.get("block_size") or 0),
                "trial_index": int(record.get("trial_index") or 0),
                "event_index": event_index,
                "pass_index": int(event.get("pass_index") or 0),
                "context_sequence_len": int(event.get("context_sequence_len") or 0),
                "remaining_token_budget": int(event.get("remaining_token_budget") or 0),
                "draft_count": draft_len,
                "accepted_draft_count": accepted_count,
                "zero_accept": accepted_count == 0,
                "full_accept": draft_len > 0 and accepted_count >= draft_len,
                "partial_reject": 0 < accepted_count < draft_len,
                "rejected": bool(event.get("rejected")),
                "verify_ms": float(event.get("verify_ms") or 0.0),
                "verify_forward_ms": float(event.get("verify_forward_ms") or 0.0),
                "verify_repair_ms": float(event.get("verify_repair_ms") or 0.0),
                "repair_fallback_ms": float(event.get("repair_fallback_ms") or 0.0),
                "trace_top_k": int(event.get("trace_top_k") or 0),
                "margin_min": min(margins) if margins else None,
                "margin_mean": mean(margins),
                "margin_max": max(margins) if margins else None,
                "draft_in_top_k_count": sum(1 for value in in_top_k[:draft_len] if value),
                "draft_in_top_k_rate": ratio(sum(1 for value in in_top_k[:draft_len] if value), draft_len),
                "target_rank_known_count": sum(1 for value in target_ranks if value is not None),
                "target_rank_min": min([value for value in target_ranks if value is not None], default=None),
                "previous_accepted_draft_count": previous.get("accepted_draft_count") if previous else None,
                "previous_zero_accept": previous.get("zero_accept") if previous else None,
                "previous_full_accept": previous.get("full_accept") if previous else None,
                "previous_margin_mean": previous.get("margin_mean") if previous else None,
                "previous_draft_in_top_k_rate": previous.get("draft_in_top_k_rate") if previous else None,
            }
            rows.append(current)
            previous = current
    return rows


def distribution(values: Iterable[int]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(Counter(values).items())}


def event_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "event_count": 0,
            "note": "no measured MTP events available",
        }
    slot_attempts: Counter[int] = Counter()
    slot_accepts: Counter[int] = Counter()
    zero_streaks: list[int] = []
    partial_rejects: Counter[int] = Counter()
    full_block_accepts: Counter[int] = Counter()
    current_zero_streak: dict[tuple[str, int, int], int] = defaultdict(int)
    previous_buckets: dict[str, list[int]] = defaultdict(list)

    for row in rows:
        key = (str(row["workload_id"]), int(row["block_size"]), int(row["trial_index"]))
        draft_count = int(row["draft_count"])
        accepted = int(row["accepted_draft_count"])
        for slot in range(1, draft_count + 1):
            slot_attempts[slot] += 1
            if accepted >= slot:
                slot_accepts[slot] += 1
        if row["zero_accept"]:
            current_zero_streak[key] += 1
        else:
            if current_zero_streak[key]:
                zero_streaks.append(current_zero_streak[key])
            current_zero_streak[key] = 0
        if row["partial_reject"]:
            partial_rejects[accepted] += 1
        if row["full_accept"]:
            full_block_accepts[draft_count] += 1
        if row["previous_zero_accept"] is not None:
            if row["previous_zero_accept"]:
                previous_buckets["after_previous_zero_accept"].append(accepted)
            if row["previous_full_accept"]:
                previous_buckets["after_previous_full_accept"].append(accepted)
            if finite(row["previous_draft_in_top_k_rate"]) and float(row["previous_draft_in_top_k_rate"]) >= 0.75:
                previous_buckets["after_previous_top_k_rate_ge_0_75"].append(accepted)
            if finite(row["previous_margin_mean"]) and float(row["previous_margin_mean"]) >= 0.5:
                previous_buckets["after_previous_margin_mean_ge_0_5"].append(accepted)
    zero_streaks.extend(value for value in current_zero_streak.values() if value)

    return {
        "event_count": len(rows),
        "slot_acceptance": {
            str(slot): {
                "accepted": slot_accepts[slot],
                "attempted": slot_attempts[slot],
                "rate": ratio(slot_accepts[slot], slot_attempts[slot]),
            }
            for slot in sorted(slot_attempts)
        },
        "zero_accept_streak_distribution": distribution(zero_streaks),
        "partial_reject_accepted_count_distribution": {
            str(key): value for key, value in sorted(partial_rejects.items())
        },
        "full_block_accept_distribution": {
            str(key): value for key, value in sorted(full_block_accepts.items())
        },
        "previous_pass_correlations": {
            bucket: {
                "events": len(values),
                "next_accept_mean": mean(values),
                "next_accept_median": median(values),
            }
            for bucket, values in sorted(previous_buckets.items())
        },
    }


def real_margin_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    real_rows = [row for row in rows if int(row.get("trace_top_k") or 0) > 1]
    if not real_rows:
        return {
            "available": False,
            "reason": "no measured event rows advertised trace_top_k > 1",
        }
    accepted_margins: list[float] = []
    rejected_margins: list[float] = []
    accepted_top_k_rates: list[float] = []
    rejected_top_k_rates: list[float] = []
    for row in real_rows:
        target = accepted_margins if int(row["accepted_draft_count"]) > 0 else rejected_margins
        if finite(row.get("margin_mean")):
            target.append(float(row["margin_mean"]))
        rate_target = accepted_top_k_rates if int(row["accepted_draft_count"]) > 0 else rejected_top_k_rates
        if finite(row.get("draft_in_top_k_rate")):
            rate_target.append(float(row["draft_in_top_k_rate"]))
    return {
        "available": True,
        "event_count": len(real_rows),
        "accepted_margin_mean": mean(accepted_margins),
        "accepted_margin_median": median(accepted_margins),
        "rejected_margin_mean": mean(rejected_margins),
        "rejected_margin_median": median(rejected_margins),
        "accepted_draft_in_top_k_rate_mean": mean(accepted_top_k_rates),
        "rejected_draft_in_top_k_rate_mean": mean(rejected_top_k_rates),
    }


def real_margin_coverage_summary(
    records: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    expected_workloads: list[str],
    expected_blocks: list[int],
    min_measured_records: int,
) -> dict[str, Any]:
    measured = measured_records(records)
    measured_workloads = sorted({str(record.get("workload_id")) for record in measured})
    measured_blocks = sorted({int(record.get("block_size") or 0) for record in measured})
    real_rows = [row for row in rows if int(row.get("trace_top_k") or 0) > 1]
    missing_workloads = [
        workload for workload in expected_workloads if workload not in measured_workloads
    ]
    missing_blocks = [block for block in expected_blocks if block not in measured_blocks]
    issues: list[str] = []
    if len(measured) < min_measured_records:
        issues.append(
            f"real-margin measured records {len(measured)} < required {min_measured_records}"
        )
    if missing_workloads:
        issues.append(f"missing real-margin workloads: {', '.join(missing_workloads)}")
    if missing_blocks:
        issues.append(
            "missing real-margin block sizes: "
            + ", ".join(str(block) for block in missing_blocks)
        )
    if not real_rows:
        issues.append("no real-margin event rows with trace_top_k > 1")
    return {
        "measured_records": len(measured),
        "real_margin_event_count": len(real_rows),
        "workloads": measured_workloads,
        "blocks": measured_blocks,
        "expected_workloads": expected_workloads,
        "expected_blocks": expected_blocks,
        "min_measured_records": min_measured_records,
        "missing_workloads": missing_workloads,
        "missing_blocks": missing_blocks,
        "sufficient_for_policy_design": not issues,
        "issues": issues,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def render_report(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# XR61 Adaptive-N Policy Search")
    lines.append("")
    lines.append(f"- Candidate records: `{result['inputs']['candidate_records']}`")
    lines.append(f"- Real-margin records: `{result['inputs'].get('real_margin_records') or 'not provided'}`")
    lines.append(f"- Decision hint: `{result['decision_hint']}`")
    lines.append("")
    if result["blockers"]:
        lines.append("## Blockers / Missing Evidence")
        lines.append("")
        for blocker in result["blockers"]:
            lines.append(f"- {blocker}")
        lines.append("")
    lines.append("## Fixed-N Aggregate")
    lines.append("")
    lines.append("| N | Exact | Speedup % | Acceptance | Peak GB |")
    lines.append("|---:|---|---:|---:|---:|")
    for block, row in result["fixed_n"]["aggregate"].items():
        lines.append(
            f"| {block} | `{row['exact']}` | {row['speedup_percent']:.3f} | "
            f"{row['acceptance_rate']:.3f} | {row['peak_memory_gb']:.3f} |"
        )
    lines.append("")
    guarded = result["guarded_policy"]
    lines.append("## Recomputed Guarded Policy")
    lines.append("")
    lines.append(f"- Aggregate speedup: `{guarded['aggregate_speedup_percent']:.3f}%`")
    lines.append(f"- Weighted acceptance: `{guarded['weighted_acceptance_rate']:.3f}`")
    lines.append(f"- Peak memory: `{guarded['max_peak_memory_gb']:.3f} GB`")
    lines.append("")
    lines.append("| Workload | Enabled | N | Speedup % | Reason |")
    lines.append("|---|---|---:|---:|---|")
    for workload, row in guarded["selected_workloads"].items():
        lines.append(
            f"| `{workload}` | `{row['mtp_enabled']}` | "
            f"{row.get('block_size', '')} | {float(row.get('speedup_percent', 0.0)):.3f} | "
            f"{row['reason']} |"
        )
    lines.append("")
    lines.append("## Event Summary")
    lines.append("")
    event_summary_data = result["candidate_event_summary"]
    lines.append(f"- Candidate event count: `{event_summary_data.get('event_count', 0)}`")
    lines.append(
        f"- Zero-accept streaks: `{event_summary_data.get('zero_accept_streak_distribution', {})}`"
    )
    lines.append(
        f"- Full-block accepts: `{event_summary_data.get('full_block_accept_distribution', {})}`"
    )
    lines.append("")
    real_summary = result["real_margin_summary"]
    lines.append("## Real-Margin Signal")
    lines.append("")
    if real_summary.get("available"):
        lines.append(f"- Real-margin event count: `{real_summary['event_count']}`")
        lines.append(f"- Accepted margin median: `{real_summary['accepted_margin_median']:.6f}`")
        lines.append(f"- Rejected margin median: `{real_summary['rejected_margin_median']:.6f}`")
    else:
        lines.append(f"- Unavailable: {real_summary.get('reason')}")
    coverage = result["real_margin_coverage"]
    lines.append(f"- Measured record coverage: `{coverage['measured_records']}`")
    lines.append(f"- Workloads covered: `{coverage['workloads']}`")
    lines.append(f"- Blocks covered: `{coverage['blocks']}`")
    lines.append(
        f"- Sufficient for policy design: `{coverage['sufficient_for_policy_design']}`"
    )
    if coverage["issues"]:
        lines.append("- Coverage issues:")
        for issue in coverage["issues"]:
            lines.append(f"  - {issue}")
    lines.append("")
    lines.append("## Policy Recommendation")
    lines.append("")
    lines.append(result["policy_recommendation"]["summary"])
    lines.append("")
    for step in result["policy_recommendation"]["next_steps"]:
        lines.append(f"- {step}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-records", default=DEFAULT_CANDIDATE_RECORDS)
    parser.add_argument("--real-margin-records", action="append", default=[])
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--expected-blocks", default=",".join(str(v) for v in DEFAULT_BLOCKS))
    parser.add_argument(
        "--expected-real-margin-workloads",
        default=",".join(DEFAULT_EXPECTED_REAL_MARGIN_WORKLOADS),
    )
    parser.add_argument(
        "--min-real-margin-measured-records",
        type=int,
        default=DEFAULT_MIN_REAL_MARGIN_MEASURED_RECORDS,
    )
    parser.add_argument("--min-speedup-percent", type=float, default=DEFAULT_MIN_SPEEDUP_PERCENT)
    parser.add_argument("--memory-cliff-gb", type=float, default=DEFAULT_MEMORY_CLIFF_GB)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    blocks = parse_csv_ints(args.expected_blocks)
    expected_real_margin_workloads = parse_csv_strings(args.expected_real_margin_workloads)
    candidate_path = Path(args.candidate_records)
    candidate_records = load_jsonl(candidate_path, required=True)
    real_margin_records: list[dict[str, Any]] = []
    missing_real_margin_paths: list[str] = []
    for raw_path in args.real_margin_records:
        path = Path(raw_path)
        rows = load_jsonl(path, required=False)
        if rows:
            real_margin_records.extend(rows)
        else:
            missing_real_margin_paths.append(str(path))

    candidate_events = event_rows(candidate_records, "candidate")
    real_margin_events = event_rows(real_margin_records, "real_margin")
    feature_rows = candidate_events + real_margin_events

    blockers: list[str] = []
    if not real_margin_records:
        blockers.append(
            "real-margin trace records are missing; run the XR61 trace-capture command before designing a margin-based adaptive policy"
        )
    blockers.extend(f"missing optional real-margin records: {path}" for path in missing_real_margin_paths)

    fixed_n = fixed_n_table(candidate_records, blocks)
    guarded = guarded_policy(
        candidate_records,
        blocks,
        args.min_speedup_percent,
        args.memory_cliff_gb,
    )
    real_signal = real_margin_summary(real_margin_events)
    real_coverage = real_margin_coverage_summary(
        real_margin_records,
        real_margin_events,
        expected_real_margin_workloads,
        blocks,
        args.min_real_margin_measured_records,
    )
    candidate_event_summary = event_summary(candidate_events)
    real_event_summary = event_summary(real_margin_events)
    if real_signal.get("available") and not real_coverage["sufficient_for_policy_design"]:
        blockers.extend(
            f"real-margin coverage incomplete: {issue}" for issue in real_coverage["issues"]
        )

    if real_signal.get("available") and real_coverage["sufficient_for_policy_design"]:
        decision_hint = "ready_for_adaptive_policy_design"
        recommendation = {
            "summary": (
                "Real-margin/top-k events are available. Use previous-pass features only for a causal "
                "pre-draft Adaptive-N policy, and compare any current-margin over-draft policy separately."
            ),
            "next_steps": [
                "Design an env-gated XR61 policy in the XR15 harness.",
                "Keep real-margin capture env-gated and report overhead.",
                "Run candidate, holdout, and sequential-oracle legs before any default-on claim.",
            ],
        }
    elif real_signal.get("available"):
        decision_hint = "needs_more_real_margin_coverage"
        recommendation = {
            "summary": (
                "Real-margin/top-k fields are present, but coverage is too narrow to justify "
                "Adaptive-N policy design. Treat this as a smoke validation of instrumentation only."
            ),
            "next_steps": [
                "Run the full XR61 real-margin trace-capture command when memory pressure allows.",
                "Re-run this script with the full trace-capture records.jsonl.",
                "Do not implement margin-based policy logic from the smoke slice alone.",
            ],
        }
    else:
        decision_hint = "needs_real_margin_trace_capture"
        recommendation = {
            "summary": (
                "The existing fixed-N records can recompute XR56-style policy behavior, but they cannot "
                "justify a margin/top-k Adaptive-N policy without XR57 real-margin trace rows."
            ),
            "next_steps": [
                "Run the XR61 real-margin trace-capture command.",
                "Re-run this script with --real-margin-records pointing at trace-capture records.jsonl.",
                "Do not implement margin-based policy logic until the report shows causal signal quality.",
            ],
        }

    result = {
        "schema_version": 1,
        "phase": "xr61_policy_search",
        "inputs": {
            "candidate_records": str(candidate_path),
            "real_margin_records": args.real_margin_records,
            "expected_blocks": blocks,
            "expected_real_margin_workloads": expected_real_margin_workloads,
            "min_real_margin_measured_records": args.min_real_margin_measured_records,
            "min_speedup_percent": args.min_speedup_percent,
            "memory_cliff_gb": args.memory_cliff_gb,
        },
        "decision_hint": decision_hint,
        "blockers": blockers,
        "fixed_n": fixed_n,
        "guarded_policy": guarded,
        "candidate_event_summary": candidate_event_summary,
        "real_margin_event_summary": real_event_summary,
        "real_margin_summary": real_signal,
        "real_margin_coverage": real_coverage,
        "policy_recommendation": recommendation,
        "generated_files": {
            "policy_candidates": str(out_dir / "policy_candidates.json"),
            "policy_report": str(out_dir / "policy_report.md"),
            "policy_features": str(out_dir / "policy_features.jsonl"),
        },
    }

    (out_dir / "policy_candidates.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "policy_report.md").write_text(render_report(result), encoding="utf-8")
    write_jsonl(out_dir / "policy_features.jsonl", feature_rows)

    print(f"XR61 policy search: {decision_hint}")
    print(f"policy_candidates: {out_dir / 'policy_candidates.json'}")
    print(f"policy_report: {out_dir / 'policy_report.md'}")
    print(f"policy_features: {out_dir / 'policy_features.jsonl'}")


if __name__ == "__main__":
    main()
