#!/usr/bin/env python3
"""Build the XR67 runtime-default decode eval decision artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_BASELINE = "native_decode_runtime_default"
DEFAULT_MIN_TRIALS = 3
DEFAULT_MEMORY_CLIFF_GB = 14.0
DEFAULT_MAX_REGRESSION_PERCENT = 5.0
DEFAULT_MIN_AGGREGATE_SPEEDUP_PERCENT = 5.0
ADAPTIVE_POLICY_PREFIX = "adaptive_policy_"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return value


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def pct_improvement(baseline: Any, candidate: Any) -> float | None:
    if not finite_number(baseline) or not finite_number(candidate):
        return None
    baseline = float(baseline)
    candidate = float(candidate)
    if baseline <= 0.0:
        return None
    return ((baseline - candidate) / baseline) * 100.0


def pct_regression(baseline: Any, candidate: Any) -> float | None:
    improvement = pct_improvement(baseline, candidate)
    if improvement is None:
        return None
    return -improvement


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def fmt_float(value: Any, digits: int = 3) -> str:
    if not finite_number(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def fmt_bool(value: bool) -> str:
    return "true" if value else "false"


def mean_record_decode_ms(aggregate: dict[str, Any]) -> float | None:
    records = aggregate.get("records", [])
    if not isinstance(records, list):
        return None
    values = [
        float(record["decode_ms"])
        for record in records
        if isinstance(record, dict) and finite_number(record.get("decode_ms"))
    ]
    return mean(values)


def profile_means(summary: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    decode_profile = summary.get("decode_profile", {})
    aggregates = decode_profile.get("aggregates", []) if isinstance(decode_profile, dict) else []
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in aggregates:
        if not isinstance(row, dict):
            continue
        variant = row.get("variant")
        workload_id = row.get("workload_id")
        if not isinstance(variant, str) or not isinstance(workload_id, str):
            continue
        out[(variant, workload_id)] = {
            "latency_mean_ms": nested_mean(row, "latency_ms"),
            "attention_kv_mutation_mean_ms": nested_mean(row, "attention_kv_mutation_ms"),
            "deferred_kv_eval_mean_ms": nested_mean(row, "deferred_kv_eval_ms"),
            "non_kv_forward_graph_mean_ms": nested_mean(row, "non_kv_forward_graph_ms"),
            "eval_sync_mean_ms": nested_mean(row, "eval_sync_ms"),
            "largest_stage_by_mean": row.get("largest_stage_by_mean"),
            "largest_stage_mean_ms": row.get("largest_stage_mean_ms"),
        }
    return out


def nested_mean(row: dict[str, Any], key: str) -> float | None:
    stats = row.get(key)
    if isinstance(stats, dict) and finite_number(stats.get("mean_ms")):
        return float(stats["mean_ms"])
    return None


def build_decode_rows(
    summary: dict[str, Any],
    baseline_variant: str,
    candidate_variants: list[str],
    include_workloads: set[str],
    exclude_workloads: set[str],
    min_trials: int,
    memory_cliff_gb: float,
    max_regression_percent: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    aggregates = [
        row for row in summary.get("aggregates", []) if isinstance(row, dict)
    ]
    by_key = {
        (str(row.get("variant")), str(row.get("workload_id"))): row
        for row in aggregates
        if isinstance(row.get("variant"), str) and isinstance(row.get("workload_id"), str)
    }
    workloads = sorted(
        row["workload_id"]
        for row in aggregates
        if row.get("variant") == baseline_variant and isinstance(row.get("workload_id"), str)
    )
    if include_workloads:
        workloads = [workload_id for workload_id in workloads if workload_id in include_workloads]
    if exclude_workloads:
        workloads = [
            workload_id for workload_id in workloads if workload_id not in exclude_workloads
        ]
    if not candidate_variants:
        variants = sorted(
            {
                row["variant"]
                for row in aggregates
                if isinstance(row.get("variant"), str) and row["variant"] != baseline_variant
            }
        )
    else:
        variants = candidate_variants

    profiles = profile_means(summary)
    blockers: list[str] = []
    rows: list[dict[str, Any]] = []
    candidate_rollups: list[dict[str, Any]] = []

    if not workloads:
        blockers.append(f"missing XR67 baseline variant {baseline_variant}")

    for candidate_variant in variants:
        candidate_rows: list[dict[str, Any]] = []
        baseline_decode_sum = 0.0
        candidate_decode_sum = 0.0
        decode_sum_count = 0
        for workload_id in workloads:
            baseline = by_key.get((baseline_variant, workload_id))
            candidate = by_key.get((candidate_variant, workload_id))
            if baseline is None:
                blockers.append(f"missing baseline row for {workload_id}")
                continue
            if candidate is None:
                blockers.append(f"missing {candidate_variant} row for {workload_id}")
                continue

            baseline_decode = mean_record_decode_ms(baseline)
            candidate_decode = mean_record_decode_ms(candidate)
            if baseline_decode is not None and candidate_decode is not None:
                baseline_decode_sum += baseline_decode
                candidate_decode_sum += candidate_decode
                decode_sum_count += 1

            p50_regression = pct_regression(
                baseline.get("raw_decode_p50_ms"), candidate.get("raw_decode_p50_ms")
            )
            p95_regression = pct_regression(
                baseline.get("raw_decode_p95_ms"), candidate.get("raw_decode_p95_ms")
            )
            p99_regression = pct_regression(
                baseline.get("raw_decode_p99_ms"), candidate.get("raw_decode_p99_ms")
            )
            steady_p50_regression = pct_regression(
                baseline.get("steady_decode_p50_ms"), candidate.get("steady_decode_p50_ms")
            )
            decode_speedup = pct_improvement(baseline_decode, candidate_decode)
            peak_delta = pct_improvement(
                baseline.get("peak_mlx_max_gb"), candidate.get("peak_mlx_max_gb")
            )
            trial_count = int(candidate.get("trial_count", 0) or 0)
            passed_trials = int(candidate.get("passed_trials", 0) or 0)
            correctness_trials = int(candidate.get("correctness_passed_trials", 0) or 0)
            peak_mlx = candidate.get("peak_mlx_max_gb")
            trials_ok = (
                trial_count >= min_trials
                and passed_trials == trial_count
                and correctness_trials == trial_count
            )
            memory_ok = bool(candidate.get("memory_gate_passed")) and (
                not finite_number(peak_mlx) or float(peak_mlx) < memory_cliff_gb
            )
            regression_ok = all(
                value is not None and value <= max_regression_percent
                for value in [
                    p50_regression,
                    p95_regression,
                    p99_regression,
                    steady_p50_regression,
                ]
            )
            row = {
                "candidate_variant": candidate_variant,
                "baseline_variant": baseline_variant,
                "workload_id": workload_id,
                "trial_count": trial_count,
                "passed_trials": passed_trials,
                "correctness_passed_trials": correctness_trials,
                "baseline_decode_mean_ms": baseline_decode,
                "candidate_decode_mean_ms": candidate_decode,
                "decode_speedup_percent": decode_speedup,
                "raw_p50_regression_percent": p50_regression,
                "raw_p95_regression_percent": p95_regression,
                "raw_p99_regression_percent": p99_regression,
                "steady_p50_regression_percent": steady_p50_regression,
                "peak_mlx_max_gb": peak_mlx,
                "peak_mlx_delta_percent": peak_delta,
                "rss_max_mb": candidate.get("rss_max_mb"),
                "active_kv_max_bytes": candidate.get("active_kv_max_bytes"),
                "trials_ok": trials_ok,
                "memory_ok": memory_ok,
                "regression_ok": regression_ok,
                "row_passed": trials_ok and memory_ok and regression_ok,
                "profile": profiles.get((candidate_variant, workload_id), {}),
            }
            rows.append(row)
            candidate_rows.append(row)

        aggregate_speedup = pct_improvement(baseline_decode_sum, candidate_decode_sum)
        all_rows_passed = bool(candidate_rows) and all(row["row_passed"] for row in candidate_rows)
        aggregate_ok = (
            aggregate_speedup is not None
            and aggregate_speedup >= DEFAULT_MIN_AGGREGATE_SPEEDUP_PERCENT
        )
        candidate_rollups.append(
            {
                "candidate_variant": candidate_variant,
                "workload_count": len(candidate_rows),
                "decode_sum_count": decode_sum_count,
                "baseline_decode_sum_ms": baseline_decode_sum if decode_sum_count else None,
                "candidate_decode_sum_ms": candidate_decode_sum if decode_sum_count else None,
                "aggregate_speedup_percent": aggregate_speedup,
                "worst_p50_regression_percent": max_optional(
                    row["raw_p50_regression_percent"] for row in candidate_rows
                ),
                "worst_p95_regression_percent": max_optional(
                    row["raw_p95_regression_percent"] for row in candidate_rows
                ),
                "worst_p99_regression_percent": max_optional(
                    row["raw_p99_regression_percent"] for row in candidate_rows
                ),
                "max_peak_mlx_gb": max_optional(row["peak_mlx_max_gb"] for row in candidate_rows),
                "all_rows_passed": all_rows_passed,
                "aggregate_ok": aggregate_ok,
                "followup_candidate": all_rows_passed and aggregate_ok,
            }
        )

    return rows, candidate_rollups, blockers


def max_optional(values: Any) -> float | None:
    numbers = [float(value) for value in values if finite_number(value)]
    if not numbers:
        return None
    return max(numbers)


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("LABEL must not be empty")
    return label, Path(path)


def adaptive_policy(summary: dict[str, Any]) -> dict[str, Any] | None:
    policies = summary.get("policy_summaries", [])
    if not isinstance(policies, list):
        return None
    for policy in policies:
        if isinstance(policy, dict) and str(policy.get("policy_name", "")).startswith(
            ADAPTIVE_POLICY_PREFIX
        ):
            return policy
    return None


def build_mtp_rows(labeled_paths: list[tuple[str, Path]]) -> tuple[list[dict[str, Any]], bool | None]:
    rows: list[dict[str, Any]] = []
    for label, path in labeled_paths:
        summary = load_json(path)
        policy = adaptive_policy(summary)
        row = {
            "label": label,
            "summary_path": str(path),
            "decision": summary.get("decision"),
            "status": summary.get("status"),
            "record_count": summary.get("record_count"),
            "measured_record_count": summary.get("measured_record_count"),
            "exact_record_count": summary.get("exact_record_count"),
            "blockers": summary.get("blockers", []),
            "failed_hypotheses": summary.get("failed_hypotheses", []),
            "adaptive_policy": policy,
        }
        if policy:
            row.update(
                {
                    "aggregate_speedup_percent": policy.get("aggregate_speedup_percent"),
                    "selected_workloads": policy.get("selected_workloads", []),
                    "total_accepted_draft_tokens": policy.get("total_accepted_draft_tokens"),
                    "total_attempted_draft_tokens": policy.get("total_attempted_draft_tokens"),
                    "weighted_acceptance_rate": policy.get("weighted_acceptance_rate"),
                    "max_peak_memory_gb": policy.get("max_peak_memory_gb"),
                }
            )
        rows.append(row)

    if len(rows) < 2:
        return rows, None

    baseline = rows[0]
    side_effect_passed = True
    for row in rows[1:]:
        same_exact = row.get("exact_record_count") == baseline.get("exact_record_count")
        same_selected = row.get("selected_workloads") == baseline.get("selected_workloads")
        same_accepted = row.get("total_accepted_draft_tokens") == baseline.get(
            "total_accepted_draft_tokens"
        )
        same_attempted = row.get("total_attempted_draft_tokens") == baseline.get(
            "total_attempted_draft_tokens"
        )
        clean = (
            row.get("status") == "completed"
            and not row.get("blockers")
            and not row.get("failed_hypotheses")
        )
        row["side_effect_passed"] = bool(
            clean and same_exact and same_selected and same_accepted and same_attempted
        )
        side_effect_passed = side_effect_passed and row["side_effect_passed"]
    baseline["side_effect_passed"] = True
    return rows, side_effect_passed


def render_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# XR67 Native Decode Deferred-KV Eval Barrier")
    lines.append("")
    lines.append(f"Decision: `{report['decision']}`")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- XR06 summary: `{report['summary_path']}`")
    lines.append(f"- Baseline variant: `{report['baseline_variant']}`")
    if report.get("include_workloads"):
        lines.append(f"- Included workloads: `{', '.join(report['include_workloads'])}`")
    if report.get("exclude_workloads"):
        lines.append(f"- Excluded workloads: `{', '.join(report['exclude_workloads'])}`")
    lines.append(f"- Git SHA: `{report.get('git_sha', 'unknown')}`")
    lines.append(f"- Git status: `{report.get('git_status_short', '').strip()}`")
    lines.append("")
    lines.append("## Candidate Rollup")
    lines.append("")
    lines.append(
        "| Candidate | Workloads | Aggregate speedup % | Worst p50 reg % | Worst p95 reg % | Worst p99 reg % | Max peak GB | Rows passed | Follow-up |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|")
    for row in report["candidate_rollups"]:
        lines.append(
            "| {candidate_variant} | {workload_count} | {speedup} | {p50} | {p95} | {p99} | {peak} | `{rows_passed}` | `{followup}` |".format(
                candidate_variant=row["candidate_variant"],
                workload_count=row["workload_count"],
                speedup=fmt_float(row["aggregate_speedup_percent"]),
                p50=fmt_float(row["worst_p50_regression_percent"]),
                p95=fmt_float(row["worst_p95_regression_percent"]),
                p99=fmt_float(row["worst_p99_regression_percent"]),
                peak=fmt_float(row["max_peak_mlx_gb"]),
                rows_passed=fmt_bool(bool(row["all_rows_passed"])),
                followup=fmt_bool(bool(row["followup_candidate"])),
            )
        )
    lines.append("")
    lines.append("## Workload Rows")
    lines.append("")
    lines.append(
        "| Candidate | Workload | Decode speedup % | p50 reg % | p95 reg % | p99 reg % | Peak GB | Active KV bytes | Largest profiled stage | Row passed |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|---|")
    for row in report["decode_rows"]:
        profile = row.get("profile") or {}
        largest = profile.get("largest_stage_by_mean") or "n/a"
        if finite_number(profile.get("largest_stage_mean_ms")):
            largest = f"{largest} ({fmt_float(profile['largest_stage_mean_ms'])} ms)"
        lines.append(
            "| {candidate} | {workload} | {speedup} | {p50} | {p95} | {p99} | {peak} | {kv} | {largest} | `{passed}` |".format(
                candidate=row["candidate_variant"],
                workload=row["workload_id"],
                speedup=fmt_float(row["decode_speedup_percent"]),
                p50=fmt_float(row["raw_p50_regression_percent"]),
                p95=fmt_float(row["raw_p95_regression_percent"]),
                p99=fmt_float(row["raw_p99_regression_percent"]),
                peak=fmt_float(row["peak_mlx_max_gb"]),
                kv=row.get("active_kv_max_bytes") or "n/a",
                largest=largest,
                passed=fmt_bool(bool(row["row_passed"])),
            )
        )
    lines.append("")
    if report["mtp_rows"]:
        lines.append("## MTP Side Effect")
        lines.append("")
        side_effect = report.get("mtp_side_effect_passed")
        lines.append(
            f"Side-effect gate: `{'not_run' if side_effect is None else fmt_bool(bool(side_effect))}`"
        )
        lines.append("")
        lines.append(
            "| Label | Decision | Exact records | Aggregate speedup % | Accepted/attempted | Weighted acceptance | Selected workloads | Side-effect passed |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---|---|")
        for row in report["mtp_rows"]:
            selected = ", ".join(row.get("selected_workloads") or [])
            accepted = row.get("total_accepted_draft_tokens")
            attempted = row.get("total_attempted_draft_tokens")
            accepted_text = "n/a"
            if finite_number(accepted) and finite_number(attempted):
                accepted_text = f"{int(accepted)}/{int(attempted)}"
            side_passed = row.get("side_effect_passed")
            lines.append(
                "| {label} | `{decision}` | {exact} | {speedup} | {accepted} | {weighted} | {selected} | `{side}` |".format(
                    label=row["label"],
                    decision=row.get("decision", "n/a"),
                    exact=row.get("exact_record_count", "n/a"),
                    speedup=fmt_float(row.get("aggregate_speedup_percent")),
                    accepted=accepted_text,
                    weighted=fmt_float(row.get("weighted_acceptance_rate"), 6),
                    selected=selected or "n/a",
                    side="n/a" if side_passed is None else fmt_bool(bool(side_passed)),
                )
            )
        lines.append("")
    if report["blockers"]:
        lines.append("## Blockers")
        lines.append("")
        for blocker in report["blockers"]:
            lines.append(f"- {blocker}")
        lines.append("")
    if report.get("source_blockers"):
        lines.append("## Source Blockers")
        lines.append("")
        for blocker in report["source_blockers"]:
            lines.append(f"- {blocker}")
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- This report does not change runtime defaults; it reinterprets XR06 output with the XR65 runtime default as the baseline."
    )
    lines.append(
        "- A follow-up candidate requires exactness, memory safety, no p50/p95/p99 regression above 5%, and at least 5% aggregate decode speedup."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path, help="XR06 summary.json")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--baseline-variant", default=DEFAULT_BASELINE)
    parser.add_argument("--candidate-variant", action="append", default=[])
    parser.add_argument("--include-workload", action="append", default=[])
    parser.add_argument("--exclude-workload", action="append", default=[])
    parser.add_argument("--min-trials", type=int, default=DEFAULT_MIN_TRIALS)
    parser.add_argument("--memory-cliff-gb", type=float, default=DEFAULT_MEMORY_CLIFF_GB)
    parser.add_argument(
        "--max-regression-percent",
        type=float,
        default=DEFAULT_MAX_REGRESSION_PERCENT,
    )
    parser.add_argument(
        "--min-aggregate-speedup-percent",
        type=float,
        default=DEFAULT_MIN_AGGREGATE_SPEEDUP_PERCENT,
    )
    parser.add_argument(
        "--mtp-summary",
        action="append",
        default=[],
        type=parse_labeled_path,
        help="Optional MTP side-effect summary as LABEL=PATH. First label is baseline.",
    )
    args = parser.parse_args()

    summary = load_json(args.summary)
    include_workloads = set(args.include_workload)
    exclude_workloads = set(args.exclude_workload)
    decode_rows, candidate_rollups, blockers = build_decode_rows(
        summary,
        args.baseline_variant,
        args.candidate_variant,
        include_workloads,
        exclude_workloads,
        args.min_trials,
        args.memory_cliff_gb,
        args.max_regression_percent,
    )
    for rollup in candidate_rollups:
        aggregate_speedup = rollup.get("aggregate_speedup_percent")
        rollup["aggregate_ok"] = (
            aggregate_speedup is not None
            and aggregate_speedup >= args.min_aggregate_speedup_percent
        )
        rollup["followup_candidate"] = bool(
            rollup["all_rows_passed"] and rollup["aggregate_ok"]
        )

    mtp_rows, mtp_side_effect_passed = build_mtp_rows(args.mtp_summary)
    followups = [
        row["candidate_variant"] for row in candidate_rollups if row["followup_candidate"]
    ]
    if blockers:
        decision = "blocked_with_evidence"
    elif followups and (mtp_side_effect_passed is not False):
        decision = "followup_candidate"
    else:
        decision = "reject_default_change"

    report = {
        "schema_version": 1,
        "goal": "XR67-native-decode-deferred-kv-eval-barrier",
        "decision": decision,
        "summary_path": str(args.summary),
        "baseline_variant": args.baseline_variant,
        "candidate_variants": args.candidate_variant,
        "include_workloads": sorted(include_workloads),
        "exclude_workloads": sorted(exclude_workloads),
        "min_trials": args.min_trials,
        "memory_cliff_gb": args.memory_cliff_gb,
        "max_regression_percent": args.max_regression_percent,
        "min_aggregate_speedup_percent": args.min_aggregate_speedup_percent,
        "git_sha": summary.get("git_sha"),
        "git_status_short": summary.get("git_status_short"),
        "command": summary.get("command"),
        "decode_rows": decode_rows,
        "candidate_rollups": candidate_rollups,
        "followup_candidates": followups,
        "mtp_rows": mtp_rows,
        "mtp_side_effect_passed": mtp_side_effect_passed,
        "source_blockers": summary.get("blockers", []),
        "source_failed_hypotheses": summary.get("failed_hypotheses", []),
        "blockers": blockers,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "xr67-deferred-kv-eval-summary.json"
    md_path = args.out_dir / "xr67-deferred-kv-eval-summary.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_report(report), encoding="utf-8")
    print(f"decision: {decision}")
    print(f"json: {json_path}")
    print(f"report: {md_path}")


if __name__ == "__main__":
    main()
