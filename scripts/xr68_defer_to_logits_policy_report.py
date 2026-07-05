#!/usr/bin/env python3
"""Build the XR68 scoped defer_to_logits policy decision artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


DEFAULT_BASELINE = "native_decode_runtime_default"
DEFAULT_CANDIDATE = "native_decode_eval_defer_to_logits"
DEFAULT_MIN_AGGREGATE_SPEEDUP = 5.0
DEFAULT_MAX_REGRESSION = 5.0
DEFAULT_MEMORY_CLIFF_GB = 14.0
ADAPTIVE_POLICY_PREFIX = "adaptive_policy_"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return value


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def pct_improvement(baseline: Any, candidate: Any) -> float | None:
    if not finite(baseline) or not finite(candidate):
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


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil(q * len(ordered)) - 1
    index = min(max(index, 0), len(ordered) - 1)
    return ordered[index]


def stats(values: Iterable[Any]) -> dict[str, Any]:
    numbers = [float(value) for value in values if finite(value)]
    if not numbers:
        return {
            "count": 0,
            "min_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
            "mean_ms": None,
            "p95_to_p50": None,
            "p99_to_p50": None,
        }
    p50 = percentile(numbers, 0.50)
    p95 = percentile(numbers, 0.95)
    p99 = percentile(numbers, 0.99)
    return {
        "count": len(numbers),
        "min_ms": min(numbers),
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "max_ms": max(numbers),
        "mean_ms": sum(numbers) / len(numbers),
        "p95_to_p50": ratio(p95, p50),
        "p99_to_p50": ratio(p99, p50),
    }


def ratio(num: Any, den: Any) -> float | None:
    if not finite(num) or not finite(den) or float(den) <= 0.0:
        return None
    return float(num) / float(den)


def mean(values: Iterable[Any]) -> float | None:
    numbers = [float(value) for value in values if finite(value)]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def max_optional(values: Iterable[Any]) -> float | None:
    numbers = [float(value) for value in values if finite(value)]
    if not numbers:
        return None
    return max(numbers)


def fmt(value: Any, digits: int = 3) -> str:
    if not finite(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def bool_text(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "true" if value else "false"


def aggregate_map(summary: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = summary.get("aggregates", [])
    return {
        (row["variant"], row["workload_id"]): row
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("variant"), str)
        and isinstance(row.get("workload_id"), str)
    }


def record_values(aggregate: dict[str, Any], key: str) -> list[Any]:
    records = aggregate.get("records", [])
    if not isinstance(records, list):
        return []
    return [record.get(key) for record in records if isinstance(record, dict)]


def token_latencies(aggregate: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for record in aggregate.get("records", []):
        if not isinstance(record, dict):
            continue
        for value in record.get("decode_token_latencies_ms", []):
            if finite(value):
                values.append(float(value))
    return values


def regression_ok(value: float | None, max_regression: float) -> bool:
    return value is not None and value <= max_regression


def adaptive_policy(summary: dict[str, Any]) -> dict[str, Any] | None:
    for policy in summary.get("policy_summaries", []):
        if isinstance(policy, dict) and str(policy.get("policy_name", "")).startswith(
            ADAPTIVE_POLICY_PREFIX
        ):
            return policy
    return None


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("LABEL must not be empty")
    return label, Path(path)


def mtp_rows(paths: list[tuple[str, Path]]) -> tuple[list[dict[str, Any]], bool | None]:
    rows: list[dict[str, Any]] = []
    for label, path in paths:
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
            "policy_name": policy.get("policy_name") if policy else None,
            "aggregate_speedup_percent": policy.get("aggregate_speedup_percent") if policy else None,
            "selected_workloads": policy.get("selected_workloads", []) if policy else [],
            "total_accepted_draft_tokens": policy.get("total_accepted_draft_tokens") if policy else None,
            "total_attempted_draft_tokens": policy.get("total_attempted_draft_tokens") if policy else None,
            "weighted_acceptance_rate": policy.get("weighted_acceptance_rate") if policy else None,
        }
        rows.append(row)

    if len(rows) < 2:
        return rows, None

    baseline = rows[0]
    baseline_clean = (
        baseline.get("status") == "completed"
        and not baseline.get("blockers")
        and not baseline.get("failed_hypotheses")
    )
    all_passed = baseline_clean
    baseline["side_effect_passed"] = baseline_clean
    for row in rows[1:]:
        clean = (
            row.get("status") == "completed"
            and not row.get("blockers")
            and not row.get("failed_hypotheses")
        )
        row["side_effect_passed"] = bool(
            clean
            and row.get("exact_record_count") == baseline.get("exact_record_count")
            and row.get("selected_workloads") == baseline.get("selected_workloads")
            and row.get("total_accepted_draft_tokens")
            == baseline.get("total_accepted_draft_tokens")
            and row.get("total_attempted_draft_tokens")
            == baseline.get("total_attempted_draft_tokens")
            and row.get("weighted_acceptance_rate") == baseline.get("weighted_acceptance_rate")
        )
        all_passed = all_passed and row["side_effect_passed"]
    return rows, all_passed


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    summary = load_json(args.summary)
    by_key = aggregate_map(summary)
    workloads = sorted(
        workload_id
        for variant, workload_id in by_key
        if variant == args.baseline_variant
    )
    if args.workload_id:
        selected = set(args.workload_id)
        workloads = [workload_id for workload_id in workloads if workload_id in selected]

    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    baseline_decode_sum = 0.0
    candidate_decode_sum = 0.0
    decode_sum_count = 0

    for workload_id in workloads:
        baseline = by_key.get((args.baseline_variant, workload_id))
        candidate = by_key.get((args.candidate_variant, workload_id))
        if baseline is None:
            blockers.append(f"missing baseline row for {workload_id}")
            continue
        if candidate is None:
            blockers.append(f"missing candidate row for {workload_id}")
            continue

        baseline_decode_ms = mean(record_values(baseline, "decode_ms"))
        candidate_decode_ms = mean(record_values(candidate, "decode_ms"))
        if baseline_decode_ms is not None and candidate_decode_ms is not None:
            baseline_decode_sum += baseline_decode_ms
            candidate_decode_sum += candidate_decode_ms
            decode_sum_count += 1

        baseline_cadence = stats(token_latencies(baseline))
        candidate_cadence = stats(token_latencies(candidate))
        ttft_baseline = stats(record_values(baseline, "ttft_ms"))
        ttft_candidate = stats(record_values(candidate, "ttft_ms"))
        p50_reg = pct_regression(baseline.get("raw_decode_p50_ms"), candidate.get("raw_decode_p50_ms"))
        p95_reg = pct_regression(baseline.get("raw_decode_p95_ms"), candidate.get("raw_decode_p95_ms"))
        p99_reg = pct_regression(baseline.get("raw_decode_p99_ms"), candidate.get("raw_decode_p99_ms"))
        cadence_p50_reg = pct_regression(baseline_cadence["p50_ms"], candidate_cadence["p50_ms"])
        cadence_p95_reg = pct_regression(baseline_cadence["p95_ms"], candidate_cadence["p95_ms"])
        cadence_p99_reg = pct_regression(baseline_cadence["p99_ms"], candidate_cadence["p99_ms"])
        ttft_p50_reg = pct_regression(ttft_baseline["p50_ms"], ttft_candidate["p50_ms"])
        peak_mlx = candidate.get("peak_mlx_max_gb")
        trial_count = int(candidate.get("trial_count", 0) or 0)
        passed_trials = int(candidate.get("passed_trials", 0) or 0)
        correctness_trials = int(candidate.get("correctness_passed_trials", 0) or 0)
        exact_ok = trial_count > 0 and passed_trials == trial_count and correctness_trials == trial_count
        memory_ok = bool(candidate.get("memory_gate_passed")) and (
            not finite(peak_mlx) or float(peak_mlx) < args.memory_cliff_gb
        )
        latency_ok = all(
            regression_ok(value, args.max_regression_percent)
            for value in [p50_reg, p95_reg, p99_reg]
        )
        cadence_ok = all(
            regression_ok(value, args.max_regression_percent)
            for value in [cadence_p50_reg, cadence_p95_reg, cadence_p99_reg]
        )
        row_passed = exact_ok and memory_ok and latency_ok and cadence_ok
        rows.append(
            {
                "workload_id": workload_id,
                "baseline_variant": args.baseline_variant,
                "candidate_variant": args.candidate_variant,
                "trial_count": trial_count,
                "passed_trials": passed_trials,
                "correctness_passed_trials": correctness_trials,
                "baseline_decode_mean_ms": baseline_decode_ms,
                "candidate_decode_mean_ms": candidate_decode_ms,
                "decode_speedup_percent": pct_improvement(baseline_decode_ms, candidate_decode_ms),
                "raw_p50_regression_percent": p50_reg,
                "raw_p95_regression_percent": p95_reg,
                "raw_p99_regression_percent": p99_reg,
                "cadence_p50_regression_percent": cadence_p50_reg,
                "cadence_p95_regression_percent": cadence_p95_reg,
                "cadence_p99_regression_percent": cadence_p99_reg,
                "ttft_p50_regression_percent": ttft_p50_reg,
                "peak_mlx_max_gb": peak_mlx,
                "rss_max_mb": candidate.get("rss_max_mb"),
                "active_kv_max_bytes": candidate.get("active_kv_max_bytes"),
                "exact_ok": exact_ok,
                "memory_ok": memory_ok,
                "latency_ok": latency_ok,
                "cadence_ok": cadence_ok,
                "row_passed": row_passed,
                "baseline_cadence": baseline_cadence,
                "candidate_cadence": candidate_cadence,
                "baseline_ttft": ttft_baseline,
                "candidate_ttft": ttft_candidate,
            }
        )

    aggregate_speedup = pct_improvement(baseline_decode_sum, candidate_decode_sum)
    aggregate_ok = aggregate_speedup is not None and aggregate_speedup >= args.min_aggregate_speedup_percent
    all_rows_passed = bool(rows) and all(row["row_passed"] for row in rows)
    mtp, mtp_passed = mtp_rows(args.mtp_summary)
    source_blockers = summary.get("blockers", [])
    if source_blockers:
        blockers.extend(str(blocker) for blocker in source_blockers)

    if blockers:
        decision = "blocked_with_evidence"
    elif not aggregate_ok or not all_rows_passed:
        decision = "reject_candidate"
    elif not mtp or mtp_passed is None:
        decision = "needs_more_data"
    elif mtp and mtp_passed is False:
        decision = "reject_candidate"
    elif mtp_passed:
        decision = "accept_candidate"
    else:
        decision = "reject_candidate"

    return {
        "schema_version": 1,
        "goal": "XR68-scoped-defer-to-logits-policy",
        "decision": decision,
        "summary_path": str(args.summary),
        "baseline_variant": args.baseline_variant,
        "candidate_variant": args.candidate_variant,
        "git_sha": summary.get("git_sha"),
        "git_status_short": summary.get("git_status_short"),
        "command": summary.get("command"),
        "environment": summary.get("environment"),
        "source_decision": summary.get("decision"),
        "source_blockers": summary.get("blockers", []),
        "workload_count": len(rows),
        "baseline_decode_sum_ms": baseline_decode_sum if decode_sum_count else None,
        "candidate_decode_sum_ms": candidate_decode_sum if decode_sum_count else None,
        "aggregate_speedup_percent": aggregate_speedup,
        "aggregate_ok": aggregate_ok,
        "all_rows_passed": all_rows_passed,
        "max_regression_percent": args.max_regression_percent,
        "min_aggregate_speedup_percent": args.min_aggregate_speedup_percent,
        "memory_cliff_gb": args.memory_cliff_gb,
        "rows": rows,
        "mtp_rows": mtp,
        "mtp_side_effect_passed": mtp_passed,
        "blockers": blockers,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# XR68 Scoped defer_to_logits Policy")
    lines.append("")
    lines.append(f"Decision: `{report['decision']}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Baseline: `{report['baseline_variant']}`")
    lines.append(f"- Candidate: `{report['candidate_variant']}`")
    lines.append(f"- XR06 summary: `{report['summary_path']}`")
    lines.append(f"- Aggregate decode: `{fmt(report['baseline_decode_sum_ms'])} -> {fmt(report['candidate_decode_sum_ms'])} ms` (`{fmt(report['aggregate_speedup_percent'])}%`)")
    lines.append(f"- MTP side-effect gate: `{bool_text(report['mtp_side_effect_passed'])}`")
    lines.append("")
    lines.append("## Workload Gates")
    lines.append("")
    lines.append("| Workload | Decode speedup % | p50 reg % | p95 reg % | p99 reg % | Cadence p50 reg % | Cadence p95 reg % | Cadence p99 reg % | TTFT p50 reg % | Peak GB | Exact | Memory | Cadence | Row |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|")
    for row in report["rows"]:
        lines.append(
            "| {workload} | {speedup} | {p50} | {p95} | {p99} | {cp50} | {cp95} | {cp99} | {ttft} | {peak} | `{exact}` | `{memory}` | `{cadence}` | `{passed}` |".format(
                workload=row["workload_id"],
                speedup=fmt(row["decode_speedup_percent"]),
                p50=fmt(row["raw_p50_regression_percent"]),
                p95=fmt(row["raw_p95_regression_percent"]),
                p99=fmt(row["raw_p99_regression_percent"]),
                cp50=fmt(row["cadence_p50_regression_percent"]),
                cp95=fmt(row["cadence_p95_regression_percent"]),
                cp99=fmt(row["cadence_p99_regression_percent"]),
                ttft=fmt(row["ttft_p50_regression_percent"]),
                peak=fmt(row["peak_mlx_max_gb"]),
                exact=bool_text(row["exact_ok"]),
                memory=bool_text(row["memory_ok"]),
                cadence=bool_text(row["cadence_ok"]),
                passed=bool_text(row["row_passed"]),
            )
        )
    lines.append("")
    lines.append("## Cadence Detail")
    lines.append("")
    lines.append("| Workload | Baseline p50/p95/p99 | Candidate p50/p95/p99 | Baseline p99:p50 | Candidate p99:p50 |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in report["rows"]:
        base = row["baseline_cadence"]
        cand = row["candidate_cadence"]
        lines.append(
            "| {workload} | {bp50}/{bp95}/{bp99} | {cp50}/{cp95}/{cp99} | {bratio} | {cratio} |".format(
                workload=row["workload_id"],
                bp50=fmt(base["p50_ms"]),
                bp95=fmt(base["p95_ms"]),
                bp99=fmt(base["p99_ms"]),
                cp50=fmt(cand["p50_ms"]),
                cp95=fmt(cand["p95_ms"]),
                cp99=fmt(cand["p99_ms"]),
                bratio=fmt(base["p99_to_p50"]),
                cratio=fmt(cand["p99_to_p50"]),
            )
        )
    lines.append("")
    if report["mtp_rows"]:
        lines.append("## MTP Side Effect")
        lines.append("")
        lines.append("| Label | Decision | Exact records | Speedup % | Accepted/attempted | Weighted acceptance | Selected workloads | Side-effect |")
        lines.append("|---|---|---:|---:|---:|---:|---|---|")
        for row in report["mtp_rows"]:
            accepted = row.get("total_accepted_draft_tokens")
            attempted = row.get("total_attempted_draft_tokens")
            accepted_text = "n/a"
            if finite(accepted) and finite(attempted):
                accepted_text = f"{int(accepted)}/{int(attempted)}"
            lines.append(
                "| {label} | `{decision}` | {exact} | {speedup} | {accepted} | {weighted} | {selected} | `{side}` |".format(
                    label=row["label"],
                    decision=row.get("decision", "n/a"),
                    exact=row.get("exact_record_count", "n/a"),
                    speedup=fmt(row.get("aggregate_speedup_percent")),
                    accepted=accepted_text,
                    weighted=fmt(row.get("weighted_acceptance_rate"), 6),
                    selected=", ".join(row.get("selected_workloads") or []) or "n/a",
                    side=bool_text(row.get("side_effect_passed")),
                )
            )
        lines.append("")
    if report["source_blockers"]:
        lines.append("## Source Blockers")
        lines.append("")
        for blocker in report["source_blockers"]:
            lines.append(f"- {blocker}")
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This is a default-off candidate; runtime defaults remain unchanged.")
    lines.append("- XR06 decode-token latency traces are treated as the streaming cadence proxy.")
    lines.append("- The long-context prefill policy is applied to both baseline and candidate to isolate decode scheduling while keeping 16K under the tiny16 memory gate.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline-variant", default=DEFAULT_BASELINE)
    parser.add_argument("--candidate-variant", default=DEFAULT_CANDIDATE)
    parser.add_argument("--workload-id", action="append", default=[])
    parser.add_argument("--max-regression-percent", type=float, default=DEFAULT_MAX_REGRESSION)
    parser.add_argument("--min-aggregate-speedup-percent", type=float, default=DEFAULT_MIN_AGGREGATE_SPEEDUP)
    parser.add_argument("--memory-cliff-gb", type=float, default=DEFAULT_MEMORY_CLIFF_GB)
    parser.add_argument("--mtp-summary", action="append", default=[], type=parse_labeled_path)
    args = parser.parse_args()

    report = build_report(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "xr68-defer-to-logits-policy-summary.json"
    md_path = args.out_dir / "xr68-defer-to-logits-policy-summary.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"decision: {report['decision']}")
    print(f"json: {json_path}")
    print(f"report: {md_path}")


if __name__ == "__main__":
    main()
