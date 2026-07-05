#!/usr/bin/env python3
"""Build a scoped MTP opt-in decision artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any


POLICY_NAME = "adaptive_policy_xr61-real-margin-v1"
WARMUP_AMORTIZED_VARIANT = "native_decode_runtime_default_warmup_amortized_4"
DEFAULT_SELECTED_WORKLOADS = ("chat_short_1k_001", "tool_json_1k_001")
DEFAULT_PROTECTED_WORKLOADS = ("mtp_candidate_1k_001",)


def load_json(path: Path, *, required: bool = True) -> dict[str, Any] | None:
    if not path.exists():
        if required:
            raise SystemExit(f"{path}: JSON file does not exist")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: invalid JSON: {exc}") from exc


def records(summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    raw = (summary or {}).get("records") or []
    return raw if isinstance(raw, list) else []


def measured_records(summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [record for record in records(summary) if record.get("measured")]


def policy_summary(summary: dict[str, Any] | None, policy_name: str = POLICY_NAME) -> dict[str, Any] | None:
    for policy in (summary or {}).get("policy_summaries") or []:
        if policy.get("policy_name") == policy_name:
            return policy
    return None


def exactness(summary: dict[str, Any] | None) -> dict[str, Any]:
    rows = records(summary)
    measured = measured_records(summary)
    exact_all = sum(1 for record in rows if (record.get("comparison") or {}).get("byte_identical"))
    exact_measured = sum(
        1 for record in measured if (record.get("comparison") or {}).get("byte_identical")
    )
    return {
        "records": len(rows),
        "measured_records": len(measured),
        "exact_records": exact_all,
        "exact_measured_records": exact_measured,
        "passed": bool(measured) and exact_measured == len(measured),
    }


def generated_tokens(record: dict[str, Any]) -> list[int]:
    return [int(token) for token in ((record.get("mtp") or {}).get("generated_tokens") or [])]


def record_key(record: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(record.get("workload_id")),
        str(record.get("trial_kind")),
        int(record.get("trial_index") or 0),
        int(record.get("block_size") or 0),
    )


def compare_oracle(
    candidate_summary: dict[str, Any] | None,
    oracle_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    if candidate_summary is None or oracle_summary is None:
        return {
            "present": oracle_summary is not None,
            "passed": False,
            "compared_records": 0,
            "missing_records": [],
            "extra_records": [],
            "mismatches": [],
            "reason": "candidate or oracle summary missing",
        }
    candidate_rows = {record_key(record): record for record in measured_records(candidate_summary)}
    oracle_rows = {record_key(record): record for record in measured_records(oracle_summary)}
    missing = [str(key) for key in sorted(candidate_rows) if key not in oracle_rows]
    extra = [str(key) for key in sorted(oracle_rows) if key not in candidate_rows]
    mismatches: list[str] = []
    compared = 0
    for key, candidate in sorted(candidate_rows.items()):
        oracle = oracle_rows.get(key)
        if oracle is None:
            continue
        compared += 1
        if generated_tokens(candidate) != generated_tokens(oracle):
            mismatches.append(str(key))
    return {
        "present": True,
        "passed": not missing and not extra and not mismatches and compared > 0,
        "compared_records": compared,
        "missing_records": missing,
        "extra_records": extra,
        "mismatches": mismatches,
        "reason": "record-by-record generated token comparison",
    }


def speedup_percent(baseline_ms: float, candidate_ms: float) -> float:
    if baseline_ms <= 0.0:
        return 0.0
    return (baseline_ms - candidate_ms) / baseline_ms * 100.0


def by_workload(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("workload_id")), []).append(row)
    return grouped


def selected_workload_ids(policy: dict[str, Any] | None) -> list[str]:
    selected = []
    for label in (policy or {}).get("selected_workloads") or []:
        selected.append(str(label).split(":", 1)[0])
    return sorted(set(selected))


def selected_lane_aggregate(summary: dict[str, Any] | None, selected_ids: list[str]) -> dict[str, Any]:
    rows = measured_records(summary)
    grouped = by_workload(rows)
    baseline_total = 0.0
    candidate_total = 0.0
    accepted = 0
    attempted = 0
    peak = 0.0
    details = []
    for workload_id in selected_ids:
        workload_rows = grouped.get(workload_id, [])
        if not workload_rows:
            continue
        baseline = median(float((row.get("baseline") or {}).get("decode_ms") or 0.0) for row in workload_rows)
        candidate = median(float((row.get("mtp") or {}).get("decode_phase_ms") or 0.0) for row in workload_rows)
        row_accepted = sum(int((row.get("mtp") or {}).get("accepted_draft_tokens") or 0) for row in workload_rows)
        row_attempted = sum(int((row.get("mtp") or {}).get("attempted_draft_tokens") or 0) for row in workload_rows)
        row_peak = max(float((row.get("mtp") or {}).get("peak_memory_gb") or 0.0) for row in workload_rows)
        baseline_total += baseline
        candidate_total += candidate
        accepted += row_accepted
        attempted += row_attempted
        peak = max(peak, row_peak)
        details.append(
            {
                "workload_id": workload_id,
                "baseline_decode_ms": baseline,
                "candidate_decode_phase_ms": candidate,
                "speedup_percent": speedup_percent(baseline, candidate),
                "accepted_draft_tokens": row_accepted,
                "attempted_draft_tokens": row_attempted,
                "weighted_acceptance_rate": row_accepted / row_attempted if row_attempted else 0.0,
                "peak_memory_gb": row_peak,
            }
        )
    return {
        "workload_count": len(details),
        "workloads": details,
        "total_baseline_decode_ms": baseline_total,
        "total_selected_decode_phase_ms": candidate_total,
        "aggregate_speedup_percent": speedup_percent(baseline_total, candidate_total),
        "accepted_draft_tokens": accepted,
        "attempted_draft_tokens": attempted,
        "weighted_acceptance_rate": accepted / attempted if attempted else 0.0,
        "max_peak_memory_gb": peak,
    }


def bypass_status(summary: dict[str, Any] | None, workload_ids: list[str]) -> dict[str, Any]:
    rows = measured_records(summary)
    issues = []
    details = []
    grouped = by_workload(rows)
    for workload_id in workload_ids:
        workload_rows = grouped.get(workload_id, [])
        if not workload_rows:
            issues.append(f"missing workload {workload_id}")
            continue
        attempted = sum(int((row.get("mtp") or {}).get("attempted_draft_tokens") or 0) for row in workload_rows)
        exact = all((row.get("comparison") or {}).get("byte_identical") for row in workload_rows)
        auto_disabled = all(bool((row.get("mtp") or {}).get("auto_disabled")) for row in workload_rows)
        if attempted != 0:
            issues.append(f"{workload_id} attempted {attempted} MTP draft tokens")
        if not exact:
            issues.append(f"{workload_id} exactness failed")
        if not auto_disabled:
            issues.append(f"{workload_id} was not explicitly bypassed/auto-disabled")
        details.append(
            {
                "workload_id": workload_id,
                "records": len(workload_rows),
                "attempted_draft_tokens": attempted,
                "exact": exact,
                "auto_disabled": auto_disabled,
                "reasons": sorted(
                    {
                        str((row.get("mtp") or {}).get("auto_disable_reason") or "")
                        for row in workload_rows
                    }
                ),
            }
        )
    return {"passed": not issues, "issues": issues, "details": details}


def default_overhead(summary: dict[str, Any] | None) -> dict[str, Any]:
    rows = measured_records(summary)
    issues = []
    baseline_total = 0.0
    disabled_total = 0.0
    for row in rows:
        mtp = row.get("mtp") or {}
        baseline = row.get("baseline") or {}
        baseline_decode = float(baseline.get("decode_ms") or 0.0)
        disabled_decode = float(mtp.get("decode_phase_ms") or 0.0)
        baseline_total += baseline_decode
        disabled_total += disabled_decode
        if not (row.get("comparison") or {}).get("byte_identical"):
            issues.append(f"{record_key(row)} generated tokens differed")
        for field in (
            "attempted_draft_tokens",
            "accepted_draft_tokens",
            "target_verify_passes",
            "rollback_count",
        ):
            if int(mtp.get(field) or 0) != 0:
                issues.append(f"{record_key(row)} {field}={mtp.get(field)}")
        for field in (
            "drafter_load_ms",
            "draft_ms",
            "verify_ms",
            "verify_stage_ms",
            "verify_forward_ms",
            "verify_repair_ms",
            "repair_clone_ms",
            "repair_forward_ms",
            "repair_fallback_ms",
        ):
            if abs(float(mtp.get(field) or 0.0)) > 0.001:
                issues.append(f"{record_key(row)} {field}={mtp.get(field)}")
        if mtp.get("events"):
            issues.append(f"{record_key(row)} emitted MTP events")
        if abs(disabled_decode - baseline_decode) > 0.001:
            issues.append(
                f"{record_key(row)} decode_phase {disabled_decode:.3f} != baseline {baseline_decode:.3f}"
            )
    blockers = list((summary or {}).get("blockers") or [])
    issues.extend(f"default-overhead blocker: {blocker}" for blocker in blockers)
    return {
        "present": summary is not None,
        "records": len(rows),
        "passed": bool(rows) and not issues,
        "issues": issues,
        "baseline_decode_ms": baseline_total,
        "disabled_decode_phase_ms": disabled_total,
        "overhead_percent": -speedup_percent(baseline_total, disabled_total),
        "decision": (summary or {}).get("decision"),
        "mtp_disabled": bool((summary or {}).get("mtp_disabled")),
    }


def run_overview(label: str, summary: dict[str, Any] | None) -> dict[str, Any]:
    policy = policy_summary(summary)
    return {
        "label": label,
        "present": summary is not None,
        "decision": (summary or {}).get("decision"),
        "status": (summary or {}).get("status"),
        "run_id": (summary or {}).get("run_id"),
        "git_sha": (summary or {}).get("git_sha"),
        "git_status_short": (summary or {}).get("git_status_short"),
        "summary_path": (summary or {}).get("summary_path"),
        "records_path": (summary or {}).get("records_path"),
        "blockers": list((summary or {}).get("blockers") or []),
        "exactness": exactness(summary),
        "adaptive_policy_enabled": bool((summary or {}).get("adaptive_policy_enabled")),
        "mtp_real_margins_enabled": bool((summary or {}).get("mtp_real_margins_enabled")),
        "mtp_disabled": bool((summary or {}).get("mtp_disabled")),
        "policy": policy,
    }


def native_warmup_context(
    summary: dict[str, Any] | None,
    *,
    summary_path: str | None,
    report_path: str | None,
    label: str,
) -> dict[str, Any]:
    if summary is None:
        return {
            "present": False,
            "label": label,
            "summary_path": summary_path,
            "report_path": report_path,
            "reason": "native warmup summary not provided",
        }

    comparisons = []
    for comparison in summary.get("comparisons") or []:
        if comparison.get("candidate_variant") != WARMUP_AMORTIZED_VARIANT:
            continue
        comparisons.append(
            {
                "workload_id": comparison.get("workload_id"),
                "candidate_variant": comparison.get("candidate_variant"),
                "baseline_variant": comparison.get("baseline_variant"),
                "baseline_tail_reproduced": bool(comparison.get("baseline_tail_reproduced")),
                "correctness_passed": bool(comparison.get("correctness_passed")),
                "candidate_trials": int(comparison.get("candidate_trials") or 0),
                "baseline_trials": int(comparison.get("baseline_trials") or 0),
                "raw_p50_regression_percent": float(
                    comparison.get("raw_p50_regression_percent") or 0.0
                ),
                "raw_p95_improvement_percent": float(
                    comparison.get("raw_p95_improvement_percent") or 0.0
                ),
                "raw_p99_improvement_percent": float(
                    comparison.get("raw_p99_improvement_percent") or 0.0
                ),
                "steady_p50_regression_percent": float(
                    comparison.get("steady_p50_regression_percent") or 0.0
                ),
                "peak_mlx_delta_percent": float(comparison.get("peak_mlx_delta_percent") or 0.0),
                "memory_gate_passed": bool(comparison.get("memory_gate_passed")),
                "accepted": bool(comparison.get("accepted")),
                "reason": comparison.get("reason"),
            }
        )

    costs = []
    for cost in summary.get("warmup_cost_aggregates") or []:
        if cost.get("variant") != WARMUP_AMORTIZED_VARIANT:
            continue
        total = cost.get("total_ms") or {}
        amortized_total = cost.get("amortized_total_ms") or {}
        costs.append(
            {
                "workload_id": cost.get("workload_id"),
                "warmup_event_count": int(cost.get("warmup_event_count") or 0),
                "measured_request_count": int(cost.get("measured_request_count") or 0),
                "context_tokens_p50": float((cost.get("context_tokens") or {}).get("p50") or 0.0),
                "warmup_total_p50_ms": float(total.get("p50_ms") or 0.0),
                "warmup_total_p95_ms": float(total.get("p95_ms") or 0.0),
                "amortized_total_p50_ms": float(amortized_total.get("p50_ms") or 0.0),
                "amortized_total_p95_ms": float(amortized_total.get("p95_ms") or 0.0),
            }
        )

    first_tokens = []
    for row in summary.get("first_token_aggregates") or []:
        if row.get("variant") not in ("native_decode_runtime_default", WARMUP_AMORTIZED_VARIANT):
            continue
        latency = row.get("latency_ms") or {}
        first_tokens.append(
            {
                "variant": row.get("variant"),
                "workload_id": row.get("workload_id"),
                "sample_count": int(row.get("sample_count") or 0),
                "p50_ms": float(latency.get("p50_ms") or 0.0),
                "p95_ms": float(latency.get("p95_ms") or 0.0),
                "p99_ms": float(latency.get("p99_ms") or 0.0),
                "max_ms": float(latency.get("max_ms") or 0.0),
            }
        )

    return {
        "present": True,
        "label": label,
        "decision": summary.get("decision"),
        "status": summary.get("status"),
        "run_id": summary.get("run_id"),
        "git_sha": summary.get("git_sha"),
        "git_status_short": summary.get("git_status_short"),
        "summary_path": summary_path or summary.get("summary_path"),
        "report_path": report_path or summary.get("report_path"),
        "record_count": summary.get("record_count"),
        "passed_records": summary.get("passed_records"),
        "failed_records": summary.get("failed_records"),
        "blockers": list(summary.get("blockers") or []),
        "comparisons": comparisons,
        "warmup_costs": costs,
        "first_token_aggregates": first_tokens,
        "claim_boundaries": [
            "Native warmup evidence is out-of-request/load-time shape work only.",
            "XR78 does not prove a request-path warmup policy.",
            "Broad MTP default-on still depends on protected aggregate gates.",
        ],
    }


def server_prefix_warm_context(
    summary: dict[str, Any] | None,
    *,
    summary_path: str | None,
    report_path: str | None,
    label: str,
) -> dict[str, Any]:
    if summary is None:
        return {
            "present": False,
            "label": label,
            "summary_path": summary_path,
            "report_path": report_path,
            "reason": "server prefix warm summary not provided",
        }

    warmups = []
    for warmup in summary.get("candidate_warmups") or []:
        warmups.append(
            {
                "workload_id": warmup.get("workload_id"),
                "status": warmup.get("status"),
                "requested_prefix_tokens": int(warmup.get("requested_prefix_tokens") or 0),
                "prompt_tokens": int(warmup.get("prompt_tokens") or 0),
                "warmup_context_tokens": int(warmup.get("warmup_context_tokens") or 0),
                "tokenize_ms": float(warmup.get("tokenize_ms") or 0.0),
                "prefill_ms": float(warmup.get("prefill_ms") or 0.0),
                "decode_ms": float(warmup.get("decode_ms") or 0.0),
                "total_ms": float(warmup.get("total_ms") or 0.0),
                "peak_memory_gb": float(warmup.get("peak_memory_gb") or 0.0),
                "active_kv_bytes": int(warmup.get("active_kv_bytes") or 0),
            }
        )

    request_deltas = []
    for row in summary.get("records") or []:
        baseline = row.get("baseline") or {}
        candidate = row.get("candidate") or {}
        baseline_metrics = baseline.get("metrics") or {}
        candidate_metrics = candidate.get("metrics") or {}
        baseline_first = (baseline_metrics.get("decode_token_latencies_ms") or [None])[0]
        candidate_first = (candidate_metrics.get("decode_token_latencies_ms") or [None])[0]
        request_deltas.append(
            {
                "workload_id": row.get("workload_id"),
                "repeat_index": int(row.get("repeat_index") or 0),
                "status": row.get("comparison_status"),
                "baseline_first_token_ms": float(baseline_first or 0.0),
                "candidate_first_token_ms": float(candidate_first or 0.0),
                "first_token_delta_ms": float(baseline_first or 0.0) - float(candidate_first or 0.0),
                "baseline_total_ms": float(baseline_metrics.get("total_ms") or 0.0),
                "candidate_total_ms": float(candidate_metrics.get("total_ms") or 0.0),
                "baseline_request_wall_ms": float(baseline.get("request_wall_ms") or 0.0),
                "candidate_request_wall_ms": float(candidate.get("request_wall_ms") or 0.0),
            }
        )

    final_metrics = summary.get("final_metrics") or {}
    candidate_metrics = final_metrics.get("candidate") or {}
    return {
        "present": True,
        "label": label,
        "decision": summary.get("decision"),
        "status": summary.get("status"),
        "run_id": summary.get("run_id"),
        "mode": summary.get("mode"),
        "summary_path": summary_path or summary.get("summary_path"),
        "report_path": report_path or summary.get("report_path"),
        "blockers": list(summary.get("blockers") or []),
        "candidate_prefix_warmup_tokens": summary.get("candidate_prefix_warmup_tokens"),
        "warmups": warmups,
        "request_deltas": request_deltas,
        "final_metrics": {
            "prefix_warmups_total": float(candidate_metrics.get("prefix_warmups_total") or 0.0),
            "prefix_warmup_tokens_total": float(
                candidate_metrics.get("prefix_warmup_tokens_total") or 0.0
            ),
            "prefix_warmup_seconds": float(candidate_metrics.get("prefix_warmup_seconds") or 0.0),
            "memory_peak_mlx_bytes": float(candidate_metrics.get("memory_peak_mlx_bytes") or 0.0),
        },
        "claim_boundaries": [
            "XR85 server prefix warmup is explicit local control-surface work only.",
            "XR85 does not make prefix warmup automatic or default-on.",
            "Broad MTP default-on still depends on protected aggregate gates.",
        ],
    }


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    candidate_summary = load_json(Path(args.candidate_summary))
    oracle_summary = load_json(Path(args.oracle_summary))
    default_summary = load_json(Path(args.default_overhead_summary))
    holdout_summary = load_json(Path(args.holdout_summary), required=False) if args.holdout_summary else None
    native_warmup_summary = (
        load_json(Path(args.native_warmup_summary), required=False)
        if args.native_warmup_summary
        else None
    )
    server_prefix_summary = (
        load_json(Path(args.server_prefix_warm_summary), required=False)
        if args.server_prefix_warm_summary
        else None
    )

    candidate_policy = policy_summary(candidate_summary)
    selected_ids = selected_workload_ids(candidate_policy)
    selected_lane = selected_lane_aggregate(candidate_summary, selected_ids)
    protected = bypass_status(candidate_summary, list(args.protected_workload))
    holdout = bypass_status(holdout_summary, list(args.holdout_workload)) if holdout_summary else {
        "passed": False,
        "issues": ["holdout summary missing"],
        "details": [],
    }
    oracle = compare_oracle(candidate_summary, oracle_summary)
    overhead = default_overhead(default_summary)
    candidate = run_overview("candidate", candidate_summary)

    protected_speed = float((candidate_policy or {}).get("aggregate_speedup_percent") or 0.0)
    peak_memory = float((candidate_policy or {}).get("max_peak_memory_gb") or 0.0)
    weighted_acceptance = float((candidate_policy or {}).get("weighted_acceptance_rate") or 0.0)

    gates = {
        "candidate_exactness": candidate["exactness"]["passed"],
        "candidate_no_blockers": not candidate["blockers"],
        "selected_lane_present": selected_lane["workload_count"] > 0,
        "selected_lane_speed_positive": selected_lane["aggregate_speedup_percent"] > 0.0,
        "protected_aggregate_reported": candidate_policy is not None,
        "protected_holdout_bypassed": protected["passed"],
        "four_k_holdouts_bypassed": holdout["passed"],
        "oracle_passed": oracle["passed"],
        "default_overhead_clean": overhead["passed"] and overhead["mtp_disabled"],
        "memory_under_tiny16": 0.0 < peak_memory <= args.memory_cliff_gb,
    }

    broad_default_gates = dict(gates)
    broad_default_gates["protected_speed_ge_gate"] = protected_speed >= args.broad_default_gate_percent
    broad_default_supported = all(broad_default_gates.values())

    blockers: list[str] = []
    if not candidate["present"]:
        blockers.append("candidate summary missing")
    if candidate["blockers"]:
        blockers.extend(f"candidate blocker: {blocker}" for blocker in candidate["blockers"])
    if not candidate["exactness"]["passed"]:
        blockers.append("candidate exactness failed")
    if not oracle["passed"]:
        blockers.append("sequential oracle mismatch or missing records")
    if not overhead["passed"]:
        blockers.extend(f"default overhead: {issue}" for issue in overhead["issues"])
    if not protected["passed"]:
        blockers.extend(f"protected holdout: {issue}" for issue in protected["issues"])
    if holdout_summary is not None and not holdout["passed"]:
        blockers.extend(f"4K holdout: {issue}" for issue in holdout["issues"])

    scoped_gates_passed = all(gates.values())
    if blockers:
        decision = "blocked_with_evidence"
    elif selected_lane["workload_count"] == 0 or selected_lane["aggregate_speedup_percent"] <= 0.0:
        decision = "reject_candidate"
    elif scoped_gates_passed:
        decision = "accept_candidate"
    else:
        decision = "needs_more_data"

    return {
        "schema_version": 1,
        "goal": args.goal,
        "title": args.title,
        "decision": decision,
        "scoped_gates_passed": scoped_gates_passed,
        "broad_default_supported": broad_default_supported,
        "broad_default_gate_percent": args.broad_default_gate_percent,
        "memory_cliff_gb": args.memory_cliff_gb,
        "candidate": candidate,
        "oracle_run": run_overview("oracle", oracle_summary),
        "default_overhead_run": run_overview("default_overhead", default_summary),
        "holdout_run": run_overview("holdout", holdout_summary),
        "candidate_policy": candidate_policy,
        "protected_aggregate": {
            "aggregate_speedup_percent": protected_speed,
            "weighted_acceptance_rate": weighted_acceptance,
            "max_peak_memory_gb": peak_memory,
            "selected_workloads": list((candidate_policy or {}).get("selected_workloads") or []),
            "total_baseline_decode_ms": float((candidate_policy or {}).get("total_baseline_decode_ms") or 0.0),
            "total_selected_decode_phase_ms": float(
                (candidate_policy or {}).get("total_selected_decode_phase_ms") or 0.0
            ),
        },
        "selected_lane_aggregate": selected_lane,
        "protected_holdout": protected,
        "four_k_holdouts": holdout,
        "oracle": oracle,
        "default_overhead": overhead,
        "native_warmup_context": native_warmup_context(
            native_warmup_summary,
            summary_path=args.native_warmup_summary,
            report_path=args.native_warmup_report,
            label=args.native_warmup_label,
        ),
        "server_prefix_warm_context": server_prefix_warm_context(
            server_prefix_summary,
            summary_path=args.server_prefix_warm_summary,
            report_path=args.server_prefix_warm_report,
            label=args.server_prefix_warm_label,
        ),
        "scoped_gates": gates,
        "broad_default_gates": broad_default_gates,
        "blockers": blockers,
        "inputs": {
            "candidate_summary": args.candidate_summary,
            "oracle_summary": args.oracle_summary,
            "default_overhead_summary": args.default_overhead_summary,
            "holdout_summary": args.holdout_summary,
            "native_warmup_summary": args.native_warmup_summary,
            "native_warmup_report": args.native_warmup_report,
            "server_prefix_warm_summary": args.server_prefix_warm_summary,
            "server_prefix_warm_report": args.server_prefix_warm_report,
        },
    }


def fmt(value: float) -> str:
    return f"{value:.3f}"


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['title']}",
        "",
        f"- Decision: `{result['decision']}`",
        f"- Scoped gates passed: `{result['scoped_gates_passed']}`",
        f"- Broad default supported: `{result['broad_default_supported']}`",
        "",
        "## Scoped Gate Checklist",
        "",
        "| Gate | Passed |",
        "|---|---|",
    ]
    for gate, passed in result["scoped_gates"].items():
        lines.append(f"| `{gate}` | `{passed}` |")
    lines.extend(
        [
            "",
            "## Protected Aggregate",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Speedup % | {fmt(result['protected_aggregate']['aggregate_speedup_percent'])} |",
            f"| Weighted acceptance | {fmt(result['protected_aggregate']['weighted_acceptance_rate'])} |",
            f"| Peak GB | {fmt(result['protected_aggregate']['max_peak_memory_gb'])} |",
            f"| Baseline decode ms | {fmt(result['protected_aggregate']['total_baseline_decode_ms'])} |",
            f"| Selected decode ms | {fmt(result['protected_aggregate']['total_selected_decode_phase_ms'])} |",
            "",
            "Selected workloads: `"
            + (", ".join(result["protected_aggregate"]["selected_workloads"]) or "none")
            + "`",
            "",
            "## Selected Chat/Tool Lane",
            "",
            "| Workload | Speedup % | Accepted/Attempted | Acceptance | Peak GB |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in result["selected_lane_aggregate"]["workloads"]:
        lines.append(
            "| `{workload_id}` | {speedup} | {accepted}/{attempted} | {acceptance} | {peak} |".format(
                workload_id=row["workload_id"],
                speedup=fmt(row["speedup_percent"]),
                accepted=row["accepted_draft_tokens"],
                attempted=row["attempted_draft_tokens"],
                acceptance=fmt(row["weighted_acceptance_rate"]),
                peak=fmt(row["peak_memory_gb"]),
            )
        )
    lines.extend(
        [
            "",
            f"Selected-lane aggregate speedup: `{fmt(result['selected_lane_aggregate']['aggregate_speedup_percent'])}%`",
            f"Selected-lane weighted acceptance: `{fmt(result['selected_lane_aggregate']['weighted_acceptance_rate'])}`",
        ]
    )
    native_warmup = result.get("native_warmup_context") or {}
    if native_warmup.get("present"):
        lines.extend(
            [
                "",
                "## Native Tail / Warmup Context",
                "",
                f"- Label: `{native_warmup['label']}`",
                f"- Decision: `{native_warmup.get('decision')}`",
                f"- Status: `{native_warmup.get('status')}`",
                f"- Records: `{native_warmup.get('passed_records')}/{native_warmup.get('record_count')}`",
                f"- Summary: `{native_warmup.get('summary_path')}`",
                f"- Report: `{native_warmup.get('report_path')}`",
                "",
                "| Workload | Accepted | Baseline tail | p50 regression % | p95 improvement % | p99 improvement % | Reason |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for comparison in native_warmup["comparisons"]:
            lines.append(
                "| `{workload_id}` | `{accepted}` | `{baseline_tail}` | {p50} | {p95} | {p99} | {reason} |".format(
                    workload_id=comparison["workload_id"],
                    accepted=comparison["accepted"],
                    baseline_tail=comparison["baseline_tail_reproduced"],
                    p50=fmt(comparison["raw_p50_regression_percent"]),
                    p95=fmt(comparison["raw_p95_improvement_percent"]),
                    p99=fmt(comparison["raw_p99_improvement_percent"]),
                    reason=comparison["reason"],
                )
            )
        lines.extend(
            [
                "",
                "| Workload | Warmup events | Measured requests | Context tokens p50 | Warmup total p50 ms | Amortized total p50 ms |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for cost in native_warmup["warmup_costs"]:
            lines.append(
                "| `{workload_id}` | {events} | {requests} | {context} | {warmup} | {amortized} |".format(
                    workload_id=cost["workload_id"],
                    events=cost["warmup_event_count"],
                    requests=cost["measured_request_count"],
                    context=fmt(cost["context_tokens_p50"]),
                    warmup=fmt(cost["warmup_total_p50_ms"]),
                    amortized=fmt(cost["amortized_total_p50_ms"]),
                )
            )
        lines.extend(["", "Claim boundaries:"])
        lines.extend(f"- {boundary}" for boundary in native_warmup["claim_boundaries"])
    server_prefix = result.get("server_prefix_warm_context") or {}
    if server_prefix.get("present"):
        lines.extend(
            [
                "",
                "## Server Prefix-Warm Context",
                "",
                f"- Label: `{server_prefix['label']}`",
                f"- Decision: `{server_prefix.get('decision')}`",
                f"- Status: `{server_prefix.get('status')}`",
                f"- Mode: `{server_prefix.get('mode')}`",
                f"- Summary: `{server_prefix.get('summary_path')}`",
                f"- Report: `{server_prefix.get('report_path')}`",
                f"- Prefix warmups total: `{fmt(server_prefix['final_metrics']['prefix_warmups_total'])}`",
                f"- Prefix warm tokens total: `{fmt(server_prefix['final_metrics']['prefix_warmup_tokens_total'])}`",
                f"- Prefix warm seconds: `{fmt(server_prefix['final_metrics']['prefix_warmup_seconds'])}`",
                "",
                "| Workload | Status | Prefix | Prompt | Warm ctx | Total ms | Prefill ms | Decode ms | Peak GB |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for warmup in server_prefix["warmups"]:
            lines.append(
                "| `{workload_id}` | `{status}` | {prefix} | {prompt} | {warm} | {total} | {prefill} | {decode} | {peak} |".format(
                    workload_id=warmup["workload_id"],
                    status=warmup["status"],
                    prefix=warmup["requested_prefix_tokens"],
                    prompt=warmup["prompt_tokens"],
                    warm=warmup["warmup_context_tokens"],
                    total=fmt(warmup["total_ms"]),
                    prefill=fmt(warmup["prefill_ms"]),
                    decode=fmt(warmup["decode_ms"]),
                    peak=fmt(warmup["peak_memory_gb"]),
                )
            )
        lines.extend(
            [
                "",
                "| Workload | Repeat | Status | Baseline first ms | Candidate first ms | Delta ms | Baseline total ms | Candidate total ms |",
                "|---|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for delta in server_prefix["request_deltas"]:
            lines.append(
                "| `{workload_id}` | {repeat} | `{status}` | {baseline_first} | {candidate_first} | {delta_ms} | {baseline_total} | {candidate_total} |".format(
                    workload_id=delta["workload_id"],
                    repeat=delta["repeat_index"],
                    status=delta["status"],
                    baseline_first=fmt(delta["baseline_first_token_ms"]),
                    candidate_first=fmt(delta["candidate_first_token_ms"]),
                    delta_ms=fmt(delta["first_token_delta_ms"]),
                    baseline_total=fmt(delta["baseline_total_ms"]),
                    candidate_total=fmt(delta["candidate_total_ms"]),
                )
            )
        lines.extend(["", "Claim boundaries:"])
        lines.extend(f"- {boundary}" for boundary in server_prefix["claim_boundaries"])
    lines.extend(
        [
            "",
            "## Default-Overhead Probe",
            "",
            f"- Passed: `{result['default_overhead']['passed']}`",
            f"- Decision: `{result['default_overhead']['decision']}`",
            f"- Records: `{result['default_overhead']['records']}`",
            f"- Overhead percent: `{fmt(result['default_overhead']['overhead_percent'])}%`",
            "",
            "## Oracle And Holdouts",
            "",
            f"- Sequential oracle passed: `{result['oracle']['passed']}`; compared records: `{result['oracle']['compared_records']}`",
            f"- Protected holdout bypass passed: `{result['protected_holdout']['passed']}`",
            f"- 4K holdout bypass passed: `{result['four_k_holdouts']['passed']}`",
            "",
            "## Broad Default Gate",
            "",
            "| Gate | Passed |",
            "|---|---|",
        ]
    )
    for gate, passed in result["broad_default_gates"].items():
        lines.append(f"| `{gate}` | `{passed}` |")
    if result["blockers"]:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in result["blockers"])
    lines.extend(
        [
            "",
            "## Inputs",
            "",
        ]
    )
    for key, value in result["inputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--oracle-summary", required=True)
    parser.add_argument("--default-overhead-summary", required=True)
    parser.add_argument("--holdout-summary")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-md", default="xr73-scoped-mtp-summary.md")
    parser.add_argument("--out-json", default="xr73-scoped-mtp-summary.json")
    parser.add_argument("--title", default="XR73 Scoped MTP Chat/Tool Opt-in")
    parser.add_argument("--goal", default="XR73-scoped-mtp-chat-tool-opt-in")
    parser.add_argument("--native-warmup-summary")
    parser.add_argument("--native-warmup-report")
    parser.add_argument("--native-warmup-label", default="XR78 native amortized warmup")
    parser.add_argument("--server-prefix-warm-summary")
    parser.add_argument("--server-prefix-warm-report")
    parser.add_argument("--server-prefix-warm-label", default="XR85 server prefix warmup")
    parser.add_argument("--protected-workload", action="append", default=list(DEFAULT_PROTECTED_WORKLOADS))
    parser.add_argument(
        "--holdout-workload",
        action="append",
        default=["benchmark_qa_4k_001", "code_review_rust_4k_001", "mtp_candidate_4k_001"],
    )
    parser.add_argument("--broad-default-gate-percent", type=float, default=25.0)
    parser.add_argument("--memory-cliff-gb", type=float, default=14.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build_result(args)
    (out_dir / args.out_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / args.out_md).write_text(render_markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
