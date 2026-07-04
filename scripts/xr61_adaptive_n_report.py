#!/usr/bin/env python3
"""Render a gate-aware XR61 Adaptive-N MTP decision report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_POLICY_CANDIDATES = (
    "benchmarks/out/XR61-adaptive-n-mtp/policy-search/policy_candidates.json"
)
DEFAULT_OUT_MD = "benchmarks/out/XR61-adaptive-n-mtp/xr61-adaptive-n-summary.md"
DEFAULT_OUT_JSON = "benchmarks/out/XR61-adaptive-n-mtp/xr61-adaptive-n-summary.json"
ADAPTIVE_POLICY_PREFIX = "adaptive_policy_"
DEFAULT_ON_SPEED_GATE_PERCENT = 25.0


def load_json(path: Path, *, required: bool = True) -> dict[str, Any] | None:
    if not path.exists():
        if required:
            raise SystemExit(f"{path}: JSON file does not exist")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: invalid JSON: {exc}") from exc


def load_optional_summary(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    return load_json(Path(path), required=False)


def records(summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not summary:
        return []
    raw = summary.get("records") or []
    return raw if isinstance(raw, list) else []


def measured_records(summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [record for record in records(summary) if record.get("measured")]


def record_key(record: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(record.get("workload_id")),
        str(record.get("trial_kind")),
        int(record.get("trial_index") or 0),
        int(record.get("block_size") or 0),
    )


def generated_tokens(record: dict[str, Any]) -> list[int]:
    raw = (record.get("mtp") or {}).get("generated_tokens") or []
    return [int(token) for token in raw]


def adaptive_policy_summary(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not summary:
        return None
    source_policy = str(summary.get("source_policy_name") or "")
    policies = summary.get("policy_summaries") or []
    if source_policy:
        for policy in policies:
            if policy.get("policy_name") == source_policy:
                return policy
    for policy in policies:
        if str(policy.get("policy_name") or "").startswith(ADAPTIVE_POLICY_PREFIX):
            return policy
    return None


def provenance_complete(summary: dict[str, Any] | None) -> bool:
    rows = records(summary)
    if not rows:
        return False
    for record in rows:
        provenance = record.get("build_provenance") or {}
        if not provenance.get("git_sha"):
            return False
        if not provenance.get("dirty_diff_sha256"):
            return False
        if not provenance.get("runner_binary_path"):
            return False
        if not provenance.get("gemma4d_env"):
            return False
    return True


def exactness_summary(summary: dict[str, Any] | None) -> dict[str, Any]:
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


def run_summary(label: str, summary: dict[str, Any] | None) -> dict[str, Any]:
    policy = adaptive_policy_summary(summary)
    exactness = exactness_summary(summary)
    blockers = list((summary or {}).get("blockers") or [])
    return {
        "label": label,
        "present": summary is not None,
        "decision": (summary or {}).get("decision"),
        "status": (summary or {}).get("status"),
        "run_id": (summary or {}).get("run_id"),
        "git_sha": (summary or {}).get("git_sha"),
        "git_status_short": (summary or {}).get("git_status_short"),
        "records_path": (summary or {}).get("records_path"),
        "summary_path": (summary or {}).get("summary_path"),
        "adaptive_policy_enabled": bool((summary or {}).get("adaptive_policy_enabled")),
        "mtp_real_margins_enabled": bool((summary or {}).get("mtp_real_margins_enabled")),
        "blockers": blockers,
        "exactness": exactness,
        "provenance_complete": provenance_complete(summary),
        "policy": policy,
        "policy_speedup_percent": float((policy or {}).get("aggregate_speedup_percent") or 0.0),
        "policy_selected_mtp_workloads": int((policy or {}).get("selected_mtp_workloads") or 0),
        "policy_weighted_acceptance_rate": float((policy or {}).get("weighted_acceptance_rate") or 0.0),
        "policy_peak_memory_gb": float((policy or {}).get("max_peak_memory_gb") or 0.0),
        "policy_selected_workloads": list((policy or {}).get("selected_workloads") or []),
        "policy_regressed_workloads": list((policy or {}).get("regressed_workload_ids") or []),
    }


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


def holdout_gate(holdout: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    if not holdout["present"]:
        issues.append("holdout summary missing")
    if holdout["blockers"]:
        issues.extend(f"holdout blocker: {blocker}" for blocker in holdout["blockers"])
    if not holdout["exactness"]["passed"]:
        issues.append(
            "holdout exactness failed: "
            f"{holdout['exactness']['exact_measured_records']}/"
            f"{holdout['exactness']['measured_records']} measured"
        )
    if holdout["policy_regressed_workloads"]:
        issues.append(
            "holdout selected-policy regressions: "
            + ", ".join(holdout["policy_regressed_workloads"])
        )
    return {
        "passed": not issues,
        "issues": issues,
        "selected_workloads": holdout["policy_selected_workloads"],
        "selected_mtp_workloads": holdout["policy_selected_mtp_workloads"],
        "aggregate_speedup_percent": holdout["policy_speedup_percent"],
    }


def default_on_gates(
    candidate: dict[str, Any],
    holdout: dict[str, Any],
    oracle: dict[str, Any],
) -> dict[str, Any]:
    holdout_result = holdout_gate(holdout)
    speed = candidate["policy_speedup_percent"]
    memory = candidate["policy_peak_memory_gb"]
    gates = {
        "candidate_exactness": candidate["exactness"]["passed"],
        "oracle_differential": oracle["passed"],
        "aggregate_speed_ge_25pct": speed >= DEFAULT_ON_SPEED_GATE_PERCENT,
        "holdout_protection": holdout_result["passed"],
        "memory_under_tiny16_cliff": 0.0 < memory <= 14.0,
        "default_overhead_measured": False,
        "real_margins_env_gated": candidate["mtp_real_margins_enabled"],
        "provenance_complete": candidate["provenance_complete"]
        and holdout["provenance_complete"],
        "ledger_updated": False,
        "risk_review_recorded": False,
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "holdout": holdout_result,
        "candidate_speedup_percent": speed,
        "candidate_peak_memory_gb": memory,
    }


def decide(
    policy_search: dict[str, Any],
    candidate_summary: dict[str, Any] | None,
    holdout_summary: dict[str, Any] | None,
    oracle_summary: dict[str, Any] | None,
) -> tuple[str, str, list[str], dict[str, Any]]:
    blockers = list(policy_search.get("blockers") or [])
    candidate = run_summary("candidate", candidate_summary)
    holdout = run_summary("holdout", holdout_summary)
    oracle = compare_oracle(candidate_summary, oracle_summary)
    gates = default_on_gates(candidate, holdout, oracle)

    if candidate_summary is None:
        blockers.append("candidate summary missing")
        return "needs_more_data", "Run the XR61 adaptive candidate benchmark.", blockers, gates
    if candidate["blockers"] or not candidate["exactness"]["passed"]:
        blockers.extend(candidate["blockers"])
        return (
            "blocked_with_evidence",
            "Fix or reject the adaptive candidate because primary exactness failed.",
            blockers,
            gates,
        )
    if holdout_summary is None:
        blockers.append("holdout summary missing")
        return (
            "needs_more_data",
            "Run the XR61 adaptive holdout before making the final P1 decision.",
            blockers,
            gates,
        )
    if not holdout_gate(holdout)["passed"]:
        blockers.extend(holdout_gate(holdout)["issues"])
        return (
            "blocked_with_evidence",
            "Treat the adaptive policy as blocked/rejected until holdout exactness and protection pass.",
            blockers,
            gates,
        )
    if oracle_summary is None:
        blockers.append("oracle summary missing")
        return (
            "needs_more_data",
            "Run the sequential-oracle adaptive-N differential.",
            blockers,
            gates,
        )
    if not oracle["passed"]:
        blockers.extend(f"oracle mismatch: {item}" for item in oracle["mismatches"])
        blockers.extend(f"oracle missing: {item}" for item in oracle["missing_records"])
        blockers.extend(f"oracle extra: {item}" for item in oracle["extra_records"])
        return (
            "blocked_with_evidence",
            "Fix or reject the adaptive policy because the sequential oracle did not match.",
            blockers,
            gates,
        )

    if candidate["policy_selected_mtp_workloads"] == 0 or candidate["policy_speedup_percent"] <= 0.0:
        return (
            "reject_candidate",
            "Adaptive policy selected no profitable primary MTP workloads.",
            blockers,
            gates,
        )
    if gates["passed"]:
        return (
            "accept_candidate",
            "All default-on gates passed; request risk review before any production/default wiring.",
            blockers,
            gates,
        )
    return (
        "keep_experimental",
        "Keep XR61 adaptive-N env-gated: exactness/protection can pass, but aggregate speed or default-on evidence is insufficient.",
        blockers,
        gates,
    )


def render_policy_table(policy_search: dict[str, Any]) -> list[str]:
    fixed = (policy_search.get("fixed_n") or {}).get("aggregate") or {}
    lines = [
        "## Fixed-N Aggregate",
        "",
        "| N | Exact | Speedup % | Acceptance | Peak GB |",
        "|---:|---|---:|---:|---:|",
    ]
    for block, row in fixed.items():
        lines.append(
            f"| {block} | `{row.get('exact')}` | "
            f"{float(row.get('speedup_percent') or 0.0):.3f} | "
            f"{float(row.get('acceptance_rate') or 0.0):.3f} | "
            f"{float(row.get('peak_memory_gb') or 0.0):.3f} |"
        )
    lines.append("")
    return lines


def render_run(label: str, run: dict[str, Any]) -> list[str]:
    lines = [f"## {label}", ""]
    if not run["present"]:
        lines.append("- Not provided.")
        lines.append("")
        return lines
    lines.extend(
        [
            f"- Decision: `{run['decision']}`",
            f"- Run ID: `{run['run_id']}`",
            f"- Git SHA: `{run['git_sha']}`",
            f"- Git status: `{run['git_status_short'] or 'clean'}`",
            f"- Exact measured: `{run['exactness']['exact_measured_records']}/"
            f"{run['exactness']['measured_records']}`",
            f"- Adaptive selections: `{run['policy_selected_workloads']}`",
            f"- Aggregate speedup: `{run['policy_speedup_percent']:.3f}%`",
            f"- Weighted acceptance: `{run['policy_weighted_acceptance_rate']:.3f}`",
            f"- Peak memory: `{run['policy_peak_memory_gb']:.3f} GB`",
        ]
    )
    if run["blockers"]:
        lines.append("- Blockers:")
        for blocker in run["blockers"]:
            lines.append(f"  - {blocker}")
    lines.append("")
    return lines


def render_markdown(result: dict[str, Any]) -> str:
    policy_search = result["policy_search"]
    guarded = policy_search.get("guarded_policy") or {}
    real_coverage = policy_search.get("real_margin_coverage") or {}
    candidate = result["candidate"]
    holdout = result["holdout"]
    oracle = result["oracle"]
    gates = result["default_on_gates"]

    lines: list[str] = []
    lines.append("# XR61 Adaptive-N MTP Summary")
    lines.append("")
    lines.append(f"- Decision: `{result['decision']}`")
    lines.append(f"- Policy-search hint: `{policy_search.get('decision_hint', 'unknown')}`")
    lines.append(f"- Next action: {result['next_action']}")
    lines.append("")
    lines.append("## Evidence Inputs")
    lines.append("")
    for label, value in result["inputs"].items():
        lines.append(f"- {label}: `{value or 'not provided'}`")
    lines.append("")
    lines.extend(render_policy_table(policy_search))
    lines.append("## Recomputed XR56 Guarded Comparator")
    lines.append("")
    lines.append(
        f"- Aggregate speedup: `{float(guarded.get('aggregate_speedup_percent') or 0.0):.3f}%`"
    )
    lines.append(
        f"- Weighted acceptance: `{float(guarded.get('weighted_acceptance_rate') or 0.0):.3f}`"
    )
    lines.append(f"- Peak memory: `{float(guarded.get('max_peak_memory_gb') or 0.0):.3f} GB`")
    lines.append("")
    lines.append("## Real-Margin Coverage")
    lines.append("")
    lines.append(f"- Measured records: `{real_coverage.get('measured_records', 0)}`")
    lines.append(f"- Workloads covered: `{real_coverage.get('workloads', [])}`")
    lines.append(f"- Blocks covered: `{real_coverage.get('blocks', [])}`")
    lines.append(
        f"- Sufficient for policy design: `{real_coverage.get('sufficient_for_policy_design', False)}`"
    )
    lines.append("")
    lines.extend(render_run("Candidate", candidate))
    lines.extend(render_run("Holdout", holdout))
    lines.append("## Sequential Oracle")
    lines.append("")
    if oracle["present"]:
        lines.append(f"- Passed: `{oracle['passed']}`")
        lines.append(f"- Compared records: `{oracle['compared_records']}`")
        if oracle["missing_records"]:
            lines.append(f"- Missing records: `{oracle['missing_records']}`")
        if oracle["extra_records"]:
            lines.append(f"- Extra records: `{oracle['extra_records']}`")
        if oracle["mismatches"]:
            lines.append(f"- Mismatches: `{oracle['mismatches']}`")
    else:
        lines.append("- Not provided.")
    lines.append("")
    lines.append("## Default-On Gates")
    lines.append("")
    lines.append("| Gate | Passed |")
    lines.append("|---|---:|")
    for name, passed in gates["gates"].items():
        lines.append(f"| `{name}` | `{passed}` |")
    lines.append("")
    lines.append("## Blockers / Missing Evidence")
    lines.append("")
    blockers = result.get("blockers") or []
    if blockers:
        for blocker in blockers:
            lines.append(f"- {blocker}")
    else:
        lines.append("No blockers recorded by the current XR61 summary inputs.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-candidates", default=DEFAULT_POLICY_CANDIDATES)
    parser.add_argument("--baseline-summary")
    parser.add_argument("--trace-summary")
    parser.add_argument("--candidate-summary")
    parser.add_argument("--holdout-summary")
    parser.add_argument("--oracle-summary")
    parser.add_argument("--out-md", default=DEFAULT_OUT_MD)
    parser.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    policy_search = load_json(Path(args.policy_candidates), required=True)
    baseline_summary = load_optional_summary(args.baseline_summary)
    trace_summary = load_optional_summary(args.trace_summary)
    candidate_summary = load_optional_summary(args.candidate_summary)
    holdout_summary = load_optional_summary(args.holdout_summary)
    oracle_summary = load_optional_summary(args.oracle_summary)
    decision, next_action, blockers, gates = decide(
        policy_search,
        candidate_summary,
        holdout_summary,
        oracle_summary,
    )

    result = {
        "schema_version": 2,
        "phase": "xr61_summary",
        "decision": decision,
        "inputs": {
            "policy_candidates": args.policy_candidates,
            "baseline_summary": args.baseline_summary,
            "trace_summary": args.trace_summary,
            "candidate_summary": args.candidate_summary,
            "holdout_summary": args.holdout_summary,
            "oracle_summary": args.oracle_summary,
        },
        "blockers": blockers,
        "next_action": next_action,
        "policy_search": policy_search,
        "baseline_summary_present": baseline_summary is not None,
        "trace_summary_present": trace_summary is not None,
        "candidate_summary_present": candidate_summary is not None,
        "holdout_summary_present": holdout_summary is not None,
        "oracle_summary_present": oracle_summary is not None,
        "candidate": run_summary("candidate", candidate_summary),
        "holdout": run_summary("holdout", holdout_summary),
        "oracle": compare_oracle(candidate_summary, oracle_summary),
        "default_on_gates": gates,
    }

    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    out_md.write_text(render_markdown(result), encoding="utf-8")

    print(f"XR61 summary decision: {decision}")
    print(f"summary_md: {out_md}")
    print(f"summary_json: {out_json}")


if __name__ == "__main__":
    main()
