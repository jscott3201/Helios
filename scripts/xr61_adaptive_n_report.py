#!/usr/bin/env python3
"""Render an XR61 summary from policy-search and benchmark artifacts."""

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


def render_markdown(result: dict[str, Any]) -> str:
    policy_search = result["policy_search"]
    guarded = policy_search.get("guarded_policy") or {}
    fixed = (policy_search.get("fixed_n") or {}).get("aggregate") or {}
    real_signal = policy_search.get("real_margin_summary") or {}
    real_coverage = policy_search.get("real_margin_coverage") or {}
    lines: list[str] = []
    lines.append("# XR61 Adaptive-N MTP Summary")
    lines.append("")
    lines.append(f"- Decision: `{result['decision']}`")
    lines.append(f"- Policy-search hint: `{policy_search.get('decision_hint', 'unknown')}`")
    lines.append("")
    lines.append("## Evidence Inputs")
    lines.append("")
    for label, value in result["inputs"].items():
        lines.append(f"- {label}: `{value or 'not provided'}`")
    lines.append("")
    lines.append("## Fixed-N Aggregate")
    lines.append("")
    lines.append("| N | Exact | Speedup % | Acceptance | Peak GB |")
    lines.append("|---:|---|---:|---:|---:|")
    for block, row in fixed.items():
        lines.append(
            f"| {block} | `{row.get('exact')}` | "
            f"{float(row.get('speedup_percent') or 0.0):.3f} | "
            f"{float(row.get('acceptance_rate') or 0.0):.3f} | "
            f"{float(row.get('peak_memory_gb') or 0.0):.3f} |"
        )
    lines.append("")
    lines.append("## Recomputed Guarded Policy")
    lines.append("")
    lines.append(
        f"- Aggregate speedup: `{float(guarded.get('aggregate_speedup_percent') or 0.0):.3f}%`"
    )
    lines.append(
        f"- Weighted acceptance: `{float(guarded.get('weighted_acceptance_rate') or 0.0):.3f}`"
    )
    lines.append(f"- Peak memory: `{float(guarded.get('max_peak_memory_gb') or 0.0):.3f} GB`")
    lines.append("")
    lines.append("## Real-Margin Status")
    lines.append("")
    if real_signal.get("available"):
        lines.append(f"- Available events: `{real_signal.get('event_count')}`")
        lines.append(
            f"- Accepted margin median: `{float(real_signal.get('accepted_margin_median') or 0.0):.6f}`"
        )
        lines.append(
            f"- Rejected margin median: `{float(real_signal.get('rejected_margin_median') or 0.0):.6f}`"
        )
    else:
        lines.append(f"- Missing or unavailable: {real_signal.get('reason', 'unknown')}")
    if real_coverage:
        lines.append(f"- Measured records: `{real_coverage.get('measured_records', 0)}`")
        lines.append(f"- Workloads covered: `{real_coverage.get('workloads', [])}`")
        lines.append(f"- Blocks covered: `{real_coverage.get('blocks', [])}`")
        lines.append(
            "- Sufficient for policy design: "
            f"`{real_coverage.get('sufficient_for_policy_design', False)}`"
        )
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
    lines.append("## Next Action")
    lines.append("")
    lines.append(result["next_action"])
    lines.append("")
    return "\n".join(lines)


def decide(policy_search: dict[str, Any], candidate_summary: dict[str, Any] | None) -> tuple[str, str, list[str]]:
    blockers = list(policy_search.get("blockers") or [])
    if candidate_summary is not None:
        decision = str(candidate_summary.get("decision") or "needs_more_data")
        return decision, "Use candidate/holdout/oracle inputs to finalize the XR61 decision.", blockers
    if policy_search.get("decision_hint") == "needs_real_margin_trace_capture":
        return (
            "needs_more_data",
            "Run the XR61 real-margin trace-capture command, then rerun policy search with --real-margin-records.",
            blockers,
        )
    if policy_search.get("decision_hint") == "needs_more_real_margin_coverage":
        return (
            "needs_more_data",
            "Run full or segmented XR61 real-margin trace capture when memory pressure allows, then rerun policy search with complete coverage.",
            blockers,
        )
    return (
        "needs_more_data",
        "Design and benchmark an env-gated adaptive policy before making a default-on claim.",
        blockers,
    )


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
    decision, next_action, blockers = decide(policy_search, candidate_summary)

    result = {
        "schema_version": 1,
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
