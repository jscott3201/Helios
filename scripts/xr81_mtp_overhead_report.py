#!/usr/bin/env python3
"""Build an XR81 MTP protected-aggregate overhead attribution report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


POLICY_NAME = "adaptive_policy_xr61-real-margin-v1"
DEFAULT_TITLE = "XR81 MTP Protected Aggregate Overhead Gap"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"{path}: JSON file does not exist")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: invalid JSON: {exc}") from exc


def records(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = summary.get("records") or []
    return rows if isinstance(rows, list) else []


def measured_records(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in records(summary) if row.get("measured")]


def by_workload(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("workload_id")), []).append(row)
    return grouped


def speedup_percent(baseline_ms: float, candidate_ms: float) -> float:
    if baseline_ms <= 0.0:
        return 0.0
    return (baseline_ms - candidate_ms) / baseline_ms * 100.0


def nested_float(row: dict[str, Any], section: str, field: str) -> float:
    return float((row.get(section) or {}).get(field) or 0.0)


def nested_int(row: dict[str, Any], section: str, field: str) -> int:
    return int((row.get(section) or {}).get(field) or 0)


def pct(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def median_field(rows: list[dict[str, Any]], section: str, field: str) -> float:
    values = [nested_float(row, section, field) for row in rows]
    return median(values) if values else 0.0


def policy_summary(summary: dict[str, Any]) -> dict[str, Any] | None:
    for policy in summary.get("policy_summaries") or []:
        if policy.get("policy_name") == POLICY_NAME:
            return policy
    return None


def base_workload_id(label: str) -> str:
    return str(label).split(":", 1)[0]


def selected_ids(combined: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    selected = [
        base_workload_id(label)
        for label in (combined.get("selected_lane_aggregate") or {}).get("workloads", [])
        if isinstance(label, str)
    ]
    if selected:
        return sorted(set(selected))

    combined_lane = combined.get("selected_lane_aggregate") or {}
    selected = [base_workload_id(row.get("workload_id")) for row in combined_lane.get("workloads") or []]
    if selected:
        return sorted(set(selected))

    policy = policy_summary(candidate)
    return sorted({base_workload_id(label) for label in (policy or {}).get("selected_workloads") or []})


def event_rows(record_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in record_rows:
        workload_id = str(row.get("workload_id"))
        trial_index = int(row.get("trial_index") or 0)
        for event in (row.get("mtp") or {}).get("events") or []:
            if not isinstance(event, dict):
                continue
            event = dict(event)
            event["workload_id"] = workload_id
            event["trial_index"] = trial_index
            events.append(event)
    return events


def split_first_later_events(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    first_pass_by_record: dict[tuple[str, int], int] = {}
    for event in events:
        key = (str(event.get("workload_id")), int(event.get("trial_index") or 0))
        pass_index = int(event.get("pass_index") or 0)
        if key not in first_pass_by_record or pass_index < first_pass_by_record[key]:
            first_pass_by_record[key] = pass_index

    first_events = []
    later_events = []
    for event in events:
        key = (str(event.get("workload_id")), int(event.get("trial_index") or 0))
        if int(event.get("pass_index") or 0) == first_pass_by_record.get(key):
            first_events.append(event)
        else:
            later_events.append(event)
    return first_events, later_events


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    verify_forward = [float(event.get("verify_forward_ms") or 0.0) for event in events]
    verify_ms = [float(event.get("verify_ms") or 0.0) for event in events]
    accepted = [int(event.get("accepted_draft_count") or 0) for event in events]
    rejected_count = sum(1 for event in events if event.get("rejected"))
    first_events, later_events = split_first_later_events(events)
    first_forward = [float(event.get("verify_forward_ms") or 0.0) for event in first_events]
    later_forward = [float(event.get("verify_forward_ms") or 0.0) for event in later_events]

    top_k_values = []
    margins = []
    for event in events:
        top_k_values.extend(bool(value) for value in event.get("draft_in_target_top_k") or [])
        margins.extend(float(value) for value in event.get("logit_margins") or [])

    buckets: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        buckets.setdefault(int(event.get("accepted_draft_count") or 0), []).append(event)

    bucket_rows = []
    for accepted_count, bucket_events in sorted(buckets.items()):
        forward = [float(event.get("verify_forward_ms") or 0.0) for event in bucket_events]
        repair = [float(event.get("verify_repair_ms") or 0.0) for event in bucket_events]
        bucket_rows.append(
            {
                "accepted_draft_count": accepted_count,
                "events": len(bucket_events),
                "rejected_events": sum(1 for event in bucket_events if event.get("rejected")),
                "verify_forward_p50_ms": median(forward) if forward else 0.0,
                "verify_forward_p95_ms": pct(forward, 0.95),
                "verify_repair_p50_ms": median(repair) if repair else 0.0,
                "verify_forward_mean_ms": mean(forward) if forward else 0.0,
            }
        )

    return {
        "events": len(events),
        "rejected_events": rejected_count,
        "accepted_draft_tokens": sum(accepted),
        "verify_forward_total_ms": sum(verify_forward),
        "verify_forward_p50_ms": median(verify_forward) if verify_forward else 0.0,
        "verify_forward_p95_ms": pct(verify_forward, 0.95),
        "verify_ms_total_ms": sum(verify_ms),
        "first_pass_events": len(first_events),
        "first_pass_verify_forward_p50_ms": median(first_forward) if first_forward else 0.0,
        "first_pass_verify_forward_total_ms": sum(first_forward),
        "later_pass_events": len(later_events),
        "later_pass_verify_forward_p50_ms": median(later_forward) if later_forward else 0.0,
        "later_pass_verify_forward_total_ms": sum(later_forward),
        "top_k_rate": (sum(1 for value in top_k_values if value) / len(top_k_values)) if top_k_values else 0.0,
        "logit_margin_p50": median(margins) if margins else 0.0,
        "by_accepted_draft_count": bucket_rows,
    }


def workload_attribution(workload_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    events = event_rows(rows)
    first_events, later_events = split_first_later_events(events)
    first_forward = [float(event.get("verify_forward_ms") or 0.0) for event in first_events]
    later_forward = [float(event.get("verify_forward_ms") or 0.0) for event in later_events]
    first_p50 = median(first_forward) if first_forward else 0.0
    later_p50 = median(later_forward) if later_forward else 0.0

    return {
        "workload_id": workload_id,
        "records": len(rows),
        "baseline_decode_ms": median_field(rows, "baseline", "decode_ms"),
        "decode_phase_ms": median_field(rows, "mtp", "decode_phase_ms"),
        "draft_ms": median_field(rows, "mtp", "draft_ms"),
        "verify_ms": median_field(rows, "mtp", "verify_ms"),
        "verify_forward_ms": median_field(rows, "mtp", "verify_forward_ms"),
        "verify_repair_ms": median_field(rows, "mtp", "verify_repair_ms"),
        "repair_fallback_ms": median_field(rows, "mtp", "repair_fallback_ms"),
        "fallback_decode_ms": median_field(rows, "mtp", "fallback_decode_ms"),
        "target_verify_passes": median(
            [nested_int(row, "mtp", "target_verify_passes") for row in rows]
        ),
        "rollback_count": median([nested_int(row, "mtp", "rollback_count") for row in rows]),
        "accepted_draft_tokens": sum(nested_int(row, "mtp", "accepted_draft_tokens") for row in rows),
        "attempted_draft_tokens": sum(nested_int(row, "mtp", "attempted_draft_tokens") for row in rows),
        "acceptance_rate": (
            sum(nested_int(row, "mtp", "accepted_draft_tokens") for row in rows)
            / sum(nested_int(row, "mtp", "attempted_draft_tokens") for row in rows)
            if sum(nested_int(row, "mtp", "attempted_draft_tokens") for row in rows)
            else 0.0
        ),
        "accepted_tokens_per_verify_p50": median_field(rows, "mtp", "accepted_tokens_per_verify"),
        "first_pass_verify_forward_p50_ms": first_p50,
        "later_pass_verify_forward_p50_ms": later_p50,
        "first_pass_excess_vs_later_p50_ms": max(0.0, first_p50 - later_p50),
        "event_summary": summarize_events(events),
    }


def component_sums(workloads: list[dict[str, Any]]) -> dict[str, float]:
    fields = (
        "decode_phase_ms",
        "draft_ms",
        "verify_ms",
        "verify_forward_ms",
        "verify_repair_ms",
        "repair_fallback_ms",
        "fallback_decode_ms",
        "first_pass_excess_vs_later_p50_ms",
    )
    return {field: sum(float(workload.get(field) or 0.0) for workload in workloads) for field in fields}


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    candidate_summary = load_json(Path(args.candidate_summary))
    combined_summary = load_json(Path(args.combined_summary))
    selected = selected_ids(combined_summary, candidate_summary)
    grouped = by_workload(measured_records(candidate_summary))
    selected_rows = [row for workload_id in selected for row in grouped.get(workload_id, [])]
    workload_rows = [
        workload_attribution(workload_id, grouped.get(workload_id, []))
        for workload_id in selected
        if grouped.get(workload_id)
    ]
    sums = component_sums(workload_rows)

    protected = combined_summary.get("protected_aggregate") or {}
    selected_lane = combined_summary.get("selected_lane_aggregate") or {}
    gate_percent = float(combined_summary.get("broad_default_gate_percent") or args.broad_default_gate_percent)
    protected_baseline = float(protected.get("total_baseline_decode_ms") or 0.0)
    protected_current = float(protected.get("total_selected_decode_phase_ms") or 0.0)
    target_total = protected_baseline * (1.0 - gate_percent / 100.0)
    gap_ms = max(0.0, protected_current - target_total)
    selected_baseline = float(selected_lane.get("total_baseline_decode_ms") or 0.0)
    selected_current = float(selected_lane.get("total_selected_decode_phase_ms") or sums["decode_phase_ms"])
    bypass_baseline = max(0.0, protected_baseline - selected_baseline)
    selected_target = max(0.0, target_total - bypass_baseline)

    independent_costs = {
        "draft_ms": sums["draft_ms"],
        "verify_forward_ms": sums["verify_forward_ms"],
        "verify_repair_ms": sums["verify_repair_ms"],
        "fallback_decode_ms": sums["fallback_decode_ms"],
    }
    dominant_component = max(independent_costs, key=independent_costs.get) if independent_costs else ""
    first_pass_excess = sums["first_pass_excess_vs_later_p50_ms"]

    exact = sum(1 for row in records(candidate_summary) if (row.get("comparison") or {}).get("byte_identical"))
    measured_exact = sum(
        1 for row in measured_records(candidate_summary) if (row.get("comparison") or {}).get("byte_identical")
    )

    return {
        "schema_version": 1,
        "title": args.title,
        "goal": args.goal,
        "decision": "needs_runtime_candidate",
        "source": {
            "candidate_summary": args.candidate_summary,
            "combined_summary": args.combined_summary,
            "run_id": candidate_summary.get("run_id"),
            "git_sha": candidate_summary.get("git_sha"),
            "git_status_short": candidate_summary.get("git_status_short"),
        },
        "correctness": {
            "records": len(records(candidate_summary)),
            "measured_records": len(measured_records(candidate_summary)),
            "exact_records": exact,
            "exact_measured_records": measured_exact,
            "scoped_gates_passed": bool(combined_summary.get("scoped_gates_passed")),
            "broad_default_supported": bool(combined_summary.get("broad_default_supported")),
            "broad_default_gates": combined_summary.get("broad_default_gates") or {},
        },
        "gate_gap": {
            "gate_percent": gate_percent,
            "protected_baseline_decode_ms": protected_baseline,
            "protected_current_decode_phase_ms": protected_current,
            "protected_current_speedup_percent": float(protected.get("aggregate_speedup_percent") or 0.0),
            "target_decode_phase_ms": target_total,
            "gap_to_gate_ms": gap_ms,
            "selected_baseline_decode_ms": selected_baseline,
            "selected_current_decode_phase_ms": selected_current,
            "protected_bypass_baseline_ms": bypass_baseline,
            "selected_target_if_bypass_unchanged_ms": selected_target,
            "selected_reduction_needed_ms": max(0.0, selected_current - selected_target),
            "selected_reduction_needed_percent_of_current": (
                max(0.0, selected_current - selected_target) / selected_current * 100.0
                if selected_current
                else 0.0
            ),
            "required_selected_speedup_percent": speedup_percent(selected_baseline, selected_target),
        },
        "selected_workloads": selected,
        "workloads": workload_rows,
        "component_median_sums_ms": sums,
        "component_shares_of_selected_decode_phase": {
            field: (value / selected_current if selected_current else 0.0)
            for field, value in sums.items()
        },
        "dominant_independent_component": dominant_component,
        "gap_as_percent_of_verify_forward": (
            gap_ms / sums["verify_forward_ms"] * 100.0 if sums["verify_forward_ms"] else 0.0
        ),
        "gap_as_percent_of_draft": gap_ms / sums["draft_ms"] * 100.0 if sums["draft_ms"] else 0.0,
        "gap_as_percent_of_repair_fallback": (
            gap_ms / sums["repair_fallback_ms"] * 100.0 if sums["repair_fallback_ms"] else 0.0
        ),
        "first_pass_hypothesis": {
            "selected_first_pass_excess_vs_later_p50_ms": first_pass_excess,
            "gap_covered_if_first_pass_normalized": first_pass_excess >= gap_ms,
            "excess_to_gap_ratio": first_pass_excess / gap_ms if gap_ms else 0.0,
        },
        "event_summary": summarize_events(event_rows(selected_rows)),
        "recommendations": [
            "Target verifier-forward overhead before changing acceptance policy.",
            "Isolate the first verifier pass warm/JIT/cache cost on selected MTP rows.",
            "Keep mtp_candidate_1k_001 and 4K holdout bypass behavior unchanged.",
            "Keep broad MTP default-off until the protected aggregate clears the gate with oracle/default-overhead/memory evidence.",
        ],
    }


def fmt(value: float) -> str:
    return f"{value:.3f}"


def render_markdown(result: dict[str, Any]) -> str:
    gap = result["gate_gap"]
    sums = result["component_median_sums_ms"]
    first_pass = result["first_pass_hypothesis"]
    lines = [
        f"# {result['title']}",
        "",
        f"- Decision: `{result['decision']}`",
        f"- Source candidate: `{result['source']['candidate_summary']}`",
        f"- Source combined report: `{result['source']['combined_summary']}`",
        f"- Run ID: `{result['source']['run_id']}`",
        f"- Git SHA: `{result['source']['git_sha']}`",
        "",
        "## Correctness / Gate Context",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Records exact | {result['correctness']['exact_records']}/{result['correctness']['records']} |",
        f"| Measured records exact | {result['correctness']['exact_measured_records']}/{result['correctness']['measured_records']} |",
        f"| Scoped gates passed | `{result['correctness']['scoped_gates_passed']}` |",
        f"| Broad default supported | `{result['correctness']['broad_default_supported']}` |",
        "",
        "## Gate Gap",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Protected aggregate speedup % | {fmt(gap['protected_current_speedup_percent'])} |",
        f"| Broad gate % | {fmt(gap['gate_percent'])} |",
        f"| Protected baseline decode ms | {fmt(gap['protected_baseline_decode_ms'])} |",
        f"| Current protected decode phase ms | {fmt(gap['protected_current_decode_phase_ms'])} |",
        f"| Target decode phase ms | {fmt(gap['target_decode_phase_ms'])} |",
        f"| Gap to gate ms | {fmt(gap['gap_to_gate_ms'])} |",
        f"| Selected current decode phase ms | {fmt(gap['selected_current_decode_phase_ms'])} |",
        f"| Selected target if bypass unchanged ms | {fmt(gap['selected_target_if_bypass_unchanged_ms'])} |",
        f"| Selected reduction needed ms | {fmt(gap['selected_reduction_needed_ms'])} |",
        f"| Required selected speedup % | {fmt(gap['required_selected_speedup_percent'])} |",
        "",
        "## Component Attribution",
        "",
        "| Component | Median-sum ms | Share of selected phase | Gap as % of component |",
        "|---|---:|---:|---:|",
    ]
    component_gap_ratios = {
        "draft_ms": result["gap_as_percent_of_draft"],
        "verify_forward_ms": result["gap_as_percent_of_verify_forward"],
        "repair_fallback_ms": result["gap_as_percent_of_repair_fallback"],
    }
    for field in (
        "draft_ms",
        "verify_forward_ms",
        "verify_repair_ms",
        "repair_fallback_ms",
        "fallback_decode_ms",
        "first_pass_excess_vs_later_p50_ms",
    ):
        share = result["component_shares_of_selected_decode_phase"].get(field, 0.0)
        ratio = component_gap_ratios.get(field, 0.0)
        ratio_cell = fmt(ratio) if ratio else "n/a"
        lines.append(f"| `{field}` | {fmt(sums.get(field, 0.0))} | {fmt(share * 100.0)}% | {ratio_cell} |")

    lines.extend(
        [
            "",
            f"Dominant independent component: `{result['dominant_independent_component']}`.",
            "",
            "## First Verifier Pass Hypothesis",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Selected first-pass excess vs later-pass p50 ms | {fmt(first_pass['selected_first_pass_excess_vs_later_p50_ms'])} |",
            f"| Gap covered if normalized | `{first_pass['gap_covered_if_first_pass_normalized']}` |",
            f"| Excess / gap ratio | {fmt(first_pass['excess_to_gap_ratio'])} |",
            "",
            "## Workload Detail",
            "",
            "| Workload | Baseline ms | MTP phase ms | Speedup % | Draft ms | Verify forward ms | Repair fallback ms | First-pass excess ms | Accepted/Attempted |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["workloads"]:
        lines.append(
            "| `{workload}` | {baseline} | {phase} | {speedup} | {draft} | {forward} | {repair} | {first_excess} | {accepted}/{attempted} |".format(
                workload=row["workload_id"],
                baseline=fmt(row["baseline_decode_ms"]),
                phase=fmt(row["decode_phase_ms"]),
                speedup=fmt(speedup_percent(row["baseline_decode_ms"], row["decode_phase_ms"])),
                draft=fmt(row["draft_ms"]),
                forward=fmt(row["verify_forward_ms"]),
                repair=fmt(row["repair_fallback_ms"]),
                first_excess=fmt(row["first_pass_excess_vs_later_p50_ms"]),
                accepted=row["accepted_draft_tokens"],
                attempted=row["attempted_draft_tokens"],
            )
        )

    events = result["event_summary"]
    lines.extend(
        [
            "",
            "## Event Summary",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Events | {events['events']} |",
            f"| Rejected events | {events['rejected_events']} |",
            f"| Verify-forward p50 ms | {fmt(events['verify_forward_p50_ms'])} |",
            f"| Verify-forward p95 ms | {fmt(events['verify_forward_p95_ms'])} |",
            f"| First-pass verify-forward p50 ms | {fmt(events['first_pass_verify_forward_p50_ms'])} |",
            f"| Later-pass verify-forward p50 ms | {fmt(events['later_pass_verify_forward_p50_ms'])} |",
            f"| Top-k rate | {fmt(events['top_k_rate'])} |",
            f"| Logit margin p50 | {fmt(events['logit_margin_p50'])} |",
            "",
            "| Accepted draft count | Events | Rejected events | Verify-forward p50 ms | Verify-forward p95 ms | Repair p50 ms |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for bucket in events["by_accepted_draft_count"]:
        lines.append(
            "| {accepted} | {events} | {rejected} | {forward_p50} | {forward_p95} | {repair_p50} |".format(
                accepted=bucket["accepted_draft_count"],
                events=bucket["events"],
                rejected=bucket["rejected_events"],
                forward_p50=fmt(bucket["verify_forward_p50_ms"]),
                forward_p95=fmt(bucket["verify_forward_p95_ms"]),
                repair_p50=fmt(bucket["verify_repair_p50_ms"]),
            )
        )

    lines.extend(["", "## Recommendations", ""])
    for recommendation in result["recommendations"]:
        lines.append(f"- {recommendation}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--combined-summary", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-md", default="xr81-mtp-overhead-gap.md")
    parser.add_argument("--out-json", default="xr81-mtp-overhead-gap.json")
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--goal", default="XR81-mtp-protected-aggregate-overhead-gap")
    parser.add_argument("--broad-default-gate-percent", type=float, default=25.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_result(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / args.out_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / args.out_md).write_text(render_markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
