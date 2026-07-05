#!/usr/bin/env python3
"""Compare XR82 first-verifier warmup A/B summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable


DEFAULT_TITLE = "XR82 MTP First Verifier-Forward Warmup"
DEFAULT_WORKLOADS = ("chat_short_1k_001", "tool_json_1k_001")


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


def nested_float(row: dict[str, Any], section: str, field: str) -> float:
    return float((row.get(section) or {}).get(field) or 0.0)


def nested_int(row: dict[str, Any], section: str, field: str) -> int:
    return int((row.get(section) or {}).get(field) or 0)


def speedup_percent(baseline_ms: float, candidate_ms: float) -> float:
    if baseline_ms <= 0.0:
        return 0.0
    return (baseline_ms - candidate_ms) / baseline_ms * 100.0


def median_field(rows: list[dict[str, Any]], section: str, field: str) -> float:
    values = [nested_float(row, section, field) for row in rows]
    return median(values) if values else 0.0


def first_event(row: dict[str, Any]) -> dict[str, Any]:
    events = (row.get("mtp") or {}).get("events") or []
    if not events:
        return {}
    return min(events, key=lambda event: int(event.get("pass_index") or 0))


def first_event_median(rows: list[dict[str, Any]], field: str) -> float:
    values = [float(first_event(row).get(field) or 0.0) for row in rows if first_event(row)]
    return median(values) if values else 0.0


def exactness(summary: dict[str, Any]) -> dict[str, Any]:
    all_rows = records(summary)
    measured = measured_records(summary)
    return {
        "records": len(all_rows),
        "measured_records": len(measured),
        "exact_records": sum(1 for row in all_rows if (row.get("comparison") or {}).get("byte_identical")),
        "exact_measured_records": sum(
            1 for row in measured if (row.get("comparison") or {}).get("byte_identical")
        ),
    }


def generated_tokens(row: dict[str, Any]) -> list[int]:
    return [int(token) for token in ((row.get("mtp") or {}).get("generated_tokens") or [])]


def record_key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row.get("workload_id")),
        str(row.get("trial_kind")),
        int(row.get("trial_index") or 0),
        int(row.get("block_size") or 0),
    )


def compare_generated_tokens(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_rows = {record_key(row): row for row in measured_records(baseline)}
    candidate_rows = {record_key(row): row for row in measured_records(candidate)}
    missing = [str(key) for key in sorted(baseline_rows) if key not in candidate_rows]
    extra = [str(key) for key in sorted(candidate_rows) if key not in baseline_rows]
    mismatches = []
    compared = 0
    for key, baseline_row in sorted(baseline_rows.items()):
        candidate_row = candidate_rows.get(key)
        if candidate_row is None:
            continue
        compared += 1
        if generated_tokens(baseline_row) != generated_tokens(candidate_row):
            mismatches.append(str(key))
    return {
        "compared_records": compared,
        "passed": compared > 0 and not missing and not extra and not mismatches,
        "missing_records": missing,
        "extra_records": extra,
        "mismatches": mismatches,
    }


def summarize_workload(
    workload_id: str,
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_phase = median_field(baseline_rows, "mtp", "decode_phase_ms")
    candidate_phase = median_field(candidate_rows, "mtp", "decode_phase_ms")
    baseline_first_forward = first_event_median(baseline_rows, "verify_forward_ms")
    candidate_first_forward = first_event_median(candidate_rows, "verify_forward_ms")
    baseline_verify_forward = median_field(baseline_rows, "mtp", "verify_forward_ms")
    candidate_verify_forward = median_field(candidate_rows, "mtp", "verify_forward_ms")
    preverify_warmup = median_field(candidate_rows, "mtp", "preverify_warmup_ms")
    baseline_accept = sum(nested_int(row, "mtp", "accepted_draft_tokens") for row in baseline_rows)
    baseline_attempt = sum(nested_int(row, "mtp", "attempted_draft_tokens") for row in baseline_rows)
    candidate_accept = sum(nested_int(row, "mtp", "accepted_draft_tokens") for row in candidate_rows)
    candidate_attempt = sum(nested_int(row, "mtp", "attempted_draft_tokens") for row in candidate_rows)
    return {
        "workload_id": workload_id,
        "records": len(candidate_rows),
        "baseline_decode_phase_ms": baseline_phase,
        "candidate_decode_phase_ms": candidate_phase,
        "decode_phase_delta_ms": candidate_phase - baseline_phase,
        "decode_phase_speedup_percent": speedup_percent(baseline_phase, candidate_phase),
        "baseline_first_verify_forward_ms": baseline_first_forward,
        "candidate_first_verify_forward_ms": candidate_first_forward,
        "first_verify_forward_delta_ms": candidate_first_forward - baseline_first_forward,
        "first_verify_forward_speedup_percent": speedup_percent(
            baseline_first_forward,
            candidate_first_forward,
        ),
        "baseline_verify_forward_ms": baseline_verify_forward,
        "candidate_verify_forward_ms": candidate_verify_forward,
        "verify_forward_delta_ms": candidate_verify_forward - baseline_verify_forward,
        "preverify_warmup_ms": preverify_warmup,
        "baseline_acceptance": baseline_accept / baseline_attempt if baseline_attempt else 0.0,
        "candidate_acceptance": candidate_accept / candidate_attempt if candidate_attempt else 0.0,
        "baseline_accepted_draft_tokens": baseline_accept,
        "baseline_attempted_draft_tokens": baseline_attempt,
        "candidate_accepted_draft_tokens": candidate_accept,
        "candidate_attempted_draft_tokens": candidate_attempt,
        "candidate_peak_memory_gb": max(
            [nested_float(row, "mtp", "peak_memory_gb") for row in candidate_rows] or [0.0]
        ),
    }


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    baseline = load_json(Path(args.baseline_summary))
    candidate = load_json(Path(args.candidate_summary))
    workload_ids = args.workload_id or list(DEFAULT_WORKLOADS)
    baseline_grouped = by_workload(measured_records(baseline))
    candidate_grouped = by_workload(measured_records(candidate))
    workloads = [
        summarize_workload(workload_id, baseline_grouped.get(workload_id, []), candidate_grouped.get(workload_id, []))
        for workload_id in workload_ids
        if baseline_grouped.get(workload_id) and candidate_grouped.get(workload_id)
    ]
    baseline_total = sum(row["baseline_decode_phase_ms"] for row in workloads)
    candidate_total = sum(row["candidate_decode_phase_ms"] for row in workloads)
    baseline_first = sum(row["baseline_first_verify_forward_ms"] for row in workloads)
    candidate_first = sum(row["candidate_first_verify_forward_ms"] for row in workloads)
    preverify_total = sum(row["preverify_warmup_ms"] for row in workloads)
    token_compare = compare_generated_tokens(baseline, candidate)
    exact = exactness(candidate)
    candidate_exact = exact["exact_measured_records"] == exact["measured_records"] and exact["measured_records"] > 0
    net_speedup = speedup_percent(baseline_total, candidate_total)
    first_speedup = speedup_percent(baseline_first, candidate_first)
    if not candidate_exact or not token_compare["passed"]:
        decision = "blocked_with_evidence"
    elif net_speedup >= args.min_net_speedup_percent:
        decision = "accept_runtime_candidate_for_protected_rerun"
    elif first_speedup > 0.0:
        decision = "warmup_hypothesis_supported_net_rejected"
    else:
        decision = "reject_runtime_candidate"
    return {
        "schema_version": 1,
        "title": args.title,
        "goal": args.goal,
        "decision": decision,
        "source": {
            "baseline_summary": args.baseline_summary,
            "candidate_summary": args.candidate_summary,
            "baseline_run_id": baseline.get("run_id"),
            "candidate_run_id": candidate.get("run_id"),
            "baseline_git_sha": baseline.get("git_sha"),
            "candidate_git_sha": candidate.get("git_sha"),
            "candidate_first_verify_warmup_enabled": bool(
                candidate.get("mtp_first_verify_warmup_enabled")
            ),
        },
        "correctness": {
            "candidate_exactness": exact,
            "generated_token_parity": token_compare,
        },
        "aggregate": {
            "workload_count": len(workloads),
            "baseline_decode_phase_ms": baseline_total,
            "candidate_decode_phase_ms": candidate_total,
            "decode_phase_delta_ms": candidate_total - baseline_total,
            "decode_phase_speedup_percent": net_speedup,
            "baseline_first_verify_forward_ms": baseline_first,
            "candidate_first_verify_forward_ms": candidate_first,
            "first_verify_forward_delta_ms": candidate_first - baseline_first,
            "first_verify_forward_speedup_percent": first_speedup,
            "candidate_preverify_warmup_ms": preverify_total,
            "candidate_max_peak_memory_gb": max(
                [row["candidate_peak_memory_gb"] for row in workloads] or [0.0]
            ),
        },
        "workloads": workloads,
        "recommendations": [
            "Use this result only as a native warm/JIT/cache hypothesis test.",
            "Do not promote broad MTP default-on from this selected-lane run.",
            "Keep preverify warmup cost included in decode_phase_ms for request-path claims.",
            "If first verifier forward improves but net decode regresses, move the warmup out of request path or target lower-level graph materialization instead.",
        ],
    }


def fmt(value: float) -> str:
    return f"{value:.3f}"


def render_markdown(result: dict[str, Any]) -> str:
    aggregate = result["aggregate"]
    lines = [
        f"# {result['title']}",
        "",
        f"- Decision: `{result['decision']}`",
        f"- Baseline: `{result['source']['baseline_summary']}`",
        f"- Candidate: `{result['source']['candidate_summary']}`",
        f"- Candidate warmup enabled: `{result['source']['candidate_first_verify_warmup_enabled']}`",
        "",
        "## Correctness",
        "",
        "| Metric | Value |",
        "|---|---:|",
        "| Candidate exact measured records | "
        f"{result['correctness']['candidate_exactness']['exact_measured_records']}/"
        f"{result['correctness']['candidate_exactness']['measured_records']} |",
        "| Generated-token parity | "
        f"`{result['correctness']['generated_token_parity']['passed']}` |",
        "| Compared records | "
        f"{result['correctness']['generated_token_parity']['compared_records']} |",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Baseline MTP decode phase ms | {fmt(aggregate['baseline_decode_phase_ms'])} |",
        f"| Candidate MTP decode phase ms | {fmt(aggregate['candidate_decode_phase_ms'])} |",
        f"| Decode phase delta ms | {fmt(aggregate['decode_phase_delta_ms'])} |",
        f"| Decode phase speedup % | {fmt(aggregate['decode_phase_speedup_percent'])} |",
        f"| Baseline first verify-forward ms | {fmt(aggregate['baseline_first_verify_forward_ms'])} |",
        f"| Candidate first verify-forward ms | {fmt(aggregate['candidate_first_verify_forward_ms'])} |",
        f"| First verify-forward delta ms | {fmt(aggregate['first_verify_forward_delta_ms'])} |",
        f"| First verify-forward speedup % | {fmt(aggregate['first_verify_forward_speedup_percent'])} |",
        f"| Candidate preverify warmup ms | {fmt(aggregate['candidate_preverify_warmup_ms'])} |",
        f"| Candidate peak GB | {fmt(aggregate['candidate_max_peak_memory_gb'])} |",
        "",
        "## Workloads",
        "",
        "| Workload | Baseline phase ms | Candidate phase ms | Phase speedup % | Warmup ms | Baseline first forward ms | Candidate first forward ms | First forward speedup % | Acceptance baseline -> candidate | Peak GB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["workloads"]:
        lines.append(
            "| `{workload}` | {baseline_phase} | {candidate_phase} | {phase_speedup} | {warmup} | {baseline_first} | {candidate_first} | {first_speedup} | {baseline_accept}/{baseline_attempt} -> {candidate_accept}/{candidate_attempt} | {peak} |".format(
                workload=row["workload_id"],
                baseline_phase=fmt(row["baseline_decode_phase_ms"]),
                candidate_phase=fmt(row["candidate_decode_phase_ms"]),
                phase_speedup=fmt(row["decode_phase_speedup_percent"]),
                warmup=fmt(row["preverify_warmup_ms"]),
                baseline_first=fmt(row["baseline_first_verify_forward_ms"]),
                candidate_first=fmt(row["candidate_first_verify_forward_ms"]),
                first_speedup=fmt(row["first_verify_forward_speedup_percent"]),
                baseline_accept=row["baseline_accepted_draft_tokens"],
                baseline_attempt=row["baseline_attempted_draft_tokens"],
                candidate_accept=row["candidate_accepted_draft_tokens"],
                candidate_attempt=row["candidate_attempted_draft_tokens"],
                peak=fmt(row["candidate_peak_memory_gb"]),
            )
        )
    lines.extend(["", "## Recommendations", ""])
    for recommendation in result["recommendations"]:
        lines.append(f"- {recommendation}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-md", default="xr82-first-verify-warmup.md")
    parser.add_argument("--out-json", default="xr82-first-verify-warmup.json")
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--goal", default="XR82-mtp-first-verifier-forward-warmup")
    parser.add_argument("--workload-id", action="append")
    parser.add_argument("--min-net-speedup-percent", type=float, default=5.0)
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
