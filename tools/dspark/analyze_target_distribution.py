#!/usr/bin/env python3
"""Summarize DSpark draft tokens against verifier target top-k traces."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from dspark_common import environment_summary, render_blockers, write_json


DEFAULT_RECORDS = Path("benchmarks/out/XR60-dspark-native-mlx/target-distribution-topk/records.jsonl")
DEFAULT_OUT_DIR = Path("benchmarks/out/XR60-dspark-native-mlx/target-distribution-diagnosis")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join(sys.argv)
    blockers: list[str] = []
    records = read_jsonl(args.records, blockers)
    observations = collect_observations(records, blockers)
    summary = summarize(records, observations, args.min_top_k)
    if summary["measured_records"] == 0:
        blockers.append("no measured records found")
    if observations and summary["min_observed_top_k"] < args.min_top_k:
        blockers.append(
            f"observed target top-k width {summary['min_observed_top_k']} is below required {args.min_top_k}"
        )

    result = {
        "schema_version": 1,
        "goal": "XR60-dspark-native-mlx",
        "phase": "target-distribution-diagnosis",
        "status": "passed" if not blockers else "blocked",
        "command": command,
        "environment": environment_summary(),
        "records": str(args.records),
        "min_top_k": args.min_top_k,
        "summary": summary,
        "observations": observations,
        "blockers": blockers,
    }
    write_json(args.out_dir / "target_distribution_report.json", result)
    (args.out_dir / "report.md").write_text(render_report(result), encoding="utf-8")
    (args.out_dir / "blockers.md").write_text(
        render_blockers("XR60 target distribution diagnosis", blockers, command),
        encoding="utf-8",
    )
    if blockers and not args.allow_blocked:
        return 2
    return 0


def read_jsonl(path: Path, blockers: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        blockers.append(f"missing records JSONL: {path}")
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except Exception as error:  # noqa: BLE001 - reported in artifact.
                blockers.append(f"could not parse {path}:{line_number}: {error}")
    return records


def collect_observations(records: list[dict[str, Any]], blockers: list[str]) -> list[dict[str, Any]]:
    observations = []
    for record_index, record in enumerate(records):
        if not record.get("measured"):
            continue
        traces = record.get("verify_trace")
        if not isinstance(traces, list):
            blockers.append(f"record {record_index} has no verify_trace list")
            continue
        for trace_index, trace in enumerate(traces):
            observations.extend(trace_observations(record_index, record, trace_index, trace))
    return observations


def trace_observations(
    record_index: int,
    record: dict[str, Any],
    trace_index: int,
    trace: dict[str, Any],
) -> list[dict[str, Any]]:
    draft_tokens = list_or_empty(trace.get("draft_tokens"))
    target_tokens = list_or_empty(trace.get("target_tokens"))
    draft_logits = list_or_empty(trace.get("draft_logits"))
    draft_margins = list_or_empty(trace.get("draft_margins"))
    draft_confidence = list_or_empty(trace.get("draft_confidence"))
    draft_in_top_k = list_or_empty(trace.get("draft_in_top_k"))
    top_token_ids = list_or_empty(trace.get("target_top_token_ids"))
    top_logits = list_or_empty(trace.get("target_top_logits"))
    position_offsets = list_or_empty(trace.get("position_offsets"))
    accepted_draft_count = int(trace.get("accepted_draft_count") or 0)

    observations = []
    for position, draft_token in enumerate(draft_tokens):
        target_top_ids = nested_list_at(top_token_ids, position)
        target_top_logits = nested_list_at(top_logits, position)
        top_k_width = len([token for token in target_top_ids if isinstance(token, int) and token >= 0])
        target_rank = rank_token(target_top_ids, draft_token)
        target_top1_logit = float_at(target_top_logits, 0)
        target_topk_floor_logit = float_at(target_top_logits, top_k_width - 1)
        draft_target_logit = float_at(target_top_logits, target_rank - 1) if target_rank else None
        if target_rank is not None and target_top1_logit is not None and draft_target_logit is not None:
            target_gap = target_top1_logit - draft_target_logit
            target_rank_lower_bound = target_rank
            target_gap_lower_bound = target_gap
        elif target_top1_logit is not None and target_topk_floor_logit is not None:
            target_gap = None
            target_rank_lower_bound = top_k_width + 1
            target_gap_lower_bound = target_top1_logit - target_topk_floor_logit
        else:
            target_gap = None
            target_rank_lower_bound = None
            target_gap_lower_bound = None

        observations.append(
            {
                "record_index": record_index,
                "workload_id": record.get("workload_id"),
                "scheduled_len": record.get("scheduled_len"),
                "warmup_target_tokens": record.get("warmup_target_tokens"),
                "trace_index": trace_index,
                "position_index": position,
                "position_offset": value_at(position_offsets, position),
                "draft_token": draft_token,
                "target_token": value_at(target_tokens, position),
                "accepted_position": position < accepted_draft_count,
                "draft_logit": value_at(draft_logits, position),
                "draft_margin": value_at(draft_margins, position),
                "draft_confidence": value_at(draft_confidence, position),
                "draft_in_target_top_k": bool(value_at(draft_in_top_k, position)),
                "target_top_k_width": top_k_width,
                "target_rank": target_rank,
                "target_rank_lower_bound": target_rank_lower_bound,
                "target_top_token_ids": target_top_ids[:top_k_width],
                "target_top_logits": target_top_logits[:top_k_width],
                "target_top1_logit": target_top1_logit,
                "target_draft_logit": draft_target_logit,
                "target_top1_minus_draft_logit": target_gap,
                "target_top1_minus_topk_floor_lower_bound": target_gap_lower_bound,
            }
        )
    return observations


def summarize(
    records: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    min_top_k: int,
) -> dict[str, Any]:
    measured = [record for record in records if record.get("measured")]
    top_k_widths = [int(obs["target_top_k_width"]) for obs in observations if obs.get("target_top_k_width")]
    in_top_k = [obs for obs in observations if obs.get("draft_in_target_top_k")]
    accepted = [obs for obs in observations if obs.get("accepted_position")]
    outside = [obs for obs in observations if not obs.get("draft_in_target_top_k")]
    lower_bounds = [
        float(obs["target_top1_minus_topk_floor_lower_bound"])
        for obs in outside
        if obs.get("target_top1_minus_topk_floor_lower_bound") is not None
    ]
    confidences = [
        float(obs["draft_confidence"])
        for obs in observations
        if obs.get("draft_confidence") is not None
    ]
    workloads = sorted({str(record.get("workload_id")) for record in measured})
    by_workload = {
        workload: summarize_workload(workload, observations)
        for workload in workloads
    }
    return {
        "record_count": len(records),
        "measured_records": len(measured),
        "exact_records": sum(1 for record in measured if record.get("exact")),
        "workloads": workloads,
        "scheduled_lens": sorted({record.get("scheduled_len") for record in measured}),
        "observation_count": len(observations),
        "accepted_observations": len(accepted),
        "accepted_observation_rate": ratio(len(accepted), len(observations)),
        "draft_in_target_top_k_count": len(in_top_k),
        "draft_in_target_top_k_rate": ratio(len(in_top_k), len(observations)),
        "all_drafts_outside_target_top_k": bool(observations) and not in_top_k,
        "min_required_top_k": min_top_k,
        "min_observed_top_k": min(top_k_widths, default=0),
        "max_observed_top_k": max(top_k_widths, default=0),
        "outside_top_k_count": len(outside),
        "outside_top_k_lower_bound_gap_min": min(lower_bounds, default=None),
        "outside_top_k_lower_bound_gap_median": median(lower_bounds),
        "outside_top_k_lower_bound_gap_max": max(lower_bounds, default=None),
        "draft_confidence_min": min(confidences, default=None),
        "draft_confidence_median": median(confidences),
        "draft_confidence_max": max(confidences, default=None),
        "by_workload": by_workload,
        "diagnosis": diagnosis(len(observations), len(in_top_k), len(accepted), top_k_widths),
    }


def summarize_workload(workload: str, observations: list[dict[str, Any]]) -> dict[str, Any]:
    items = [obs for obs in observations if obs.get("workload_id") == workload]
    in_top_k = [obs for obs in items if obs.get("draft_in_target_top_k")]
    accepted = [obs for obs in items if obs.get("accepted_position")]
    lower_bounds = [
        float(obs["target_top1_minus_topk_floor_lower_bound"])
        for obs in items
        if not obs.get("draft_in_target_top_k")
        and obs.get("target_top1_minus_topk_floor_lower_bound") is not None
    ]
    return {
        "observation_count": len(items),
        "accepted_observations": len(accepted),
        "accepted_observation_rate": ratio(len(accepted), len(items)),
        "draft_in_target_top_k_count": len(in_top_k),
        "draft_in_target_top_k_rate": ratio(len(in_top_k), len(items)),
        "unique_draft_tokens": sorted({obs.get("draft_token") for obs in items}),
        "unique_target_tokens": sorted({obs.get("target_token") for obs in items}),
        "outside_top_k_lower_bound_gap_min": min(lower_bounds, default=None),
        "outside_top_k_lower_bound_gap_median": median(lower_bounds),
        "outside_top_k_lower_bound_gap_max": max(lower_bounds, default=None),
    }


def diagnosis(
    observation_count: int,
    in_top_k_count: int,
    accepted_count: int,
    top_k_widths: list[int],
) -> str:
    if observation_count == 0:
        return "no_measured_observations"
    if min(top_k_widths, default=0) <= 1:
        return "target_top_k_not_available"
    if accepted_count == 0 and in_top_k_count == 0:
        return "released_dspark_drafts_outside_target_top_k_on_measured_corpus"
    if accepted_count == 0:
        return "released_dspark_drafts_not_accepted_on_measured_corpus"
    return "some_drafts_align_with_target_distribution"


def render_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# XR60 Target Distribution Diagnosis",
        "",
        f"- Status: `{result['status']}`",
        f"- Diagnosis: `{summary['diagnosis']}`",
        f"- Records: `{result['records']}`",
        f"- Measured records: `{summary['measured_records']}`",
        f"- Observations: `{summary['observation_count']}`",
        f"- Target top-k width: `{summary['min_observed_top_k']}` to `{summary['max_observed_top_k']}`",
        f"- Draft-in-target-top-k rate: `{summary['draft_in_target_top_k_rate']:.3f}`",
        f"- Accepted observation rate: `{summary['accepted_observation_rate']:.3f}`",
        f"- Outside-top-k lower-bound gap median: `{summary['outside_top_k_lower_bound_gap_median']}`",
        f"- Draft confidence median: `{summary['draft_confidence_median']}`",
        "",
        "## Workloads",
        "",
    ]
    for workload, item in summary["by_workload"].items():
        lines.extend(
            [
                f"### {workload}",
                "",
                f"- observations: `{item['observation_count']}`",
                f"- draft-in-target-top-k rate: `{item['draft_in_target_top_k_rate']:.3f}`",
                f"- accepted observation rate: `{item['accepted_observation_rate']:.3f}`",
                f"- unique draft tokens: `{item['unique_draft_tokens']}`",
                f"- unique target tokens: `{item['unique_target_tokens']}`",
                f"- outside-top-k lower-bound gap min/median/max: "
                f"`{item['outside_top_k_lower_bound_gap_min']}` / "
                f"`{item['outside_top_k_lower_bound_gap_median']}` / "
                f"`{item['outside_top_k_lower_bound_gap_max']}`",
                "",
            ]
        )
    lines.extend(["## Blockers", ""])
    if result["blockers"]:
        lines.extend(f"- {blocker}" for blocker in result["blockers"])
    else:
        lines.append("No blockers recorded.")
    lines.append("")
    return "\n".join(lines)


def rank_token(token_ids: list[Any], token: Any) -> int | None:
    for index, item in enumerate(token_ids, start=1):
        if item == token:
            return index
    return None


def nested_list_at(value: list[Any], index: int) -> list[Any]:
    item = value_at(value, index)
    return item if isinstance(item, list) else []


def list_or_empty(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def value_at(value: list[Any], index: int) -> Any | None:
    return value[index] if index < len(value) else None


def float_at(value: list[Any], index: int) -> float | None:
    item = value_at(value, index)
    if item is None:
        return None
    return float(item)


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-top-k", type=int, default=5)
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args()
    if args.min_top_k <= 0:
        parser.error("--min-top-k must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
