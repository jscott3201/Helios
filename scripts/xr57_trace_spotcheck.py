#!/usr/bin/env python3
"""Spot-check XR57 MTP trace top-k and margin fields in XR15 records."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
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
    return records


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def event_prefix(record: dict[str, Any], event: dict[str, Any]) -> str:
    return (
        f"{record.get('workload_id')} block={record.get('block_size')} "
        f"trial={record.get('trial_index')} pass={event.get('pass_index')}"
    )


def event_ref(record: dict[str, Any], event: dict[str, Any], event_index: int) -> dict[str, Any]:
    return {
        "record": record,
        "event": event,
        "event_index": event_index,
        "key": (
            record.get("workload_id"),
            record.get("block_size"),
            record.get("trial_index"),
            event.get("pass_index"),
            event_index,
        ),
    }


def check_event(record: dict[str, Any], event: dict[str, Any], atol: float) -> list[str]:
    prefix = event_prefix(record, event)
    issues: list[str] = []
    trace_top_k = event.get("trace_top_k")
    if trace_top_k not in (1, 5):
        issues.append(f"{prefix}: trace_top_k={trace_top_k} expected 1 or 5")
        trace_top_k = 0

    target_tokens = event.get("target_tokens") or []
    top_ids = event.get("target_top_token_ids") or []
    top_logits = event.get("target_top_logits") or []
    draft_tokens = event.get("draft_tokens") or []
    draft_logits = event.get("draft_logits") or []
    margins = event.get("logit_margins") or []
    in_top_k = event.get("draft_in_target_top_k") or []

    if not draft_tokens:
        issues.append(f"{prefix}: event has no draft tokens")
    if not target_tokens:
        issues.append(f"{prefix}: event has no target tokens")
    if not top_ids or not top_logits:
        issues.append(f"{prefix}: event has no target top-k arrays")

    for position, token in enumerate(target_tokens):
        if position >= len(top_ids) or position >= len(top_logits):
            issues.append(f"{prefix}: missing top-k arrays for target position {position}")
            continue
        ids = top_ids[position]
        logits = top_logits[position]
        if len(ids) != trace_top_k or len(logits) != trace_top_k:
            issues.append(
                f"{prefix}: top-k position {position} has lengths {len(ids)}/{len(logits)} "
                f"for trace_top_k={trace_top_k}"
            )
            continue
        if any(int(token_id) < 0 for token_id in ids):
            issues.append(f"{prefix}: sentinel target ids inside advertised top-k at position {position}: {ids}")
        if ids and ids[0] != token:
            issues.append(f"{prefix}: target token {token} not top-1 at position {position}: {ids}")
        if not all(finite(value) for value in logits):
            issues.append(f"{prefix}: non-finite target logits at position {position}: {logits}")
        for left, right in zip(logits, logits[1:]):
            if float(left) + atol < float(right):
                issues.append(f"{prefix}: target logits not descending at position {position}: {logits}")
                break

    for position, token in enumerate(draft_tokens):
        if position >= len(draft_logits) or position >= len(margins):
            issues.append(f"{prefix}: missing drafter fields for draft position {position}")
            continue
        if not finite(draft_logits[position]) or not finite(margins[position]):
            issues.append(f"{prefix}: non-finite drafter fields at position {position}")
        if float(margins[position]) < -atol:
            issues.append(f"{prefix}: negative drafter margin at position {position}: {margins[position]}")
        if position < len(top_ids):
            expected = token in top_ids[position]
            observed = bool(in_top_k[position]) if position < len(in_top_k) else False
            if observed != expected:
                issues.append(
                    f"{prefix}: draft_in_top_k mismatch at position {position}: "
                    f"token={token} top_ids={top_ids[position]} observed={observed}"
                )

    return issues


def accepted_beyond_slot0(event: dict[str, Any]) -> bool:
    return int(event.get("accepted_draft_count") or 0) > 1


def rejected_slot(event: dict[str, Any]) -> bool:
    draft_tokens = event.get("draft_tokens") or []
    accepted = int(event.get("accepted_draft_count") or 0)
    return bool(event.get("rejected")) and accepted < len(draft_tokens)


def full_accept(event: dict[str, Any]) -> bool:
    draft_tokens = event.get("draft_tokens") or []
    return bool(draft_tokens) and not bool(event.get("rejected")) and int(
        event.get("accepted_draft_count") or 0
    ) == len(draft_tokens)


def example_from_ref(ref: dict[str, Any], category: str) -> dict[str, Any]:
    record = ref["record"]
    event = ref["event"]
    return {
        "category": category,
        "workload_id": record.get("workload_id"),
        "block_size": record.get("block_size"),
        "trial_index": record.get("trial_index"),
        "pass_index": event.get("pass_index"),
        "trace_top_k": event.get("trace_top_k"),
        "accepted_draft_count": event.get("accepted_draft_count"),
        "rejected": event.get("rejected"),
        "draft_tokens": event.get("draft_tokens"),
        "target_top_token_ids": event.get("target_top_token_ids"),
        "draft_logits": event.get("draft_logits"),
        "logit_margins": event.get("logit_margins"),
        "draft_in_target_top_k": event.get("draft_in_target_top_k"),
    }


def add_sample(samples: list[dict[str, Any]], seen: set[tuple[Any, ...]], ref: dict[str, Any], category: str) -> None:
    if ref["key"] in seen:
        return
    seen.add(ref["key"])
    samples.append(example_from_ref(ref, category))


def recompute_top_k(logits: list[float], k: int) -> tuple[list[int], list[float]]:
    ranked = sorted(range(len(logits)), key=lambda index: (-float(logits[index]), index))[:k]
    return ranked, [float(logits[index]) for index in ranked]


def close_enough(left: list[float], right: list[float], atol: float) -> bool:
    return len(left) == len(right) and all(abs(float(a) - float(b)) <= atol for a, b in zip(left, right))


def verify_anchor(
    anchor_path: Path,
    event_refs: list[dict[str, Any]],
    atol: float,
) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    try:
        anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"path": str(anchor_path), "status": "failed_to_load"}, [f"{anchor_path}: failed to load anchor: {exc}"]

    raw_logits = anchor.get("logits")
    if not isinstance(raw_logits, list) or not raw_logits:
        return {"path": str(anchor_path), "status": "missing_logits"}, [f"{anchor_path}: anchor logits are empty"]
    if not all(finite(value) for value in raw_logits):
        issues.append(f"{anchor_path}: anchor logits contain non-finite values")
    vocab_size = int(anchor.get("vocab_size") or 0)
    if vocab_size != len(raw_logits):
        issues.append(f"{anchor_path}: vocab_size={vocab_size} does not match logits length {len(raw_logits)}")

    recomputed_ids, recomputed_logits = recompute_top_k([float(value) for value in raw_logits], 5)
    trace_ids = [int(value) for value in (anchor.get("trace_top_token_ids") or [])[:5]]
    trace_logits = [float(value) for value in (anchor.get("trace_top_logits") or [])[:5]]
    if trace_ids != recomputed_ids:
        issues.append(f"{anchor_path}: trace ids {trace_ids} do not match raw-logit top-5 {recomputed_ids}")
    if not close_enough(trace_logits, recomputed_logits, atol):
        issues.append(
            f"{anchor_path}: trace logits {trace_logits} do not match raw-logit top-5 {recomputed_logits}"
        )

    matched_event: dict[str, Any] | None = None
    for ref in event_refs:
        event = ref["event"]
        for position, ids in enumerate(event.get("target_top_token_ids") or []):
            logits_rows = event.get("target_top_logits") or []
            if len(ids) < 5 or position >= len(logits_rows) or len(logits_rows[position]) < 5:
                continue
            row_ids = [int(value) for value in ids[:5]]
            row_logits = [float(value) for value in logits_rows[position][:5]]
            if row_ids == recomputed_ids and close_enough(row_logits, recomputed_logits, atol):
                matched_event = example_from_ref(ref, f"anchor_position_{position}")
                matched_event["target_position"] = position
                break
        if matched_event is not None:
            break
    if matched_event is None:
        issues.append(f"{anchor_path}: raw-logit top-5 did not match any recorded sampled target row")

    top1_top2_margin = recomputed_logits[0] - recomputed_logits[1]
    return {
        "path": str(anchor_path),
        "status": "passed" if not issues else "failed",
        "vocab_size": len(raw_logits),
        "recomputed_top_token_ids": recomputed_ids,
        "recomputed_top_logits": recomputed_logits,
        "recomputed_top1_top2_margin": top1_top2_margin,
        "trace_top_token_ids": trace_ids,
        "trace_top_logits": trace_logits,
        "trace_top1_top2_margin": trace_logits[0] - trace_logits[1] if len(trace_logits) >= 2 else None,
        "matched_event": matched_event,
    }, issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--anchor-logits", type=Path, required=True)
    parser.add_argument("--min-events", type=int, default=3)
    parser.add_argument("--atol", type=float, default=1.0e-4)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    issues: list[str] = []
    event_refs: list[dict[str, Any]] = []
    for record in load_records(args.records):
        if not record.get("measured", False):
            continue
        events = (record.get("mtp") or {}).get("events") or []
        if not events:
            issues.append(
                f"{record.get('workload_id')} block={record.get('block_size')} "
                f"trial={record.get('trial_index')}: measured record has no MTP events"
            )
            continue
        for event_index, event in enumerate(events):
            ref = event_ref(record, event, event_index)
            event_refs.append(ref)
            issues.extend(check_event(record, event, args.atol))

    if not event_refs:
        issues.append("zero eligible measured MTP events found")

    clean_refs = [ref for ref in event_refs if not check_event(ref["record"], ref["event"], args.atol)]
    rejected_refs = [ref for ref in clean_refs if rejected_slot(ref["event"])]
    accepted_beyond_refs = [ref for ref in clean_refs if accepted_beyond_slot0(ref["event"])]
    full_accept_refs = [ref for ref in clean_refs if full_accept(ref["event"])]

    if not rejected_refs:
        issues.append("no clean rejected-slot event found")
    if not accepted_beyond_refs:
        issues.append("no clean accepted-slot-beyond-0 event found")

    samples: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    if rejected_refs:
        add_sample(samples, seen, rejected_refs[0], "rejected_slot")
    if accepted_beyond_refs:
        add_sample(samples, seen, accepted_beyond_refs[0], "accepted_slot_beyond_0")
    full_accept_status = "not_present"
    if full_accept_refs:
        full_accept_status = "sampled"
        add_sample(samples, seen, full_accept_refs[0], "full_accept")
    while len(samples) < args.min_events and len(samples) < len(clean_refs):
        for ref in clean_refs:
            if ref["key"] not in seen:
                add_sample(samples, seen, ref, "additional_clean")
                break

    if len(samples) < args.min_events:
        issues.append(f"sampled only {len(samples)} clean event(s); expected at least {args.min_events}")

    anchor, anchor_issues = verify_anchor(args.anchor_logits, event_refs, args.atol)
    issues.extend(anchor_issues)

    result = {
        "schema_version": 2,
        "status": "passed" if not issues else "failed",
        "records": str(args.records),
        "anchor_logits": str(args.anchor_logits),
        "checked_events": len(event_refs),
        "clean_events": len(clean_refs),
        "sampled_events": len(samples),
        "min_events": args.min_events,
        "coverage": {
            "rejected_slot": bool(rejected_refs),
            "accepted_slot_beyond_0": bool(accepted_beyond_refs),
            "full_accept": full_accept_status,
        },
        "anchor": anchor,
        "examples": samples,
        "issues": issues,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if issues:
        for issue in issues:
            print(issue)
        return 1
    print(f"XR57 trace spot-check passed for {len(samples)} sampled event(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
