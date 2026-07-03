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


def check_event(record: dict[str, Any], event: dict[str, Any], event_index: int, atol: float) -> list[str]:
    prefix = (
        f"{record.get('workload_id')} block={record.get('block_size')} "
        f"trial={record.get('trial_index')} pass={event.get('pass_index')}"
    )
    issues: list[str] = []
    if event.get("trace_top_k") != 5:
        issues.append(f"{prefix}: trace_top_k={event.get('trace_top_k')} expected 5")

    target_tokens = event.get("target_tokens") or []
    top_ids = event.get("target_top_token_ids") or []
    top_logits = event.get("target_top_logits") or []
    draft_tokens = event.get("draft_tokens") or []
    draft_logits = event.get("draft_logits") or []
    margins = event.get("logit_margins") or []
    in_top_k = event.get("draft_in_target_top_k") or []

    for position, token in enumerate(target_tokens):
        if position >= len(top_ids) or position >= len(top_logits):
            issues.append(f"{prefix}: missing top-k arrays for target position {position}")
            continue
        ids = top_ids[position]
        logits = top_logits[position]
        if len(ids) != 5 or len(logits) != 5:
            issues.append(f"{prefix}: top-k position {position} has lengths {len(ids)}/{len(logits)}")
            continue
        if ids[0] != token:
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

    if event_index == 0 and not draft_tokens:
        issues.append(f"{prefix}: event has no draft tokens")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--min-events", type=int, default=3)
    parser.add_argument("--atol", type=float, default=1.0e-4)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    checked = 0
    issues: list[str] = []
    examples: list[dict[str, Any]] = []
    for record in load_records(args.records):
        if not record.get("measured", False):
            continue
        for event_index, event in enumerate((record.get("mtp") or {}).get("events") or []):
            event_issues = check_event(record, event, event_index, args.atol)
            issues.extend(event_issues)
            if not event_issues:
                checked += 1
                if len(examples) < args.min_events:
                    examples.append(
                        {
                            "workload_id": record.get("workload_id"),
                            "block_size": record.get("block_size"),
                            "trial_index": record.get("trial_index"),
                            "pass_index": event.get("pass_index"),
                            "draft_tokens": event.get("draft_tokens"),
                            "target_top_token_ids": event.get("target_top_token_ids"),
                            "draft_logits": event.get("draft_logits"),
                            "logit_margins": event.get("logit_margins"),
                            "draft_in_target_top_k": event.get("draft_in_target_top_k"),
                        }
                    )
            if checked >= args.min_events:
                break
        if checked >= args.min_events:
            break

    if checked < args.min_events:
        issues.append(f"checked only {checked} clean events; expected at least {args.min_events}")

    result = {
        "schema_version": 1,
        "status": "passed" if not issues else "failed",
        "records": str(args.records),
        "checked_events": checked,
        "min_events": args.min_events,
        "examples": examples,
        "issues": issues,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if issues:
        for issue in issues:
            print(issue)
        return 1
    print(f"XR57 trace spot-check passed for {checked} event(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
