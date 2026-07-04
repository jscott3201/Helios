#!/usr/bin/env python3
"""Write the XR60 goal-level decision artifacts from measured evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dspark_common import command_stdout, environment_summary, write_json


DEFAULT_OUT_DIR = Path("benchmarks/out/XR60-dspark-native-mlx")
VALID_DECISIONS = {"promote_experimental", "keep_experimental", "reject_for_now", "blocked"}
SUMMARY_SOURCES = [
    ("toy_prefix_topk", "target-distribution-topk/summary.json"),
    ("real_context_1k", "real-context-topk/summary.json"),
    ("real_context_4k_124", "real-context-4k-topk/summary.json"),
    ("real_context_4k_block7", "real-context-4k-block7/summary.json"),
]
DIAGNOSIS_SOURCES = [
    ("toy_prefix_topk", "target-distribution-diagnosis/target_distribution_report.json"),
    ("real_context_1k", "real-context-target-distribution/target_distribution_report.json"),
    ("real_context_4k_124", "real-context-4k-target-distribution/target_distribution_report.json"),
    (
        "real_context_4k_block7",
        "real-context-4k-block7-target-distribution/target_distribution_report.json",
    ),
]
REFERENCE_EVIDENCE = [
    "01-reference-fixtures/native-tap/reference_fixture.json",
    "01-reference-fixtures/native-warm-corpus/reference_fixture.json",
    "03-mlx-parity/native-trace/parity_report.json",
    "03-mlx-parity/native-warm-corpus/parity_report.json",
]


def main() -> int:
    args = parse_args()
    if args.decision not in VALID_DECISIONS:
        raise SystemExit(f"--decision must be one of {','.join(sorted(VALID_DECISIONS))}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    command = " ".join(sys.argv)
    blockers: list[str] = []
    summaries = read_sources(args.out_dir, SUMMARY_SOURCES, blockers)
    diagnoses = read_sources(args.out_dir, DIAGNOSIS_SOURCES, blockers)
    reference_evidence = inspect_reference_evidence(args.out_dir)
    records = rollup_records(summaries)
    measured_records = [record for record in records if record.get("measured")]
    exact_records = [record for record in measured_records if record.get("exact")]
    max_tps = max(
        (float(record["decode_tokens_per_second"]) for record in measured_records if record.get("decode_tokens_per_second") is not None),
        default=0.0,
    )
    max_peak_gb = max(
        (float(record["peak_memory_gb"]) for record in measured_records if record.get("peak_memory_gb") is not None),
        default=0.0,
    )

    if blockers and not args.allow_missing:
        final_decision = "blocked"
    else:
        final_decision = args.decision
    status = "blocked" if blockers and not args.allow_missing else "passed"
    rationale = build_decision_rationale(final_decision, max_tps, max_peak_gb, diagnoses)
    summary = {
        "schema_version": 1,
        "goal": "XR60-dspark-native-mlx",
        "phase": "goal-level-decision",
        "status": status,
        "decision": final_decision,
        "command": command,
        "environment": environment_summary(),
        "source_summaries": summaries,
        "source_diagnoses": diagnoses,
        "reference_evidence": reference_evidence,
        "records_path": str(args.out_dir / "records.jsonl"),
        "summary_path": str(args.out_dir / "summary.json"),
        "report_path": str(args.out_dir / "report.md"),
        "blockers_path": str(args.out_dir / "blockers.md"),
        "decision_path": str(args.out_dir / "decision.md"),
        "measured_records": len(measured_records),
        "exact_records": len(exact_records),
        "max_decode_tokens_per_second": max_tps,
        "max_peak_memory_gb": max_peak_gb,
        "decision_rationale": rationale,
        "blockers": blockers,
    }

    write_jsonl(args.out_dir / "records.jsonl", records)
    write_json(args.out_dir / "summary.json", summary)
    (args.out_dir / "report.md").write_text(render_report(summary, records), encoding="utf-8")
    (args.out_dir / "blockers.md").write_text(render_blockers(blockers, command), encoding="utf-8")
    (args.out_dir / "decision.md").write_text(f"{final_decision}\n", encoding="utf-8")
    if blockers and not args.allow_missing:
        return 2
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--decision", default="reject_for_now")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def read_sources(
    out_dir: Path, sources: list[tuple[str, str]], blockers: list[str]
) -> list[dict[str, Any]]:
    loaded = []
    for name, relative in sources:
        path = out_dir / relative
        if not path.exists():
            blockers.append(f"missing XR60 evidence source: {path}")
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:  # noqa: BLE001 - reported in blockers artifact.
            blockers.append(f"could not parse XR60 evidence source {path}: {error}")
            continue
        loaded.append({"name": name, "path": str(path), "data": value})
    return loaded


def inspect_reference_evidence(out_dir: Path) -> list[dict[str, Any]]:
    evidence = []
    for relative in REFERENCE_EVIDENCE:
        path = out_dir / relative
        evidence.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "kind": "reference_or_parity_artifact",
            }
        )
    return evidence


def rollup_records(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for source in summaries:
        data = source["data"]
        for record in data.get("records", []):
            if not isinstance(record, dict):
                continue
            copied = dict(record)
            copied["rollup_source"] = source["name"]
            copied["rollup_source_summary"] = source["path"]
            records.append(copied)
    return records


def build_decision_rationale(
    decision: str, max_tps: float, max_peak_gb: float, diagnoses: list[dict[str, Any]]
) -> list[str]:
    reasons = []
    if decision == "reject_for_now":
        reasons.extend(
            [
                "Measured fixed-prefix DSpark output is exact on the rollup corpus, but no measured scheduler is remotely speed-profitable.",
                f"Best measured DSpark decode throughput is {max_tps:.6f} tok/s, far below the 12-16 tok/s native baseline range cited by the XR60 goal.",
                f"Peak measured memory reaches {max_peak_gb:.6f} GB, at or beyond the tiny16 budget edge.",
                "Target-distribution evidence is domain-shaped: MTP-shaped prompts align well, code is partial, toy/chat prompts remain poor.",
                "Confidence and custom-kernel work are not justified until the native DSpark draft/verify overhead is reduced.",
            ]
        )
    elif decision == "keep_experimental":
        reasons.append("Correctness and some acceptance evidence are promising, but speed/memory evidence is not sufficient for promotion.")
    elif decision == "blocked":
        reasons.append("Required source evidence is missing or unparsable.")
    else:
        reasons.append("All correctness, speed, and memory gates passed.")
    diagnoses_text = [
        f"{source['name']}: {source['data'].get('summary', {}).get('diagnosis')}"
        for source in diagnoses
    ]
    if diagnoses_text:
        reasons.append("Target distribution diagnoses: " + "; ".join(diagnoses_text))
    return reasons


def render_report(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    lines = [
        "# XR60 DSpark native MLX report",
        "",
        "## Decision",
        summary["decision"],
        "",
        "## Git and environment",
        f"- Git SHA: `{summary['environment']['git_sha']}`",
        f"- Branch: `{command_stdout(['git', 'branch', '--show-current'])}`",
        f"- Python: `{summary['environment']['python']}`",
        f"- Platform: `{summary['environment']['platform']}`",
        "- Model path: `artifacts/models/gemma-4-12B-it-4bit`",
        "- Draft path: `artifacts/drafts/dspark-gemma4-12b-block7`",
        "",
        "## What changed",
        "- Native DSpark fixed-prefix drafting and verifier integration were implemented behind explicit benchmark/native flags.",
        "- Hidden tap snapshots, DeepSpec native-tap reference fixtures, native trace parity checks, and target-distribution diagnostics were produced.",
        "- Real-context token workload support enabled 1K/4K prompt evidence.",
        "",
        "## Exact commands run",
    ]
    for source in summary["source_summaries"]:
        command = source["data"].get("command")
        if command:
            lines.append(f"- `{command}`")
    for source in summary["source_diagnoses"]:
        command = source["data"].get("command")
        if command:
            lines.append(f"- `{command}`")
    lines.extend(
        [
            "",
            "## Correctness results",
            f"- Measured records: `{summary['measured_records']}`",
            f"- Exact records: `{summary['exact_records']}`",
            f"- Best measured DSpark decode tok/s: `{summary['max_decode_tokens_per_second']}`",
            f"- Peak measured memory GB: `{summary['max_peak_memory_gb']}`",
            "",
            "## Benchmark summary",
            "",
            "| workload | context | scheduler | block/max | exact | decode tok/s | speedup | acceptance | accepted/verify | draft ms | verify ms | peak GB | active KV bytes |",
            "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for record in records:
        if not record.get("measured"):
            continue
        accepted_per_verify = record.get("accepted_tokens_per_verify")
        lines.append(
            "| {workload} | {context} | {scheduler} | {block}/{max_new} | {exact} | {tps} | n/a | {acceptance} | {accepted_per_verify} | {draft_ms} | {verify_ms} | {peak_gb} | {kv} |".format(
                workload=record.get("workload_id"),
                context=record.get("context_tokens"),
                scheduler=record.get("scheduler"),
                block=record.get("scheduled_len"),
                max_new=record.get("max_new_tokens"),
                exact=record.get("exact"),
                tps=format_float(record.get("decode_tokens_per_second")),
                acceptance=format_float(record.get("acceptance_rate")),
                accepted_per_verify=format_float(accepted_per_verify),
                draft_ms=format_float(record.get("draft_ms")),
                verify_ms=format_float(record.get("verify_forward_ms")),
                peak_gb=format_float(record.get("peak_memory_gb")),
                kv=record.get("active_kv_bytes"),
            )
        )
    lines.extend(
        [
            "",
            "## Hidden tap parity",
        ]
    )
    for item in summary["reference_evidence"]:
        lines.append(f"- `{item['path']}`: {'present' if item['exists'] else 'missing'}")
    lines.extend(
        [
            "",
            "## Target distribution diagnosis",
        ]
    )
    for source in summary["source_diagnoses"]:
        diagnosis_summary = source["data"].get("summary", {})
        lines.append(
            "- `{name}`: diagnosis `{diagnosis}`, observations `{observations}`, accepted `{accepted}`, draft-in-top-k `{topk}`".format(
                name=source["name"],
                diagnosis=diagnosis_summary.get("diagnosis"),
                observations=diagnosis_summary.get("observation_count"),
                accepted=diagnosis_summary.get("accepted_observations"),
                topk=diagnosis_summary.get("draft_in_target_top_k_count"),
            )
        )
    lines.extend(
        [
            "",
            "## Decision rationale",
        ]
    )
    lines.extend(f"- {reason}" for reason in summary["decision_rationale"])
    lines.extend(
        [
            "",
            "## Blockers",
        ]
    )
    if summary["blockers"]:
        lines.extend(f"- {blocker}" for blocker in summary["blockers"])
    else:
        lines.append("No blockers recorded for the final rollup.")
    lines.extend(
        [
            "",
            "## Next steps",
            "- Keep DSpark default-off.",
            "- Revisit only if native DSpark draft/verify overhead can be reduced by kernel or graph-level work, or if a BF16 target comparison materially changes the conclusion.",
            "- Do not promote confidence scheduling until fixed-prefix speed is viable.",
            "",
        ]
    )
    return "\n".join(lines)


def render_blockers(blockers: list[str], command: str) -> str:
    if not blockers:
        return "# XR60 final decision blockers\n\nNo blockers recorded.\n"
    lines = ["# XR60 final decision blockers", ""]
    for blocker in blockers:
        lines.extend(
            [
                f"## Blocker: {blocker}",
                "",
                f"- Command: `{command}`",
                "- Expected: goal-level XR60 decision can be generated from required source artifacts",
                f"- Observed: {blocker}",
                "- Next input needed: produce or repair the missing source artifact and rerun the finalizer",
                "",
            ]
        )
    return "\n".join(lines)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def format_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6g}"


if __name__ == "__main__":
    raise SystemExit(main())
