#!/usr/bin/env python3
"""Create or block an XR60 DeepSpec/PyTorch DSpark reference fixture."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any

from dspark_common import (
    EXPECTED_BLOCK_SIZE,
    EXPECTED_DSPARK_REVISION,
    EXPECTED_MODEL_ID,
    EXPECTED_TARGET_LAYER_IDS,
    EXPECTED_TARGET_MODEL_ID,
    EXPECTED_TARGET_REVISION,
    config_summary,
    environment_summary,
    reference_revisions,
    render_blockers,
    sha256_file,
    sha256_text,
    write_json,
)


DEFAULT_DRAFT_PATH = Path("artifacts/drafts/dspark-gemma4-12b-block7")
DEFAULT_OUT_DIR = Path("benchmarks/out/XR60-dspark-native-mlx/01-reference-fixtures")
DEFAULT_NATIVE_TAP_MANIFEST = Path(
    "benchmarks/out/XR60-dspark-native-mlx/02-hidden-tap-parity/native-smoke/native_tap_snapshot_manifest.json"
)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    command = " ".join(sys.argv)
    revision = args.revision or EXPECTED_DSPARK_REVISION
    draft, blockers = config_summary(args.draft_path, args.model_id, revision)
    env = environment_summary()
    native_taps = native_tap_summary(args.native_tap_manifest, blockers)
    for package in ["torch", "safetensors", "transformers"]:
        if not env["packages"][package]:
            blockers.append(f"missing Python package required for reference fixture: {package}")
    if not env["packages"]["deepspec"]:
        blockers.append("DeepSpec Python reference package is not installed or importable as `deepspec`")
    if args.native_tap_manifest is None:
        blockers.append("native tap manifest is required for the current DeepSpec native-tap reference path")

    prompts = prompts_from_native_taps(native_taps) or [
        prompt_summary("xr60_smoke_tokens", args.prompt_token_ids)
    ]

    reference_output = args.reference_output or (args.out_dir / "reference_fixture.json")
    fixture_result: dict[str, Any] | None = None
    if not blockers:
        try:
            fixture_result = export_native_tap_reference_fixture(
                args=args,
                native_taps=native_taps,
                reference_output=reference_output,
            )
        except Exception as error:  # noqa: BLE001 - fail closed into manifest/blockers.
            blockers.append(f"DeepSpec native-tap reference export failed: {error}")

    manifest = {
        "schema_version": 1,
        "goal": "XR60-dspark-native-mlx",
        "phase": "01-reference-fixtures",
        "status": "blocked" if blockers else "passed",
        "command": command,
        "environment": env,
        "reference_revisions": reference_revisions(),
        "deepseek_dspark": draft,
        "native_tap_manifest": native_taps,
        "target_model_id": args.target_model_id,
        "tokenizer_revision": args.tokenizer_revision,
        "prompts": prompts,
        "reference_output": str(reference_output),
        "reference_output_summary": fixture_result,
        "reference_fixture_mode": "native_tap_snapshot",
        "block_size": args.block_size,
        "top_k": args.top_k,
        "include_full_logits": args.include_full_logits,
        "expected_fixture_fields": [
            "input_token_ids",
            "target_hidden_taps",
            "target_last_hidden",
            "dspark_base_logits or dspark_base_top_k",
            "dspark_markov_logits or dspark_markov_top_k",
            "confidence_logits",
            "confidence",
            "greedy_draft_tokens",
        ],
        "blockers": blockers,
    }
    write_json(args.out_dir / "manifest.json", manifest)
    (args.out_dir / "blockers.md").write_text(
        render_blockers("XR60 reference fixture", blockers, command),
        encoding="utf-8",
    )
    if blockers:
        if args.allow_blocked:
            return 0
        return 2

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-path", type=Path, default=DEFAULT_DRAFT_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-id", default=EXPECTED_MODEL_ID)
    parser.add_argument("--revision", default=EXPECTED_DSPARK_REVISION)
    parser.add_argument("--target-model-id", default=EXPECTED_TARGET_MODEL_ID)
    parser.add_argument("--tokenizer-revision", default=EXPECTED_TARGET_REVISION)
    parser.add_argument(
        "--native-tap-manifest",
        type=Path,
        default=DEFAULT_NATIVE_TAP_MANIFEST,
        help="native_tap_snapshot_manifest.json emitted by dspark_fixed_block_matrix",
    )
    parser.add_argument(
        "--reference-output",
        type=Path,
        default=None,
        help="path for the generated DeepSpec/PyTorch reference fixture JSON",
    )
    parser.add_argument("--block-size", type=int, default=EXPECTED_BLOCK_SIZE)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--include-full-logits",
        action="store_true",
        help="write full base/Markov logits arrays; default writes compact top-k logits",
    )
    parser.add_argument(
        "--prompt-token-ids",
        default="2,106,107",
        help="comma-separated tiny smoke prompt token ids",
    )
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="write manifest/blockers and exit 0 even when fixture generation is blocked",
    )
    args = parser.parse_args()
    args.prompt_token_ids = parse_token_ids(args.prompt_token_ids)
    if args.block_size <= 0:
        parser.error("--block-size must be positive")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    return args


def parse_token_ids(value: str) -> list[int]:
    tokens = [item.strip() for item in value.split(",") if item.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("prompt token list must not be empty")
    return [int(token) for token in tokens]


def prompt_summary(prompt_id: str, token_ids: list[int]) -> dict[str, Any]:
    return {
        "id": prompt_id,
        "token_ids": token_ids,
        "sha256": sha256_text(",".join(str(token) for token in token_ids)),
    }


def prompts_from_native_taps(native_taps: dict[str, Any]) -> list[dict[str, Any]]:
    prompts = []
    for snapshot in native_taps.get("snapshots", []):
        prompt_tokens = snapshot.get("prompt_tokens")
        if isinstance(prompt_tokens, list):
            prompts.append(prompt_summary(str(snapshot.get("workload_id", "native_tap")), prompt_tokens))
    return prompts


def native_tap_summary(path: Path | None, blockers: list[str]) -> dict[str, Any]:
    if path is None:
        return {"path": None, "status": "not_provided", "snapshots": []}
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "status": "ready",
        "snapshots": [],
    }
    if not path.exists():
        blockers.append(f"missing native tap manifest: {path}")
        summary["status"] = "missing"
        return summary
    manifest = read_native_tap_manifest(path, blockers)
    if manifest is None:
        summary["status"] = "invalid"
        return summary

    target_layers = manifest.get("target_layer_ids")
    summary["schema_version"] = manifest.get("schema_version")
    summary["phase"] = manifest.get("phase")
    summary["source_status"] = manifest.get("status")
    summary["target_layer_ids"] = target_layers
    summary["run_id"] = manifest.get("run_id")
    if target_layers != EXPECTED_TARGET_LAYER_IDS:
        blockers.append(
            f"native tap manifest target_layer_ids {target_layers} do not match {EXPECTED_TARGET_LAYER_IDS}"
        )
        summary["status"] = "invalid"

    snapshots = manifest.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        blockers.append(f"native tap manifest has no snapshots: {path}")
        summary["status"] = "invalid"
        return summary

    for snapshot in snapshots:
        snapshot_summary = summarize_native_tap_snapshot(snapshot, blockers)
        summary["snapshots"].append(snapshot_summary)
        if snapshot_summary.get("status") != "ready":
            summary["status"] = "invalid"
    return summary


def read_native_tap_manifest(path: Path, blockers: list[str]) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001 - reported in manifest.
        blockers.append(f"could not read native tap manifest {path}: {error}")
        return None


def summarize_native_tap_snapshot(snapshot: dict[str, Any], blockers: list[str]) -> dict[str, Any]:
    snapshot_path = Path(str(snapshot.get("snapshot_path", "")))
    summary: dict[str, Any] = {
        "workload_id": snapshot.get("workload_id"),
        "prompt_tokens": snapshot.get("prompt_tokens"),
        "prompt_sha256": snapshot.get("prompt_sha256"),
        "prefill_greedy_token": snapshot.get("prefill_greedy_token"),
        "prefill_greedy_logit": snapshot.get("prefill_greedy_logit"),
        "context_tokens": snapshot.get("context_tokens"),
        "snapshot_path": str(snapshot_path),
        "snapshot_exists": snapshot_path.exists(),
        "tap_layer_ids": snapshot.get("tap_layer_ids"),
        "tap_shapes": snapshot.get("tap_shapes"),
        "tap_bytes": snapshot.get("tap_bytes"),
        "status": "ready",
    }
    if not snapshot_path.exists():
        blockers.append(f"missing native tap snapshot payload: {snapshot_path}")
        summary["status"] = "missing"
        return summary
    try:
        header = safetensors_header(snapshot_path)
    except Exception as error:  # noqa: BLE001 - reported in manifest.
        blockers.append(f"could not inspect native tap snapshot {snapshot_path}: {error}")
        summary["status"] = "invalid"
        return summary

    tensor_headers = {key: value for key, value in header.items() if key != "__metadata__"}
    metadata = header.get("__metadata__", {})
    expected_context_tokens = int(snapshot.get("context_tokens", 0))
    tap_tensors = []
    for index, layer_id in enumerate(EXPECTED_TARGET_LAYER_IDS):
        key = f"dspark_context.tap_{index}.hidden"
        tensor = tensor_headers.get(key)
        if tensor is None:
            blockers.append(f"native tap snapshot missing tensor {key}")
            summary["status"] = "invalid"
            continue
        shape = tensor.get("shape")
        dtype = tensor.get("dtype")
        metadata_layer = metadata.get(f"dspark_context.tap_{index}.layer_id")
        expected_shape = [1, expected_context_tokens, 3840]
        if shape != expected_shape:
            blockers.append(f"native tap tensor {key} shape {shape} does not match {expected_shape}")
            summary["status"] = "invalid"
        if dtype != "BF16":
            blockers.append(f"native tap tensor {key} dtype {dtype} is not BF16")
            summary["status"] = "invalid"
        if str(layer_id) != str(metadata_layer):
            blockers.append(f"native tap tensor {key} metadata layer {metadata_layer} is not {layer_id}")
            summary["status"] = "invalid"
        tap_tensors.append(
            {
                "key": key,
                "layer_id": layer_id,
                "shape": shape,
                "dtype": dtype,
                "metadata_layer_id": metadata_layer,
            }
        )

    summary["snapshot_sha256"] = sha256_file(snapshot_path)
    summary["snapshot_bytes"] = snapshot_path.stat().st_size
    summary["snapshot_metadata"] = {
        "format": metadata.get("format"),
        "snapshot_format": metadata.get("snapshot_format"),
        "hidden_present": metadata.get("hidden_present"),
        "hidden_sequence_len": metadata.get("hidden_sequence_len"),
        "native_tokens_csv": metadata.get("native_tokens.csv"),
        "last_step_greedy_token": metadata.get("last_step.greedy_token"),
    }
    last_hidden = tensor_headers.get("hidden.last")
    if last_hidden is not None:
        summary["last_hidden_tensor"] = {
            "key": "hidden.last",
            "shape": last_hidden.get("shape"),
            "dtype": last_hidden.get("dtype"),
        }
    summary["tap_tensors"] = tap_tensors
    return summary


def safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(header_len))


def export_native_tap_reference_fixture(
    *,
    args: argparse.Namespace,
    native_taps: dict[str, Any],
    reference_output: Path,
) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file
    from transformers import DynamicCache

    from deepspec.eval.dspark.draft_ops import forward_dspark_draft_block
    from deepspec.modeling.dspark.gemma4 import Gemma4DSparkModel

    device = torch.device(args.device)
    model = load_deepspec_dspark_model(Gemma4DSparkModel, args.draft_path, device, torch)
    results = []
    for snapshot in native_taps["snapshots"]:
        tensors = load_file(snapshot["snapshot_path"], device=str(device))
        tap_arrays = [
            tensors[f"dspark_context.tap_{index}.hidden"].to(device=device, dtype=torch.bfloat16)
            for index in range(len(EXPECTED_TARGET_LAYER_IDS))
        ]
        target_last_hidden = tensors.get("hidden.last")
        if target_last_hidden is not None:
            target_last_hidden = target_last_hidden.to(device=device, dtype=torch.bfloat16)
        target_hidden_states = torch.cat(tap_arrays, dim=-1)
        block_size = int(args.block_size)
        draft_input_ids = torch.full(
            (1, block_size),
            int(model.mask_token_id),
            dtype=torch.long,
            device=device,
        )
        draft_input_ids[:, 0] = int(snapshot["prefill_greedy_token"])
        context_tokens = int(snapshot["context_tokens"])
        position_ids = torch.arange(context_tokens + block_size, device=device).unsqueeze(0)
        with torch.inference_mode():
            block_hidden = forward_dspark_draft_block(
                model,
                draft_input_ids=draft_input_ids,
                position_ids=position_ids,
                past_key_values_draft=DynamicCache(),
                target_hidden_states=target_hidden_states,
                start=context_tokens,
                block_size=block_size,
            )
            proposal_hidden = block_hidden[:, :block_size, :]
            base_logits = model.compute_logits(proposal_hidden).float()
            greedy_tokens, markov_logits = model.sample_draft_tokens(
                base_logits,
                first_prev_token_ids=draft_input_ids[:, 0],
                temperature=0.0,
                hidden_states=proposal_hidden,
            )
            markov_logits = markov_logits.float()
            prev_token_ids = torch.cat([draft_input_ids[:, :1], greedy_tokens[:, :-1]], dim=1)
            confidence_logits = model.predict_confidence_step(
                proposal_hidden,
                prev_token_ids=prev_token_ids,
            )
            confidence = None
            if confidence_logits is not None:
                confidence_logits = confidence_logits.float().reshape(
                    confidence_logits.shape[0],
                    block_size,
                    -1,
                )[:, :, 0]
                confidence = torch.sigmoid(confidence_logits)

        result: dict[str, Any] = {
            "workload_id": snapshot["workload_id"],
            "input_token_ids": snapshot["prompt_tokens"],
            "anchor_token_id": int(snapshot["prefill_greedy_token"]),
            "target_hidden_source": snapshot["snapshot_path"],
            "target_layer_ids": EXPECTED_TARGET_LAYER_IDS,
            "target_hidden_taps": [tensor_to_list(tap.float()) for tap in tap_arrays],
            "target_last_hidden": tensor_to_list(
                target_last_hidden.float() if target_last_hidden is not None else None
            ),
            "block_size": block_size,
            "greedy_draft_tokens": greedy_tokens[0].detach().cpu().tolist(),
            "dspark_base_top_k": top_k_payload(base_logits, args.top_k, torch),
            "dspark_markov_top_k": top_k_payload(markov_logits, args.top_k, torch),
            "dspark_base_selected_logits": gather_logits(base_logits, greedy_tokens, torch),
            "dspark_markov_selected_logits": gather_logits(markov_logits, greedy_tokens, torch),
            "confidence_logits": tensor_to_list(confidence_logits),
            "confidence": tensor_to_list(confidence),
        }
        if args.include_full_logits:
            result["dspark_base_logits"] = tensor_to_list(base_logits)
            result["dspark_markov_logits"] = tensor_to_list(markov_logits)
        results.append(result)

    fixture = {
        "schema_version": 1,
        "goal": "XR60-dspark-native-mlx",
        "phase": "01-reference-fixtures",
        "reference_mode": "deepspec_native_tap_snapshot",
        "reference_revisions": reference_revisions(),
        "draft_path": str(args.draft_path),
        "native_tap_manifest": str(args.native_tap_manifest),
        "device": args.device,
        "include_full_logits": args.include_full_logits,
        "top_k": args.top_k,
        "fixtures": results,
    }
    write_json(reference_output, fixture)
    return {
        "path": str(reference_output),
        "fixture_count": len(results),
        "workload_ids": [result["workload_id"] for result in results],
        "fields": sorted(results[0].keys()) if results else [],
    }


def load_deepspec_dspark_model(model_cls: Any, draft_path: Path, device: Any, torch_module: Any) -> Any:
    try:
        model = model_cls.from_pretrained(
            str(draft_path),
            dtype=torch_module.bfloat16,
            attn_implementation="sdpa",
        )
    except TypeError:
        model = model_cls.from_pretrained(
            str(draft_path),
            torch_dtype=torch_module.bfloat16,
            attn_implementation="sdpa",
        )
    return model.to(device).eval()


def top_k_payload(logits: Any, top_k: int, torch_module: Any) -> dict[str, Any]:
    k = min(int(top_k), int(logits.shape[-1]))
    values, indices = torch_module.topk(logits, k=k, dim=-1)
    return {
        "token_ids": indices.detach().cpu().tolist(),
        "logits": values.detach().cpu().tolist(),
    }


def gather_logits(logits: Any, token_ids: Any, torch_module: Any) -> list[Any]:
    selected = torch_module.gather(logits, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    return selected.detach().cpu().tolist()


def tensor_to_list(tensor: Any | None) -> Any | None:
    if tensor is None:
        return None
    return tensor.detach().cpu().tolist()


if __name__ == "__main__":
    raise SystemExit(main())
