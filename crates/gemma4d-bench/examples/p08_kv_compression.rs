use std::{
    collections::BTreeMap,
    env, fs,
    io::Write,
    num::NonZeroU32,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use gemma4d_ffi::{
    KvCache, KvMode, KvPolicy, KvSnapshot, LoadConfig, Target, decode_one, prefill, runtime_version,
};
use gemma4d_tokenizer::{file_sha256, sha256_hex};
use serde::{Deserialize, Serialize};

const DEFAULT_MODEL: &str = "artifacts/models/gemma-4-12B-it-4bit";
const DEFAULT_OUT_DIR: &str = "benchmarks/out/P08-kv-compression";
const MODE: &str = "native_kv_prefix_payload_compression";
const EXACT_LOGIT_TOLERANCE: f64 = 0.000_001;
const Q8_MAX_GREEDY_LOGIT_DELTA: f64 = 0.5;
const Q4_MAX_GREEDY_LOGIT_DELTA: f64 = 2.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    fs::create_dir_all(&args.out_dir)?;

    let run_id = run_id();
    let records_path = args.out_dir.join("records.jsonl");
    let summary_path = args.out_dir.join("summary.json");
    let report_path = args.out_dir.join("report.md");
    let blockers_path = args.out_dir.join("blockers.md");
    let environment = capture_environment();
    let model_identity = capture_model_identity(&args.model_path);
    let mut blockers = startup_blockers(&args);
    let mut records = Vec::new();

    let model_load_ms = if blockers.is_empty() {
        let load_started = Instant::now();
        let target = Target::load(&target_config(&args))?;
        let model_load_ms = duration_ms(load_started.elapsed());
        for context_tokens in &args.contexts {
            records.push(run_context(&args, &target, &run_id, *context_tokens)?);
        }
        Some(model_load_ms)
    } else {
        None
    };

    blockers.extend(blockers_for_records(&records, &args.contexts));
    let status = if blockers.is_empty() {
        "passed"
    } else {
        "failed"
    };

    let summary = P08Summary {
        schema_version: 1,
        goal: "P08-kv-compression",
        status,
        run_id,
        timestamp_unix: unix_now(),
        mode: MODE,
        model_path: args.model_path.display().to_string(),
        out_dir: args.out_dir.display().to_string(),
        model_load_ms,
        records_path: records_path.display().to_string(),
        summary_path: summary_path.display().to_string(),
        report_path: report_path.display().to_string(),
        blockers_path: blockers_path.display().to_string(),
        contexts: args.contexts.clone(),
        max_context_tokens: args.max_context_tokens,
        exact_logit_tolerance: EXACT_LOGIT_TOLERANCE,
        q8_max_greedy_logit_delta: Q8_MAX_GREEDY_LOGIT_DELTA,
        q4_max_greedy_logit_delta: Q4_MAX_GREEDY_LOGIT_DELTA,
        environment,
        relevant_environment: capture_relevant_environment(),
        model_identity,
        default_recommendation: "keep_compressed_active_decode_disabled",
        planar_iso: PlanarIsoReport::default_disabled(),
        claims: claim_inventory(&records),
        records,
        blockers,
        measurement_notes: vec![
            "cold_ttft_ms measures native BF16 prefill plus KV materialization for the full prefix",
            "warm_restore_ms measures payload load, transparent decompression if needed, snapshot import, and cached last-step retrieval",
            "q8/q4 compression is applied only to global/full-attention KV tensors; sliding-window KV tensors and hidden state remain BF16",
            "continued_decode compares one decode_one call after restore against the cold BF16 continuation and is the quality gate that exercises restored KV tensors",
            "q8/q4 quality gate failures are reportable evidence and do not fail the benchmark unless the mode cannot be measured",
            "restored last-step logits are snapshot metadata and are recorded only as an import sanity check",
            "payload_memory_reduction is measured from actual safetensors payload bytes on disk, not an estimate",
            "active_kv_memory_reduction is expected to be zero because compressed SSD payloads are decompressed before active decode",
            "compressed active decode remains disabled by default",
            "Planar/Iso candidates remain feature-disabled by default and are not reportable in P08 without real evidence",
        ],
    };

    write_jsonl(&records_path, &summary.records)?;
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)?;
    fs::write(&report_path, render_report(&summary))?;
    fs::write(&blockers_path, render_blockers(&summary))?;

    println!("P08 KV compression: {}", summary.status);
    println!("records: {}", records_path.display());
    println!("summary: {}", summary_path.display());
    println!("report: {}", report_path.display());
    println!("blockers: {}", blockers_path.display());

    if summary.status == "failed" {
        Err("P08 KV compression checks failed".into())
    } else {
        Ok(())
    }
}

#[derive(Debug)]
struct Args {
    out_dir: PathBuf,
    model_path: PathBuf,
    contexts: Vec<usize>,
    max_context_tokens: usize,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut out_dir = PathBuf::from(DEFAULT_OUT_DIR);
        let mut model_path = PathBuf::from(DEFAULT_MODEL);
        let mut contexts = vec![4096, 8192, 16_384];
        let mut max_context_tokens = 32_768;

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--out-dir" => {
                    out_dir = args
                        .next()
                        .map(PathBuf::from)
                        .ok_or("--out-dir requires a path")?;
                }
                "--model-path" => {
                    model_path = args
                        .next()
                        .map(PathBuf::from)
                        .ok_or("--model-path requires a path")?;
                }
                "--contexts" => {
                    let value = args.next().ok_or("--contexts requires a comma list")?;
                    contexts = parse_contexts(&value)?;
                }
                "--max-context-tokens" => {
                    let value = args.next().ok_or("--max-context-tokens requires a value")?;
                    max_context_tokens = parse_positive_usize(&value, "--max-context-tokens")?;
                }
                "-h" | "--help" => {
                    println!(
                        "usage: GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example p08_kv_compression -- [--out-dir PATH] [--model-path PATH] [--contexts 4096,8192,16384] [--max-context-tokens N]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown option '{other}'").into()),
            }
        }

        if contexts.is_empty() {
            return Err("--contexts must include at least one context".into());
        }
        contexts.sort_unstable();
        contexts.dedup();
        if contexts.contains(&0) {
            return Err("--contexts values must be > 0".into());
        }
        if contexts.iter().any(|context| *context > max_context_tokens) {
            return Err("--contexts cannot exceed --max-context-tokens".into());
        }

        Ok(Self {
            out_dir,
            model_path,
            contexts,
            max_context_tokens,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct P08Summary {
    schema_version: u32,
    goal: &'static str,
    status: &'static str,
    run_id: String,
    timestamp_unix: u64,
    mode: &'static str,
    model_path: String,
    out_dir: String,
    model_load_ms: Option<f64>,
    records_path: String,
    summary_path: String,
    report_path: String,
    blockers_path: String,
    contexts: Vec<usize>,
    max_context_tokens: usize,
    exact_logit_tolerance: f64,
    q8_max_greedy_logit_delta: f64,
    q4_max_greedy_logit_delta: f64,
    environment: Environment,
    relevant_environment: BTreeMap<String, Option<String>>,
    model_identity: ModelIdentity,
    default_recommendation: &'static str,
    planar_iso: PlanarIsoReport,
    claims: ClaimInventory,
    records: Vec<P08Record>,
    blockers: Vec<String>,
    measurement_notes: Vec<&'static str>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct P08Record {
    schema_version: u32,
    goal: &'static str,
    run_id: String,
    timestamp_unix: u64,
    context_tokens: usize,
    prompt_token_id: i32,
    mode: &'static str,
    cold: ColdPrefill,
    baseline_decode: BaselineDecode,
    modes: Vec<ModeRecord>,
    gate: ContextGate,
    blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ColdPrefill {
    ttft_ms: f64,
    greedy_token: i32,
    greedy_logit: f32,
    sequence_len: u64,
    active_kv_bytes: u64,
    peak_memory_gb: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BaselineDecode {
    decode_ms: f64,
    greedy_token: i32,
    greedy_logit: f32,
    sequence_len: u64,
    active_kv_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModeRecord {
    cache_mode: &'static str,
    payload_path: String,
    payload_sha256: String,
    payload_bytes: u64,
    payload_save_ms: f64,
    payload_load_ms: f64,
    snapshot_import_last_step_ms: f64,
    warm_restore_ms: f64,
    compress_global_layers: bool,
    compress_sliding_layers: bool,
    active_compressed_decode_enabled: bool,
    compressed_full_attention_only: bool,
    restored_last_step: RestoredLastStep,
    continued_decode: ContinuedDecode,
    memory: MemoryComparison,
    quality_gate: QualityGate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RestoredLastStep {
    greedy_token: i32,
    greedy_logit: f32,
    token_agreement: bool,
    greedy_logit_delta: f64,
    sequence_len_parity: bool,
    active_kv_bytes_parity: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ContinuedDecode {
    decode_ms: f64,
    greedy_token: i32,
    greedy_logit: f32,
    greedy_agreement: bool,
    greedy_logit_delta: f64,
    sequence_len_parity: bool,
    active_kv_bytes_parity: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MemoryComparison {
    bf16_payload_bytes: u64,
    compressed_payload_bytes: u64,
    payload_delta_bytes: i64,
    payload_memory_reduction: f64,
    bf16_active_kv_bytes: u64,
    restored_active_kv_bytes: u64,
    active_kv_delta_bytes: i64,
    active_kv_memory_reduction: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QualityGate {
    passed: bool,
    threshold_greedy_logit_delta: f64,
    greedy_agreement: bool,
    greedy_logit_delta_within_threshold: bool,
    payload_smaller_than_bf16: bool,
    active_decode_remains_bf16: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ContextGate {
    passed: bool,
    bf16_exact_restore: bool,
    q8_measured: bool,
    q4_measured: bool,
    q8_payload_smaller: bool,
    q4_payload_smaller: bool,
    compressed_active_decode_disabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClaimInventory {
    exactness: Vec<String>,
    quality: Vec<String>,
    memory: Vec<String>,
    latency: Vec<String>,
    defaults: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PlanarIsoReport {
    feature_enabled: bool,
    accepted_by_default: bool,
    status: &'static str,
    candidates: Vec<&'static str>,
}

impl PlanarIsoReport {
    fn default_disabled() -> Self {
        Self {
            feature_enabled: false,
            accepted_by_default: false,
            status: "feature_disabled_default",
            candidates: vec!["planar4", "planar3", "iso4", "iso3"],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Environment {
    machine: String,
    macos: String,
    rustc: String,
    cargo: String,
    runtime_backend: String,
    runtime_backend_version: String,
    git_commit: String,
    git_status_short: String,
    hw_memsize_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelIdentity {
    model_path: String,
    exists: bool,
    configured_revision: String,
    config_sha256: String,
    tokenizer_sha256: String,
    tokenizer_config_sha256: String,
    chat_template_sha256: String,
    safetensors_inventory_sha256: String,
    safetensors_file_count: usize,
    safetensors_total_bytes: u64,
}

struct SafetensorsInventory {
    inventory_sha256: String,
    file_count: usize,
    total_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
enum BenchMode {
    Bf16,
    Q8,
    Q4,
}

impl BenchMode {
    fn all() -> [Self; 3] {
        [Self::Bf16, Self::Q8, Self::Q4]
    }

    fn label(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Q8 => "mlx_affine_q8",
            Self::Q4 => "mlx_affine_q4",
        }
    }

    fn ffi_mode(self) -> KvMode {
        match self {
            Self::Bf16 => KvMode::Bf16,
            Self::Q8 => KvMode::MlxAffineQ8,
            Self::Q4 => KvMode::MlxAffineQ4,
        }
    }

    fn max_logit_delta(self) -> f64 {
        match self {
            Self::Bf16 => EXACT_LOGIT_TOLERANCE,
            Self::Q8 => Q8_MAX_GREEDY_LOGIT_DELTA,
            Self::Q4 => Q4_MAX_GREEDY_LOGIT_DELTA,
        }
    }

    fn compresses_full_attention(self) -> bool {
        !matches!(self, Self::Bf16)
    }
}

fn run_context(
    args: &Args,
    target: &Target,
    run_id: &str,
    context_tokens: usize,
) -> Result<P08Record, Box<dyn std::error::Error>> {
    let prompt_tokens = vec![9259_i32; context_tokens];
    let context_dir = args.out_dir.join(format!("{context_tokens}-tokens"));
    if context_dir.exists() {
        fs::remove_dir_all(&context_dir)?;
    }
    fs::create_dir_all(&context_dir)?;

    let mut cold_cache = KvCache::create(&KvPolicy::default())?;
    let cold_started = Instant::now();
    let cold_step = prefill(target, &mut cold_cache, &prompt_tokens)?;
    let cold_ttft = duration_ms(cold_started.elapsed());
    let snapshot = cold_cache.export_snapshot()?;

    let mut mode_records = Vec::new();
    for mode in BenchMode::all() {
        mode_records.push(write_mode_payload(
            &context_dir,
            &snapshot,
            mode,
            cold_step.active_kv_bytes,
        )?);
    }
    let bf16_payload_bytes = mode_records
        .iter()
        .find(|record| record.cache_mode == "bf16")
        .map(|record| record.payload_bytes)
        .ok_or("BF16 mode record missing")?;

    let baseline_started = Instant::now();
    let baseline_next = decode_one(target, &mut cold_cache, cold_step.greedy_token)?;
    let baseline_decode = BaselineDecode {
        decode_ms: duration_ms(baseline_started.elapsed()),
        greedy_token: baseline_next.greedy_token,
        greedy_logit: baseline_next.greedy_logit,
        sequence_len: baseline_next.sequence_len,
        active_kv_bytes: baseline_next.active_kv_bytes,
    };

    for record in &mut mode_records {
        restore_and_score_mode(
            target,
            record,
            &cold_step,
            &baseline_decode,
            bf16_payload_bytes,
        )?;
    }

    let gate = context_gate(&mode_records);
    let blockers = blockers_for_context(context_tokens, &mode_records, &gate);
    Ok(P08Record {
        schema_version: 1,
        goal: "P08-kv-compression",
        run_id: run_id.to_owned(),
        timestamp_unix: unix_now(),
        context_tokens,
        prompt_token_id: 9259,
        mode: MODE,
        cold: ColdPrefill {
            ttft_ms: cold_ttft,
            greedy_token: cold_step.greedy_token,
            greedy_logit: cold_step.greedy_logit,
            sequence_len: cold_step.sequence_len,
            active_kv_bytes: cold_step.active_kv_bytes,
            peak_memory_gb: cold_step.peak_memory_gb,
        },
        baseline_decode,
        modes: mode_records,
        gate,
        blockers,
    })
}

fn write_mode_payload(
    context_dir: &Path,
    snapshot: &KvSnapshot,
    mode: BenchMode,
    bf16_active_kv_bytes: u64,
) -> Result<ModeRecord, Box<dyn std::error::Error>> {
    let payload_path = context_dir.join(format!("{}.safetensors", mode.label()));
    let save_started = Instant::now();
    if matches!(mode, BenchMode::Bf16) {
        snapshot.save_to_path(&payload_path)?;
    } else {
        snapshot.save_compressed_to_path(&payload_path, mode.ffi_mode(), true, false)?;
    }
    let payload_save_ms = duration_ms(save_started.elapsed());
    let payload_bytes = fs::metadata(&payload_path)?.len();
    let payload_sha256 = file_sha256(&payload_path)?;

    Ok(ModeRecord {
        cache_mode: mode.label(),
        payload_path: payload_path.display().to_string(),
        payload_sha256,
        payload_bytes,
        payload_save_ms,
        payload_load_ms: 0.0,
        snapshot_import_last_step_ms: 0.0,
        warm_restore_ms: 0.0,
        compress_global_layers: mode.compresses_full_attention(),
        compress_sliding_layers: false,
        active_compressed_decode_enabled: false,
        compressed_full_attention_only: mode.compresses_full_attention(),
        restored_last_step: RestoredLastStep {
            greedy_token: 0,
            greedy_logit: 0.0,
            token_agreement: false,
            greedy_logit_delta: f64::INFINITY,
            sequence_len_parity: false,
            active_kv_bytes_parity: false,
        },
        continued_decode: ContinuedDecode {
            decode_ms: 0.0,
            greedy_token: 0,
            greedy_logit: 0.0,
            greedy_agreement: false,
            greedy_logit_delta: f64::INFINITY,
            sequence_len_parity: false,
            active_kv_bytes_parity: false,
        },
        memory: MemoryComparison {
            bf16_payload_bytes: payload_bytes,
            compressed_payload_bytes: payload_bytes,
            payload_delta_bytes: 0,
            payload_memory_reduction: 0.0,
            bf16_active_kv_bytes,
            restored_active_kv_bytes: 0,
            active_kv_delta_bytes: 0,
            active_kv_memory_reduction: 0.0,
        },
        quality_gate: QualityGate {
            passed: false,
            threshold_greedy_logit_delta: mode.max_logit_delta(),
            greedy_agreement: false,
            greedy_logit_delta_within_threshold: false,
            payload_smaller_than_bf16: matches!(mode, BenchMode::Bf16),
            active_decode_remains_bf16: true,
        },
    })
}

fn restore_and_score_mode(
    target: &Target,
    record: &mut ModeRecord,
    cold_step: &gemma4d_ffi::StepResult,
    baseline_decode: &BaselineDecode,
    bf16_payload_bytes: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let mode = match record.cache_mode {
        "bf16" => BenchMode::Bf16,
        "mlx_affine_q8" => BenchMode::Q8,
        "mlx_affine_q4" => BenchMode::Q4,
        other => return Err(format!("unsupported mode record {other}").into()),
    };

    let load_started = Instant::now();
    let loaded_snapshot = KvSnapshot::load_from_path(&record.payload_path)?;
    record.payload_load_ms = duration_ms(load_started.elapsed());

    let mut restored_cache = KvCache::create(&policy_for_mode(mode))?;
    let import_started = Instant::now();
    restored_cache.import_snapshot(&loaded_snapshot)?;
    let restored_last = restored_cache.last_step()?;
    record.snapshot_import_last_step_ms = duration_ms(import_started.elapsed());
    record.warm_restore_ms = record.payload_load_ms + record.snapshot_import_last_step_ms;

    let decode_started = Instant::now();
    let restored_next = decode_one(target, &mut restored_cache, cold_step.greedy_token)?;
    let restored_decode_ms = duration_ms(decode_started.elapsed());

    let last_logit_delta =
        (f64::from(cold_step.greedy_logit) - f64::from(restored_last.greedy_logit)).abs();
    let decode_logit_delta =
        (f64::from(baseline_decode.greedy_logit) - f64::from(restored_next.greedy_logit)).abs();
    let payload_delta = record.payload_bytes as i64 - bf16_payload_bytes as i64;
    let payload_memory_reduction = if bf16_payload_bytes == 0 {
        0.0
    } else {
        1.0 - (record.payload_bytes as f64 / bf16_payload_bytes as f64)
    };
    let active_delta = restored_last.active_kv_bytes as i64 - cold_step.active_kv_bytes as i64;
    let active_reduction = if cold_step.active_kv_bytes == 0 {
        0.0
    } else {
        1.0 - (restored_last.active_kv_bytes as f64 / cold_step.active_kv_bytes as f64)
    };

    record.restored_last_step = RestoredLastStep {
        greedy_token: restored_last.greedy_token,
        greedy_logit: restored_last.greedy_logit,
        token_agreement: cold_step.greedy_token == restored_last.greedy_token,
        greedy_logit_delta: last_logit_delta,
        sequence_len_parity: cold_step.sequence_len == restored_last.sequence_len,
        active_kv_bytes_parity: cold_step.active_kv_bytes == restored_last.active_kv_bytes,
    };
    record.continued_decode = ContinuedDecode {
        decode_ms: restored_decode_ms,
        greedy_token: restored_next.greedy_token,
        greedy_logit: restored_next.greedy_logit,
        greedy_agreement: baseline_decode.greedy_token == restored_next.greedy_token,
        greedy_logit_delta: decode_logit_delta,
        sequence_len_parity: baseline_decode.sequence_len == restored_next.sequence_len,
        active_kv_bytes_parity: baseline_decode.active_kv_bytes == restored_next.active_kv_bytes,
    };
    record.memory = MemoryComparison {
        bf16_payload_bytes,
        compressed_payload_bytes: record.payload_bytes,
        payload_delta_bytes: payload_delta,
        payload_memory_reduction,
        bf16_active_kv_bytes: cold_step.active_kv_bytes,
        restored_active_kv_bytes: restored_last.active_kv_bytes,
        active_kv_delta_bytes: active_delta,
        active_kv_memory_reduction: active_reduction,
    };
    let payload_smaller_than_bf16 =
        matches!(mode, BenchMode::Bf16) || record.payload_bytes < bf16_payload_bytes;
    let greedy_logit_delta_within_threshold = decode_logit_delta <= mode.max_logit_delta();
    record.quality_gate = QualityGate {
        passed: record.continued_decode.greedy_agreement
            && greedy_logit_delta_within_threshold
            && payload_smaller_than_bf16
            && !record.active_compressed_decode_enabled,
        threshold_greedy_logit_delta: mode.max_logit_delta(),
        greedy_agreement: record.continued_decode.greedy_agreement,
        greedy_logit_delta_within_threshold,
        payload_smaller_than_bf16,
        active_decode_remains_bf16: !record.active_compressed_decode_enabled,
    };

    Ok(())
}

fn policy_for_mode(mode: BenchMode) -> KvPolicy {
    let mut policy = KvPolicy::default();
    policy.ssd_prefix_mode = mode.ffi_mode();
    policy.compress_global_layers = mode.compresses_full_attention();
    policy.compress_sliding_layers = false;
    policy.allow_active_compressed_decode = false;
    policy
}

fn context_gate(records: &[ModeRecord]) -> ContextGate {
    let bf16 = records.iter().find(|record| record.cache_mode == "bf16");
    let q8 = records
        .iter()
        .find(|record| record.cache_mode == "mlx_affine_q8");
    let q4 = records
        .iter()
        .find(|record| record.cache_mode == "mlx_affine_q4");
    let bf16_exact_restore = bf16.is_some_and(|record| {
        record.restored_last_step.token_agreement
            && record.restored_last_step.greedy_logit_delta <= EXACT_LOGIT_TOLERANCE
            && record.continued_decode.greedy_agreement
            && record.continued_decode.greedy_logit_delta <= EXACT_LOGIT_TOLERANCE
    });
    let q8_payload_smaller = q8.is_some_and(|record| record.memory.payload_delta_bytes < 0);
    let q4_payload_smaller = q4.is_some_and(|record| record.memory.payload_delta_bytes < 0);
    let compressed_active_decode_disabled = records
        .iter()
        .all(|record| !record.active_compressed_decode_enabled);
    ContextGate {
        passed: bf16_exact_restore
            && q8.is_some()
            && q4.is_some()
            && q8_payload_smaller
            && q4_payload_smaller
            && compressed_active_decode_disabled,
        bf16_exact_restore,
        q8_measured: q8.is_some(),
        q4_measured: q4.is_some(),
        q8_payload_smaller,
        q4_payload_smaller,
        compressed_active_decode_disabled,
    }
}

fn blockers_for_context(
    context_tokens: usize,
    records: &[ModeRecord],
    gate: &ContextGate,
) -> Vec<String> {
    let mut blockers = Vec::new();
    if !gate.bf16_exact_restore {
        blockers.push(format!(
            "{context_tokens} tokens: BF16 snapshot restore was not exact"
        ));
    }
    for record in records {
        if matches!(record.cache_mode, "mlx_affine_q8" | "mlx_affine_q4")
            && !record.quality_gate.payload_smaller_than_bf16
        {
            blockers.push(format!(
                "{context_tokens} tokens {}: compressed payload was not smaller than BF16",
                record.cache_mode
            ));
        }
    }
    blockers
}

fn blockers_for_records(records: &[P08Record], contexts: &[usize]) -> Vec<String> {
    let mut blockers = Vec::new();
    for context in contexts {
        if !records
            .iter()
            .any(|record| record.context_tokens == *context)
        {
            blockers.push(format!("{context} tokens: missing benchmark record"));
        }
    }
    for record in records {
        blockers.extend(record.blockers.iter().cloned());
    }
    blockers
}

fn claim_inventory(records: &[P08Record]) -> ClaimInventory {
    let mut exactness = Vec::new();
    let mut quality = Vec::new();
    let mut memory = Vec::new();
    let mut latency = Vec::new();
    let defaults = vec![
        "compressed active decode remains disabled by default".to_owned(),
        "Planar/Iso candidates remain feature-disabled without real P08 evidence".to_owned(),
    ];

    for record in records {
        exactness.push(format!(
            "{} tokens BF16 exact restore={}",
            record.context_tokens, record.gate.bf16_exact_restore
        ));
        latency.push(format!(
            "{} tokens cold TTFT {:.3} ms, BF16 warm restore {:.3} ms, q8 {:.3} ms, q4 {:.3} ms",
            record.context_tokens,
            record.cold.ttft_ms,
            mode_record(record, "bf16").map_or(0.0, |mode| mode.warm_restore_ms),
            mode_record(record, "mlx_affine_q8").map_or(0.0, |mode| mode.warm_restore_ms),
            mode_record(record, "mlx_affine_q4").map_or(0.0, |mode| mode.warm_restore_ms),
        ));
        for mode in &record.modes {
            quality.push(format!(
                "{} tokens {} greedy_agreement={} greedy_logit_delta={:.6} gate={}",
                record.context_tokens,
                mode.cache_mode,
                mode.continued_decode.greedy_agreement,
                mode.continued_decode.greedy_logit_delta,
                mode.quality_gate.passed
            ));
            memory.push(format!(
                "{} tokens {} payload reduction {:.3}% active reduction {:.3}%",
                record.context_tokens,
                mode.cache_mode,
                mode.memory.payload_memory_reduction * 100.0,
                mode.memory.active_kv_memory_reduction * 100.0
            ));
        }
    }

    ClaimInventory {
        exactness,
        quality,
        memory,
        latency,
        defaults,
    }
}

fn mode_record<'a>(record: &'a P08Record, cache_mode: &str) -> Option<&'a ModeRecord> {
    record
        .modes
        .iter()
        .find(|mode| mode.cache_mode == cache_mode)
}

fn render_report(summary: &P08Summary) -> String {
    let mut out = String::new();
    out.push_str("# P08 KV Compression\n\n");
    out.push_str(&format!("Status: `{}`\n\n", summary.status));
    out.push_str("## Run\n\n");
    out.push_str("| Field | Value |\n|---|---|\n");
    out.push_str(&format!("| Run ID | `{}` |\n", summary.run_id));
    out.push_str(&format!("| Mode | `{}` |\n", summary.mode));
    out.push_str(&format!("| Model path | `{}` |\n", summary.model_path));
    out.push_str(&format!(
        "| Model load ms | `{}` |\n",
        option_ms(summary.model_load_ms)
    ));
    out.push_str(&format!(
        "| Runtime | `{}` `{}` |\n",
        summary.environment.runtime_backend, summary.environment.runtime_backend_version
    ));
    out.push_str(&format!("| Git | `{}` |\n", summary.environment.git_commit));
    out.push_str(&format!(
        "| Default recommendation | `{}` |\n\n",
        summary.default_recommendation
    ));

    out.push_str("## Results\n\n");
    out.push_str("| Context | Mode | Gate | Greedy Agree | Logit Delta | Payload MiB | Payload Reduction | Warm Restore ms | Decode ms | Active KV Reduction |\n");
    out.push_str("|---:|---|---|---|---:|---:|---:|---:|---:|---:|\n");
    for record in &summary.records {
        for mode in &record.modes {
            out.push_str(&format!(
                "| {} | `{}` | `{}` | `{}` | {:.6} | {:.3} | {:.3}% | {:.3} | {:.3} | {:.3}% |\n",
                record.context_tokens,
                mode.cache_mode,
                mode.quality_gate.passed,
                mode.continued_decode.greedy_agreement,
                mode.continued_decode.greedy_logit_delta,
                mode.payload_bytes as f64 / 1_048_576.0,
                mode.memory.payload_memory_reduction * 100.0,
                mode.warm_restore_ms,
                mode.continued_decode.decode_ms,
                mode.memory.active_kv_memory_reduction * 100.0,
            ));
        }
    }

    out.push_str("\n## Context Gates\n\n");
    out.push_str("| Context | BF16 Exact | q8 Measured | q4 Measured | q8 Smaller | q4 Smaller | Active Decode Disabled |\n");
    out.push_str("|---:|---|---|---|---|---|---|\n");
    for record in &summary.records {
        out.push_str(&format!(
            "| {} | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |\n",
            record.context_tokens,
            record.gate.bf16_exact_restore,
            record.gate.q8_measured,
            record.gate.q4_measured,
            record.gate.q8_payload_smaller,
            record.gate.q4_payload_smaller,
            record.gate.compressed_active_decode_disabled,
        ));
    }

    out.push_str("\n## Verification Command\n\n```sh\n");
    out.push_str("GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example p08_kv_compression -- --out-dir benchmarks/out/P08-kv-compression --model-path artifacts/models/gemma-4-12B-it-4bit\n");
    out.push_str("```\n\n## Notes\n\n");
    for note in &summary.measurement_notes {
        out.push_str(&format!("- {note}.\n"));
    }
    out
}

fn render_blockers(summary: &P08Summary) -> String {
    if summary.blockers.is_empty() {
        return "# P08 Blockers\n\nNo blockers recorded.\n".to_owned();
    }
    let mut out = "# P08 Blockers\n\n".to_owned();
    for blocker in &summary.blockers {
        out.push_str(&format!("- {blocker}\n"));
    }
    out
}

fn write_jsonl(path: &Path, records: &[P08Record]) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = fs::File::create(path)?;
    for record in records {
        serde_json::to_writer(&mut file, record)?;
        file.write_all(b"\n")?;
    }
    Ok(())
}

fn startup_blockers(args: &Args) -> Vec<String> {
    let mut blockers = Vec::new();
    if !args.model_path.exists() {
        blockers.push(format!(
            "model path does not exist: {}",
            args.model_path.display()
        ));
    }
    if env::var("GEMMA4D_REQUIRE_MLX").ok().as_deref() != Some("1") {
        blockers.push("GEMMA4D_REQUIRE_MLX=1 is required for real native P08 evidence".to_owned());
    }
    if env::var("GEMMA4D_USE_NATIVE_GRAPH").ok().as_deref() != Some("1") {
        blockers
            .push("GEMMA4D_USE_NATIVE_GRAPH=1 is required for real native P08 evidence".to_owned());
    }
    blockers
}

fn target_config(args: &Args) -> LoadConfig {
    LoadConfig {
        model_path: args.model_path.display().to_string(),
        model_id: Some("mlx-community/gemma-4-12B-it-4bit".to_owned()),
        model_revision: env::var("GEMMA4D_MODEL_REVISION").ok(),
        expected_architecture: Some("gemma4".to_owned()),
        max_context_tokens: NonZeroU32::new(args.max_context_tokens as u32)
            .expect("max_context_tokens is non-zero"),
        allow_unsupported_config: false,
    }
}

fn capture_environment() -> Environment {
    let runtime = runtime_version().ok();
    Environment {
        machine: command_output("uname", &["-m"]),
        macos: command_output("sw_vers", &["-productVersion"]),
        rustc: command_output("rustc", &["--version"]),
        cargo: command_output("cargo", &["--version"]),
        runtime_backend: runtime
            .as_ref()
            .map(|version| version.backend_name.clone())
            .unwrap_or_else(|| "unavailable".to_owned()),
        runtime_backend_version: runtime
            .as_ref()
            .map(|version| version.backend_version.clone())
            .unwrap_or_else(|| "unavailable".to_owned()),
        git_commit: command_output("git", &["rev-parse", "--short", "HEAD"]),
        git_status_short: command_output_allow_empty("git", &["status", "--short"]),
        hw_memsize_bytes: sysctl_hw_memsize(),
    }
}

fn capture_relevant_environment() -> BTreeMap<String, Option<String>> {
    [
        "GEMMA4D_FULL_MODEL_TESTS",
        "GEMMA4D_MLX_LM_PYTHON",
        "GEMMA4D_MODEL_PATH",
        "GEMMA4D_MODEL_REVISION",
        "GEMMA4D_REQUIRE_MLX",
        "GEMMA4D_USE_NATIVE_GRAPH",
        "RUSTFLAGS",
    ]
    .into_iter()
    .map(|key| (key.to_owned(), env::var(key).ok()))
    .collect()
}

fn capture_model_identity(model_path: &Path) -> ModelIdentity {
    let inventory = safetensors_inventory(model_path).unwrap_or(SafetensorsInventory {
        inventory_sha256: format!("unavailable:{}", model_path.display()),
        file_count: 0,
        total_bytes: 0,
    });
    ModelIdentity {
        model_path: model_path.display().to_string(),
        exists: model_path.exists(),
        configured_revision: env::var("GEMMA4D_MODEL_REVISION")
            .unwrap_or_else(|_| "unavailable:GEMMA4D_MODEL_REVISION not set".to_owned()),
        config_sha256: file_sha_or_unavailable(&model_path.join("config.json")),
        tokenizer_sha256: file_sha_or_unavailable(&model_path.join("tokenizer.json")),
        tokenizer_config_sha256: file_sha_or_unavailable(&model_path.join("tokenizer_config.json")),
        chat_template_sha256: file_sha_or_unavailable(&model_path.join("chat_template.json")),
        safetensors_inventory_sha256: inventory.inventory_sha256,
        safetensors_file_count: inventory.file_count,
        safetensors_total_bytes: inventory.total_bytes,
    }
}

fn safetensors_inventory(
    model_path: &Path,
) -> Result<SafetensorsInventory, Box<dyn std::error::Error>> {
    let mut files = fs::read_dir(model_path)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("safetensors"))
        .collect::<Vec<_>>();
    files.sort();
    let mut input = Vec::new();
    let mut total_bytes = 0_u64;
    for path in &files {
        let metadata = fs::metadata(path)?;
        total_bytes += metadata.len();
        input.extend_from_slice(
            path.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .as_bytes(),
        );
        input.push(0);
        input.extend_from_slice(file_sha256(path)?.as_bytes());
        input.push(0);
        input.extend_from_slice(metadata.len().to_string().as_bytes());
        input.push(0);
    }
    Ok(SafetensorsInventory {
        inventory_sha256: sha256_hex(&input),
        file_count: files.len(),
        total_bytes,
    })
}

fn file_sha_or_unavailable(path: &Path) -> String {
    file_sha256(path).unwrap_or_else(|error| format!("unavailable:{}: {error}", path.display()))
}

fn command_output(command: &str, args: &[&str]) -> String {
    let value = command_output_allow_empty(command, args);
    if value.is_empty() {
        "unavailable".to_owned()
    } else {
        value
    }
}

fn command_output_allow_empty(command: &str, args: &[&str]) -> String {
    Command::new(command)
        .args(args)
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                Some(String::from_utf8_lossy(&output.stdout).trim().to_owned())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unavailable".to_owned())
}

fn sysctl_hw_memsize() -> Option<u64> {
    let output = command_output("sysctl", &["-n", "hw.memsize"]);
    output.parse::<u64>().ok()
}

fn run_id() -> String {
    format!("p08-{}", unix_now())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn option_ms(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "unavailable".to_owned())
}

fn parse_contexts(value: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    value
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(|item| parse_positive_usize(item, "--contexts"))
        .collect()
}

fn parse_positive_usize(value: &str, name: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{name} must be a positive integer"))?;
    if parsed == 0 {
        return Err(format!("{name} must be > 0").into());
    }
    Ok(parsed)
}
