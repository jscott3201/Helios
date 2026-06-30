use std::{
    collections::BTreeMap,
    env, fs,
    io::Write,
    num::{NonZeroU32, NonZeroU64},
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use gemma4d_ffi::{
    KvCache, KvPolicy, KvSnapshot, LoadConfig, Target, decode_one, prefill, runtime_version,
};
use gemma4d_kv::{
    CacheMode, Error as KvError, KV_LAYOUT_VERSION, KvBlockKey, KvNamespace, PrefillObservation,
    RamPrefixBlock, SsdCacheAccountingSnapshot, SsdPrefixCache, SsdRestorePhase,
};
use gemma4d_tokenizer::{file_sha256, sha256_hex};
use serde::{Deserialize, Serialize};

const DEFAULT_MODEL: &str = "artifacts/models/gemma-4-12B-it-4bit";
const DEFAULT_OUT_DIR: &str = "benchmarks/out/P07-real-ssd-prefix-cache";
const MODE: &str = "native_ssd_prefix_snapshot_payload";
const LOGIT_TOLERANCE: f64 = 0.000_001;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    fs::create_dir_all(&args.out_dir)?;
    fs::create_dir_all(&args.cache_dir)?;

    let run_id = run_id();
    let records_path = args.out_dir.join("records.jsonl");
    let summary_path = args.out_dir.join("summary.json");
    let report_path = args.out_dir.join("report.md");
    let blockers_path = args.out_dir.join("blockers.md");
    let environment = capture_environment();
    let model_identity = capture_model_identity(&args.model_path);
    let mut records = Vec::new();
    let mut blockers = startup_blockers(&args);

    let model_load_ms = if blockers.is_empty() {
        let load_started = Instant::now();
        let target = Target::load(&target_config(&args))?;
        let model_load_ms = duration_ms(load_started.elapsed());
        for context_tokens in &args.contexts {
            records.push(run_context(
                &args,
                &target,
                &model_identity,
                &run_id,
                *context_tokens,
            )?);
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
    let claims = claim_inventory(&records);

    let summary = P07Summary {
        schema_version: 1,
        goal: "P07-real-ssd-prefix-cache",
        status,
        run_id,
        timestamp_unix: unix_now(),
        mode: MODE,
        model_path: args.model_path.display().to_string(),
        cache_dir: args.cache_dir.display().to_string(),
        model_load_ms,
        records_path: records_path.display().to_string(),
        summary_path: summary_path.display().to_string(),
        report_path: report_path.display().to_string(),
        blockers_path: blockers_path.display().to_string(),
        contexts: args.contexts.clone(),
        max_context_tokens: args.max_context_tokens,
        ssd_budget_bytes: args.ssd_budget_bytes,
        logit_tolerance: LOGIT_TOLERANCE,
        environment,
        relevant_environment: capture_relevant_environment(),
        model_identity,
        default_recommendation: "keep_ssd_disabled_by_default_until_more_variance_data",
        claims,
        records,
        blockers,
        measurement_notes: vec![
            "cold_ttft_ms measures native prefill plus KV materialization for the full prefix",
            "warm_ssd_ttft_ms measures SSD metadata restore, payload checksum, safetensors payload load, native snapshot import, and cached last-step retrieval",
            "payload write/export time is reported separately and is not counted as warm SSD TTFT",
            "warm decode parity runs one decode_one call after restore to verify the imported SSD snapshot can continue generation",
            "namespace rejection checks stop at SSD metadata restore and do not read/import the native payload",
            "corruption rejection checks the payload checksum before native safetensors load/import",
            "mid-decode SSD restore is explicitly rejected before disk read/import and mid_decode_fetches must remain zero",
            "SSD remains disabled by default for tiny16 pending broader variance and server integration evidence",
        ],
    };

    write_jsonl(&records_path, &summary.records)?;
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)?;
    fs::write(&report_path, render_report(&summary))?;
    fs::write(&blockers_path, render_blockers(&summary))?;

    println!("P07 real SSD prefix cache: {}", summary.status);
    println!("records: {}", records_path.display());
    println!("summary: {}", summary_path.display());
    println!("report: {}", report_path.display());
    println!("blockers: {}", blockers_path.display());

    if summary.status == "failed" {
        Err("P07 real SSD prefix cache checks failed".into())
    } else {
        Ok(())
    }
}

#[derive(Debug)]
struct Args {
    out_dir: PathBuf,
    cache_dir: PathBuf,
    model_path: PathBuf,
    contexts: Vec<usize>,
    max_context_tokens: usize,
    ssd_budget_bytes: u64,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut out_dir = PathBuf::from(DEFAULT_OUT_DIR);
        let mut cache_dir: Option<PathBuf> = None;
        let mut model_path = PathBuf::from(DEFAULT_MODEL);
        let mut contexts = vec![4096, 8192, 16_384];
        let mut max_context_tokens = 32_768;
        let mut ssd_budget_bytes = 16 * 1024 * 1024 * 1024;

        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--out-dir" => {
                    out_dir = args
                        .next()
                        .map(PathBuf::from)
                        .ok_or("--out-dir requires a path")?;
                }
                "--cache-dir" => {
                    cache_dir = Some(
                        args.next()
                            .map(PathBuf::from)
                            .ok_or("--cache-dir requires a path")?,
                    );
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
                "--ssd-budget-bytes" => {
                    let value = args.next().ok_or("--ssd-budget-bytes requires a value")?;
                    ssd_budget_bytes = value
                        .parse::<u64>()
                        .map_err(|_| "--ssd-budget-bytes must be an integer")?;
                    if ssd_budget_bytes == 0 {
                        return Err("--ssd-budget-bytes must be > 0".into());
                    }
                }
                "-h" | "--help" => {
                    println!(
                        "usage: GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example p07_real_ssd_prefix_cache -- [--out-dir PATH] [--cache-dir PATH] [--model-path PATH] [--contexts 4096,8192,16384] [--max-context-tokens N] [--ssd-budget-bytes N]"
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
        if contexts.iter().any(|context| *context == 0) {
            return Err("--contexts values must be > 0".into());
        }
        if contexts.iter().any(|context| *context > max_context_tokens) {
            return Err("--contexts cannot exceed --max-context-tokens".into());
        }

        let cache_dir = cache_dir.unwrap_or_else(|| out_dir.join("ssd-cache"));
        Ok(Self {
            out_dir,
            cache_dir,
            model_path,
            contexts,
            max_context_tokens,
            ssd_budget_bytes,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct P07Summary {
    schema_version: u32,
    goal: &'static str,
    status: &'static str,
    run_id: String,
    timestamp_unix: u64,
    mode: &'static str,
    model_path: String,
    cache_dir: String,
    model_load_ms: Option<f64>,
    records_path: String,
    summary_path: String,
    report_path: String,
    blockers_path: String,
    contexts: Vec<usize>,
    max_context_tokens: usize,
    ssd_budget_bytes: u64,
    logit_tolerance: f64,
    environment: Environment,
    relevant_environment: BTreeMap<String, Option<String>>,
    model_identity: ModelIdentity,
    default_recommendation: &'static str,
    claims: ClaimInventory,
    records: Vec<P07Record>,
    blockers: Vec<String>,
    measurement_notes: Vec<&'static str>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct P07Record {
    schema_version: u32,
    goal: &'static str,
    run_id: String,
    timestamp_unix: u64,
    context_tokens: usize,
    prompt_token_id: i32,
    mode: &'static str,
    cache_mode: &'static str,
    namespace_hash: String,
    block_id: String,
    metadata_manifest_path: String,
    payload_manifest_path: String,
    payload_path: String,
    cold: ColdPrefill,
    snapshot: SnapshotRecord,
    ssd_write: SsdWriteRecord,
    warm: WarmSsdRestore,
    continued_decode: ContinuedDecode,
    rejection: RejectionRecord,
    accounting: SsdCacheAccountingSnapshot,
    gate: GateOutcome,
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
struct SnapshotRecord {
    sequence_len: u64,
    active_kv_bytes: u64,
    token_count: u64,
    has_last_step: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SsdWriteRecord {
    metadata_write_ms: f64,
    payload_save_ms: f64,
    payload_manifest_write_ms: f64,
    metadata_bytes_written: u64,
    payload_bytes_written: u64,
    payload_sha256: String,
    metadata_manifest_layers: usize,
    manifest_has_shape_metadata: bool,
    manifest_has_per_layer_attention_metadata: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WarmSsdRestore {
    ttft_ms: f64,
    metadata_restore_ms: f64,
    payload_checksum_ms: f64,
    payload_load_ms: f64,
    snapshot_import_last_step_ms: f64,
    greedy_token: i32,
    greedy_logit: f32,
    sequence_len: u64,
    active_kv_bytes: u64,
    token_parity: bool,
    logit_delta: f64,
    sequence_len_parity: bool,
    active_kv_bytes_parity: bool,
    ttft_improvement_ms: f64,
    ttft_speedup: f64,
    payload_bytes_read: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ContinuedDecode {
    baseline_decode_ms: f64,
    restored_decode_ms: f64,
    baseline_greedy_token: i32,
    restored_greedy_token: i32,
    baseline_greedy_logit: f32,
    restored_greedy_logit: f32,
    token_parity: bool,
    logit_delta: f64,
    sequence_len_parity: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RejectionRecord {
    wrong_namespace_rejected: bool,
    wrong_adapter_rejected: bool,
    wrong_cache_mode_rejected: bool,
    payload_corruption_rejected: bool,
    mid_decode_restore_rejected: bool,
    zero_mid_decode_fetches: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GateOutcome {
    passed: bool,
    warm_ttft_improved: bool,
    prefill_logit_parity: bool,
    prefill_token_parity: bool,
    continued_decode_parity: bool,
    namespace_rejections: bool,
    corruption_rejection: bool,
    zero_mid_decode_fetches: bool,
    bytes_metrics_present: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PayloadManifest {
    schema_version: u32,
    format: String,
    block_id: String,
    namespace_hash: String,
    cache_mode: String,
    kv_layout_version: u32,
    payload_relative_path: String,
    payload_sha256: String,
    payload_bytes: u64,
    active_kv_bytes: u64,
    token_count: u64,
    sequence_len: u64,
    has_safetensors_shape_metadata: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClaimInventory {
    exactness: Vec<String>,
    speed: Vec<String>,
    safety: Vec<String>,
    io: Vec<String>,
    memory: Vec<String>,
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

fn run_context(
    args: &Args,
    target: &Target,
    model_identity: &ModelIdentity,
    run_id: &str,
    context_tokens: usize,
) -> Result<P07Record, Box<dyn std::error::Error>> {
    let prompt_tokens = vec![9259_i32; context_tokens];
    let policy = KvPolicy::default();
    let namespace = namespace_for(model_identity, &prompt_tokens, CacheMode::Bf16)?;
    let namespace_hash = namespace.namespace_hash()?.0;
    let block_size = NonZeroU64::new(context_tokens as u64).expect("context is non-zero");
    let context_cache_dir = args.cache_dir.join(format!("{context_tokens}-tokens"));
    if context_cache_dir.exists() {
        fs::remove_dir_all(&context_cache_dir)?;
    }
    let mut ssd_cache = SsdPrefixCache::open(
        &context_cache_dir,
        NonZeroU64::new(args.ssd_budget_bytes).expect("SSD budget is non-zero"),
    )?;

    let mut cold_cache = KvCache::create(&policy)?;
    let cold_started = Instant::now();
    let cold_step = prefill(target, &mut cold_cache, &prompt_tokens)?;
    let cold_ttft = cold_started.elapsed();
    let snapshot = cold_cache.export_snapshot()?;
    let snapshot_info = snapshot.info()?;

    let observation = PrefillObservation {
        sequence_len: cold_step.sequence_len,
        greedy_token: cold_step.greedy_token as u32,
        greedy_logit_bits: cold_step.greedy_logit.to_bits(),
    };
    let block = RamPrefixBlock::from_observation(
        namespace.clone(),
        0,
        block_size,
        0,
        observation,
        snapshot_info.active_kv_bytes,
    )?
    .with_native_handle(context_tokens as u64);
    let key = block.key.clone();

    let metadata_write_started = Instant::now();
    let entry = ssd_cache.write_block(&block)?;
    let metadata_write_ms = duration_ms(metadata_write_started.elapsed());
    let metadata_manifest_path = ssd_cache.entry_path(&entry);
    let metadata_bytes_written = fs::metadata(&metadata_manifest_path)?.len();
    let metadata_manifest_layers = manifest_layer_count(&metadata_manifest_path)?;

    let payload_dir = context_cache_dir.join("payloads");
    fs::create_dir_all(&payload_dir)?;
    let payload_path = payload_dir.join(format!("{}.safetensors", key.block_id.0));
    let payload_save_started = Instant::now();
    snapshot.save_to_path(&payload_path)?;
    let payload_save_ms = duration_ms(payload_save_started.elapsed());
    let payload_bytes = fs::metadata(&payload_path)?.len();
    let payload_sha256 = file_sha256(&payload_path)?;
    let payload_manifest_path = payload_dir.join(format!("{}.manifest.json", key.block_id.0));
    let payload_manifest = PayloadManifest {
        schema_version: 1,
        format: "gemma4d_native_kv_snapshot_safetensors_v1".to_owned(),
        block_id: key.block_id.0.clone(),
        namespace_hash: namespace_hash.clone(),
        cache_mode: namespace.cache_mode.label().to_owned(),
        kv_layout_version: namespace.kv_layout_version,
        payload_relative_path: payload_path
            .strip_prefix(&context_cache_dir)
            .unwrap_or(&payload_path)
            .display()
            .to_string(),
        payload_sha256: payload_sha256.clone(),
        payload_bytes,
        active_kv_bytes: snapshot_info.active_kv_bytes,
        token_count: snapshot_info.token_count,
        sequence_len: snapshot_info.sequence_len,
        has_safetensors_shape_metadata: true,
    };
    let payload_manifest_started = Instant::now();
    fs::write(
        &payload_manifest_path,
        serde_json::to_vec_pretty(&payload_manifest)?,
    )?;
    let payload_manifest_write_ms = duration_ms(payload_manifest_started.elapsed());

    let metadata_restore_started = Instant::now();
    ssd_cache.restore_before_prefill(&key, &namespace)?;
    let metadata_restore_ms = duration_ms(metadata_restore_started.elapsed());

    let payload_checksum_started = Instant::now();
    let verified_payload = load_payload_manifest(&payload_manifest_path, &context_cache_dir)?;
    let payload_checksum_ms = duration_ms(payload_checksum_started.elapsed());
    let payload_load_started = Instant::now();
    let loaded_snapshot = KvSnapshot::load_from_path(&verified_payload)?;
    let payload_load_ms = duration_ms(payload_load_started.elapsed());

    let mut restored_cache = KvCache::create(&policy)?;
    let import_started = Instant::now();
    restored_cache.import_snapshot(&loaded_snapshot)?;
    let warm_step = restored_cache.last_step()?;
    let import_last_step_ms = duration_ms(import_started.elapsed());
    let warm_ms = metadata_restore_ms + payload_checksum_ms + payload_load_ms + import_last_step_ms;

    let baseline_decode_started = Instant::now();
    let baseline_next = decode_one(target, &mut cold_cache, cold_step.greedy_token)?;
    let baseline_decode_ms = duration_ms(baseline_decode_started.elapsed());
    let restored_decode_started = Instant::now();
    let restored_next = decode_one(target, &mut restored_cache, warm_step.greedy_token)?;
    let restored_decode_ms = duration_ms(restored_decode_started.elapsed());

    let wrong_namespace_rejected =
        namespace_rejected(&mut ssd_cache, &key, wrong_model(&namespace));
    let wrong_adapter_rejected =
        namespace_rejected(&mut ssd_cache, &key, wrong_adapter(&namespace));
    let wrong_cache_mode_rejected = namespace_rejected(
        &mut ssd_cache,
        &key,
        namespace.clone().with_cache_mode(CacheMode::MlxAffineQ8),
    );
    let payload_corruption_rejected =
        payload_corruption_rejected(&payload_manifest_path, &payload_path, &context_cache_dir)?;
    let mid_decode_restore_rejected = matches!(
        ssd_cache.restore_for_phase(&key, &namespace, SsdRestorePhase::MidDecode),
        Err(KvError::InvalidBlock(_))
    );

    let accounting = ssd_cache.accounting();
    let warm_logit_delta =
        (f64::from(cold_step.greedy_logit) - f64::from(warm_step.greedy_logit)).abs();
    let decode_logit_delta =
        (f64::from(baseline_next.greedy_logit) - f64::from(restored_next.greedy_logit)).abs();
    let cold_ms = duration_ms(cold_ttft);
    let warm = WarmSsdRestore {
        ttft_ms: warm_ms,
        metadata_restore_ms,
        payload_checksum_ms,
        payload_load_ms,
        snapshot_import_last_step_ms: import_last_step_ms,
        greedy_token: warm_step.greedy_token,
        greedy_logit: warm_step.greedy_logit,
        sequence_len: warm_step.sequence_len,
        active_kv_bytes: warm_step.active_kv_bytes,
        token_parity: cold_step.greedy_token == warm_step.greedy_token,
        logit_delta: warm_logit_delta,
        sequence_len_parity: cold_step.sequence_len == warm_step.sequence_len,
        active_kv_bytes_parity: cold_step.active_kv_bytes == warm_step.active_kv_bytes,
        ttft_improvement_ms: cold_ms - warm_ms,
        ttft_speedup: if warm_ms == 0.0 {
            f64::INFINITY
        } else {
            cold_ms / warm_ms
        },
        payload_bytes_read: payload_bytes,
    };
    let continued_decode = ContinuedDecode {
        baseline_decode_ms,
        restored_decode_ms,
        baseline_greedy_token: baseline_next.greedy_token,
        restored_greedy_token: restored_next.greedy_token,
        baseline_greedy_logit: baseline_next.greedy_logit,
        restored_greedy_logit: restored_next.greedy_logit,
        token_parity: baseline_next.greedy_token == restored_next.greedy_token,
        logit_delta: decode_logit_delta,
        sequence_len_parity: baseline_next.sequence_len == restored_next.sequence_len,
    };
    let rejection = RejectionRecord {
        wrong_namespace_rejected,
        wrong_adapter_rejected,
        wrong_cache_mode_rejected,
        payload_corruption_rejected,
        mid_decode_restore_rejected,
        zero_mid_decode_fetches: accounting.mid_decode_fetches == 0,
    };
    let gate = GateOutcome {
        passed: false,
        warm_ttft_improved: warm.ttft_ms < cold_ms,
        prefill_logit_parity: warm.logit_delta <= LOGIT_TOLERANCE,
        prefill_token_parity: warm.token_parity
            && warm.sequence_len_parity
            && warm.active_kv_bytes_parity,
        continued_decode_parity: continued_decode.token_parity
            && continued_decode.sequence_len_parity
            && continued_decode.logit_delta <= LOGIT_TOLERANCE,
        namespace_rejections: rejection.wrong_namespace_rejected
            && rejection.wrong_adapter_rejected
            && rejection.wrong_cache_mode_rejected,
        corruption_rejection: rejection.payload_corruption_rejected,
        zero_mid_decode_fetches: rejection.mid_decode_restore_rejected
            && rejection.zero_mid_decode_fetches,
        bytes_metrics_present: accounting.bytes_written > 0
            && accounting.bytes_read > 0
            && payload_bytes > 0
            && warm.payload_bytes_read > 0,
    };
    let mut gate = gate;
    gate.passed = gate.warm_ttft_improved
        && gate.prefill_logit_parity
        && gate.prefill_token_parity
        && gate.continued_decode_parity
        && gate.namespace_rejections
        && gate.corruption_rejection
        && gate.zero_mid_decode_fetches
        && gate.bytes_metrics_present;

    let mut blockers = Vec::new();
    if !gate.warm_ttft_improved {
        blockers.push(format!(
            "{context_tokens} token warm SSD restore did not improve TTFT"
        ));
    }
    if !gate.prefill_logit_parity || !gate.prefill_token_parity {
        blockers.push(format!(
            "{context_tokens} token restored SSD last-step prefill parity failed"
        ));
    }
    if !gate.continued_decode_parity {
        blockers.push(format!(
            "{context_tokens} token restored SSD cache failed continued decode parity"
        ));
    }
    if !gate.namespace_rejections {
        blockers.push(format!(
            "{context_tokens} token SSD namespace rejection matrix did not reject every mismatch"
        ));
    }
    if !gate.corruption_rejection {
        blockers.push(format!(
            "{context_tokens} token SSD payload corruption was not rejected"
        ));
    }
    if !gate.zero_mid_decode_fetches {
        blockers.push(format!(
            "{context_tokens} token SSD mid-decode restore was not rejected before fetch"
        ));
    }
    if !gate.bytes_metrics_present {
        blockers.push(format!(
            "{context_tokens} token SSD byte metrics are incomplete"
        ));
    }

    Ok(P07Record {
        schema_version: 1,
        goal: "P07-real-ssd-prefix-cache",
        run_id: run_id.to_owned(),
        timestamp_unix: unix_now(),
        context_tokens,
        prompt_token_id: 9259,
        mode: MODE,
        cache_mode: namespace.cache_mode.label(),
        namespace_hash,
        block_id: key.block_id.0,
        metadata_manifest_path: metadata_manifest_path.display().to_string(),
        payload_manifest_path: payload_manifest_path.display().to_string(),
        payload_path: payload_path.display().to_string(),
        cold: ColdPrefill {
            ttft_ms: cold_ms,
            greedy_token: cold_step.greedy_token,
            greedy_logit: cold_step.greedy_logit,
            sequence_len: cold_step.sequence_len,
            active_kv_bytes: cold_step.active_kv_bytes,
            peak_memory_gb: cold_step.peak_memory_gb,
        },
        snapshot: SnapshotRecord {
            sequence_len: snapshot_info.sequence_len,
            active_kv_bytes: snapshot_info.active_kv_bytes,
            token_count: snapshot_info.token_count,
            has_last_step: snapshot_info.has_last_step,
        },
        ssd_write: SsdWriteRecord {
            metadata_write_ms,
            payload_save_ms,
            payload_manifest_write_ms,
            metadata_bytes_written,
            payload_bytes_written: payload_bytes,
            payload_sha256,
            metadata_manifest_layers,
            manifest_has_shape_metadata: metadata_manifest_layers > 0,
            manifest_has_per_layer_attention_metadata: metadata_manifest_layers > 0,
        },
        warm,
        continued_decode,
        rejection,
        accounting,
        gate,
        blockers,
    })
}

fn load_payload_manifest(
    manifest_path: &Path,
    cache_root: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let manifest: PayloadManifest = serde_json::from_slice(&fs::read(manifest_path)?)?;
    let payload_path = cache_root.join(&manifest.payload_relative_path);
    let actual_sha256 = file_sha256(&payload_path)?;
    if actual_sha256 != manifest.payload_sha256 {
        return Err(format!(
            "payload checksum mismatch for {}: expected {}, got {}",
            payload_path.display(),
            manifest.payload_sha256,
            actual_sha256
        )
        .into());
    }
    let actual_bytes = fs::metadata(&payload_path)?.len();
    if actual_bytes != manifest.payload_bytes {
        return Err(format!(
            "payload byte length mismatch for {}: expected {}, got {}",
            payload_path.display(),
            manifest.payload_bytes,
            actual_bytes
        )
        .into());
    }
    Ok(payload_path)
}

fn payload_corruption_rejected(
    manifest_path: &Path,
    payload_path: &Path,
    cache_root: &Path,
) -> Result<bool, Box<dyn std::error::Error>> {
    let corrupt_path = payload_path.with_extension("corrupt.safetensors");
    let mut bytes = fs::read(payload_path)?;
    if bytes.is_empty() {
        return Ok(false);
    }
    bytes[0] ^= 0xff;
    fs::write(&corrupt_path, bytes)?;

    let mut manifest: PayloadManifest = serde_json::from_slice(&fs::read(manifest_path)?)?;
    manifest.payload_relative_path = corrupt_path
        .strip_prefix(cache_root)
        .unwrap_or(&corrupt_path)
        .display()
        .to_string();
    let corrupt_manifest_path = manifest_path.with_extension("corrupt.json");
    fs::write(
        &corrupt_manifest_path,
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(load_payload_manifest(&corrupt_manifest_path, cache_root).is_err())
}

fn manifest_layer_count(path: &Path) -> Result<usize, Box<dyn std::error::Error>> {
    let value: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
    Ok(value
        .get("manifest")
        .and_then(|manifest| manifest.get("layers"))
        .and_then(|layers| layers.as_array())
        .map_or(0, Vec::len))
}

fn namespace_for(
    model_identity: &ModelIdentity,
    prompt_tokens: &[i32],
    cache_mode: CacheMode,
) -> Result<KvNamespace, Box<dyn std::error::Error>> {
    let version = runtime_version()?;
    Ok(KvNamespace {
        model_id: "mlx-community/gemma-4-12B-it-4bit".to_owned(),
        model_revision: model_identity.configured_revision.clone(),
        weights_sha256: model_identity.safetensors_inventory_sha256.clone(),
        quantization_sha256: quantization_hash(model_identity),
        tokenizer_sha256: model_identity.tokenizer_sha256.clone(),
        chat_template_sha256: model_identity.chat_template_sha256.clone(),
        prompt_token_hash: prompt_token_hash(prompt_tokens),
        raw_prompt_hash: raw_prompt_hash(prompt_tokens),
        adapter_id: None,
        adapter_weight_hash: None,
        kv_layout_version: KV_LAYOUT_VERSION,
        cache_mode,
        mlx_version: version.backend_version,
        engine_version: env!("CARGO_PKG_VERSION").to_owned(),
    })
}

fn quantization_hash(model_identity: &ModelIdentity) -> String {
    sha256_hex(
        format!(
            "config={}\nsafetensors={}\n",
            model_identity.config_sha256, model_identity.safetensors_inventory_sha256
        )
        .as_bytes(),
    )
}

fn prompt_token_hash(tokens: &[i32]) -> String {
    let mut bytes = b"gemma4d:p07:prompt-token-ids:v1\0".to_vec();
    for token in tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    sha256_hex(&bytes)
}

fn raw_prompt_hash(tokens: &[i32]) -> String {
    sha256_hex(format!("token_ids:9259x{}", tokens.len()).as_bytes())
}

fn wrong_model(namespace: &KvNamespace) -> KvNamespace {
    let mut wrong = namespace.clone();
    wrong.model_id = "wrong-model".to_owned();
    wrong
}

fn wrong_adapter(namespace: &KvNamespace) -> KvNamespace {
    let mut wrong = namespace.clone();
    wrong.adapter_id = Some("wrong-adapter".to_owned());
    wrong.adapter_weight_hash = Some("wrong-adapter-weight-hash".to_owned());
    wrong
}

fn namespace_rejected(
    cache: &mut SsdPrefixCache,
    key: &KvBlockKey,
    namespace: KvNamespace,
) -> bool {
    matches!(
        cache.restore_before_prefill(key, &namespace),
        Err(KvError::NamespaceMismatch { .. })
    )
}

fn target_config(args: &Args) -> LoadConfig {
    LoadConfig {
        model_path: args.model_path.display().to_string(),
        model_id: Some("mlx-community/gemma-4-12B-it-4bit".to_owned()),
        model_revision: env::var("GEMMA4D_MODEL_REVISION").ok(),
        expected_architecture: Some("gemma4".to_owned()),
        max_context_tokens: NonZeroU32::new(args.max_context_tokens as u32)
            .expect("max context is non-zero"),
        allow_unsupported_config: false,
    }
}

fn startup_blockers(args: &Args) -> Vec<String> {
    let mut blockers = Vec::new();
    if !args.model_path.exists() {
        blockers.push(format!(
            "model path does not exist: {}",
            args.model_path.display()
        ));
    }
    if env::var_os("GEMMA4D_USE_NATIVE_GRAPH").is_none() {
        blockers.push("GEMMA4D_USE_NATIVE_GRAPH=1 is required for P07 native SSD cache".to_owned());
    }
    if env::var_os("GEMMA4D_REQUIRE_MLX").is_none() {
        blockers
            .push("GEMMA4D_REQUIRE_MLX=1 is required so gemma4d-ffi builds with MLX".to_owned());
    }
    blockers
}

fn blockers_for_records(records: &[P07Record], expected_contexts: &[usize]) -> Vec<String> {
    let mut blockers = Vec::new();
    for context in expected_contexts {
        if !records
            .iter()
            .any(|record| record.context_tokens == *context)
        {
            blockers.push(format!("{context} token P07 record is missing"));
        }
    }
    for record in records {
        blockers.extend(record.blockers.clone());
    }
    blockers
}

fn claim_inventory(records: &[P07Record]) -> ClaimInventory {
    ClaimInventory {
        exactness: records
            .iter()
            .map(|record| {
                format!(
                    "{} tokens restored SSD prefill parity={}, continued decode parity={}",
                    record.context_tokens,
                    record.gate.prefill_token_parity && record.gate.prefill_logit_parity,
                    record.gate.continued_decode_parity
                )
            })
            .collect(),
        speed: records
            .iter()
            .map(|record| {
                format!(
                    "{} tokens cold TTFT {:.3} ms, warm SSD TTFT {:.3} ms, speedup {:.2}x",
                    record.context_tokens,
                    record.cold.ttft_ms,
                    record.warm.ttft_ms,
                    record.warm.ttft_speedup
                )
            })
            .collect(),
        safety: records
            .iter()
            .map(|record| {
                format!(
                    "{} tokens namespace={}, corruption={}, mid_decode_fetches={}",
                    record.context_tokens,
                    record.gate.namespace_rejections,
                    record.gate.corruption_rejection,
                    record.accounting.mid_decode_fetches
                )
            })
            .collect(),
        io: records
            .iter()
            .map(|record| {
                format!(
                    "{} tokens metadata write/read bytes {}/{}, payload write/read bytes {}/{}",
                    record.context_tokens,
                    record.accounting.bytes_written,
                    record.accounting.bytes_read,
                    record.ssd_write.payload_bytes_written,
                    record.warm.payload_bytes_read
                )
            })
            .collect(),
        memory: records
            .iter()
            .map(|record| {
                format!(
                    "{} tokens active KV {:.3} MiB, payload {:.3} MiB, peak MLX {:.3} GB",
                    record.context_tokens,
                    bytes_to_mib(record.snapshot.active_kv_bytes),
                    bytes_to_mib(record.ssd_write.payload_bytes_written),
                    record.cold.peak_memory_gb
                )
            })
            .collect(),
    }
}

fn capture_environment() -> Environment {
    let version = runtime_version().ok();
    Environment {
        machine: command_stdout("uname", &["-m"]).unwrap_or_else(|| "unknown".to_owned()),
        macos: command_stdout("sw_vers", &["-productVersion"])
            .unwrap_or_else(|| "unknown".to_owned()),
        rustc: command_stdout("rustc", &["--version"]).unwrap_or_else(|| "unknown".to_owned()),
        cargo: command_stdout("cargo", &["--version"]).unwrap_or_else(|| "unknown".to_owned()),
        runtime_backend: version
            .as_ref()
            .map(|value| value.backend_name.clone())
            .unwrap_or_else(|| "unknown".to_owned()),
        runtime_backend_version: version
            .as_ref()
            .map(|value| value.backend_version.clone())
            .unwrap_or_else(|| "unknown".to_owned()),
        git_commit: command_stdout("git", &["rev-parse", "--short", "HEAD"])
            .unwrap_or_else(|| "unknown".to_owned()),
        git_status_short: command_stdout("git", &["status", "--short"])
            .unwrap_or_else(|| "unknown".to_owned()),
        hw_memsize_bytes: command_stdout("sysctl", &["-n", "hw.memsize"])
            .and_then(|value| value.trim().parse::<u64>().ok()),
    }
}

fn capture_relevant_environment() -> BTreeMap<String, Option<String>> {
    [
        "GEMMA4D_MLX_LM_PYTHON",
        "GEMMA4D_MODEL_PATH",
        "GEMMA4D_MODEL_REVISION",
        "GEMMA4D_USE_NATIVE_GRAPH",
        "GEMMA4D_REQUIRE_MLX",
        "GEMMA4D_FULL_MODEL_TESTS",
        "RUSTFLAGS",
    ]
    .into_iter()
    .map(|key| (key.to_owned(), env::var(key).ok()))
    .collect()
}

fn capture_model_identity(model_path: &Path) -> ModelIdentity {
    let safetensors = safetensors_inventory(model_path);
    ModelIdentity {
        model_path: model_path.display().to_string(),
        exists: model_path.exists(),
        configured_revision: env::var("GEMMA4D_MODEL_REVISION")
            .unwrap_or_else(|_| "unavailable:GEMMA4D_MODEL_REVISION not set".to_owned()),
        config_sha256: sha256_file_or_unavailable(&model_path.join("config.json")),
        tokenizer_sha256: sha256_file_or_unavailable(&model_path.join("tokenizer.json")),
        tokenizer_config_sha256: sha256_file_or_unavailable(
            &model_path.join("tokenizer_config.json"),
        ),
        chat_template_sha256: sha256_file_or_unavailable(&model_path.join("chat_template.json")),
        safetensors_inventory_sha256: safetensors.inventory_sha256,
        safetensors_file_count: safetensors.file_count,
        safetensors_total_bytes: safetensors.total_bytes,
    }
}

fn safetensors_inventory(model_path: &Path) -> SafetensorsInventory {
    let mut entries = Vec::new();
    collect_safetensors(model_path, model_path, &mut entries);
    entries.sort();
    let total_bytes = entries
        .iter()
        .filter_map(|entry| entry.rsplit_once('\t'))
        .filter_map(|(_, bytes)| bytes.parse::<u64>().ok())
        .sum();
    let body = entries.join("\n");
    SafetensorsInventory {
        inventory_sha256: if entries.is_empty() {
            "unavailable:no safetensors files found".to_owned()
        } else {
            sha256_hex(body.as_bytes())
        },
        file_count: entries.len(),
        total_bytes,
    }
}

fn collect_safetensors(root: &Path, current: &Path, entries: &mut Vec<String>) {
    let Ok(read_dir) = fs::read_dir(current) else {
        return;
    };
    for entry in read_dir.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_safetensors(root, &path, entries);
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("safetensors") {
            let relative = path.strip_prefix(root).unwrap_or(&path);
            let bytes = entry.metadata().map(|metadata| metadata.len()).unwrap_or(0);
            entries.push(format!("{}\t{}", relative.display(), bytes));
        }
    }
}

fn render_report(summary: &P07Summary) -> String {
    let mut out = String::new();
    out.push_str("# P07 Real SSD Prefix Cache\n\n");
    out.push_str(&format!("Status: `{}`\n\n", summary.status));
    out.push_str("## Run\n\n");
    out.push_str("| Field | Value |\n|---|---|\n");
    out.push_str(&format!("| Run ID | `{}` |\n", summary.run_id));
    out.push_str(&format!("| Mode | `{}` |\n", summary.mode));
    out.push_str(&format!("| Model path | `{}` |\n", summary.model_path));
    out.push_str(&format!("| Cache dir | `{}` |\n", summary.cache_dir));
    if let Some(model_load_ms) = summary.model_load_ms {
        out.push_str(&format!("| Model load ms | `{model_load_ms:.3}` |\n"));
    }
    out.push_str(&format!(
        "| Runtime | `{}` `{}` |\n",
        escape_md(&summary.environment.runtime_backend),
        escape_md(&summary.environment.runtime_backend_version)
    ));
    out.push_str(&format!(
        "| Git | `{}` |\n",
        escape_md(&summary.environment.git_commit)
    ));
    out.push_str(&format!(
        "| Default recommendation | `{}` |\n",
        summary.default_recommendation
    ));
    out.push('\n');

    out.push_str("## Results\n\n");
    out.push_str("| Context | Parity | Cold TTFT ms | Warm SSD TTFT ms | Speedup | Payload MiB | Metadata Read/Write | Payload Read/Write | Mid-Decode Fetches |\n");
    out.push_str("|---:|---|---:|---:|---:|---:|---:|---:|---:|\n");
    for record in &summary.records {
        out.push_str(&format!(
            "| {} | `{}` | {:.3} | {:.3} | {:.2}x | {:.3} | {}/{} | {}/{} | {} |\n",
            record.context_tokens,
            record.gate.passed,
            record.cold.ttft_ms,
            record.warm.ttft_ms,
            record.warm.ttft_speedup,
            bytes_to_mib(record.ssd_write.payload_bytes_written),
            record.accounting.bytes_read,
            record.accounting.bytes_written,
            record.warm.payload_bytes_read,
            record.ssd_write.payload_bytes_written,
            record.accounting.mid_decode_fetches
        ));
    }
    out.push('\n');

    out.push_str("## Rejections\n\n");
    out.push_str(
        "| Context | Wrong Model | Wrong Adapter | Wrong Cache Mode | Corruption | Mid Decode |\n",
    );
    out.push_str("|---:|---|---|---|---|---|\n");
    for record in &summary.records {
        out.push_str(&format!(
            "| {} | `{}` | `{}` | `{}` | `{}` | `{}` |\n",
            record.context_tokens,
            record.rejection.wrong_namespace_rejected,
            record.rejection.wrong_adapter_rejected,
            record.rejection.wrong_cache_mode_rejected,
            record.rejection.payload_corruption_rejected,
            record.rejection.mid_decode_restore_rejected
        ));
    }
    out.push('\n');

    out.push_str("## Verification Command\n\n");
    out.push_str("```sh\n");
    out.push_str(&format!(
        "GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example p07_real_ssd_prefix_cache -- --out-dir {} --cache-dir {} --model-path {}\n",
        summary.records_path
            .strip_suffix("/records.jsonl")
            .unwrap_or(DEFAULT_OUT_DIR),
        summary.cache_dir,
        summary.model_path
    ));
    out.push_str("```\n\n");

    out.push_str("## Notes\n\n");
    for note in &summary.measurement_notes {
        out.push_str(&format!("- {note}.\n"));
    }
    out
}

fn render_blockers(summary: &P07Summary) -> String {
    let mut out = String::new();
    out.push_str("# P07 Blockers\n\n");
    if summary.blockers.is_empty() {
        out.push_str("No blockers recorded.\n");
    } else {
        for blocker in &summary.blockers {
            out.push_str(&format!("- {blocker}\n"));
        }
    }
    out
}

fn write_jsonl(path: &Path, records: &[P07Record]) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = fs::File::create(path)?;
    for record in records {
        writeln!(file, "{}", serde_json::to_string(record)?)?;
    }
    Ok(())
}

fn parse_contexts(value: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>()
                .map_err(|_| format!("invalid context '{part}'").into())
        })
        .collect()
}

fn parse_positive_usize(value: &str, option: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{option} must be an integer"))?;
    if parsed == 0 {
        Err(format!("{option} must be > 0").into())
    } else {
        Ok(parsed)
    }
}

fn command_stdout(command: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(command).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn sha256_file_or_unavailable(path: &Path) -> String {
    file_sha256(path).unwrap_or_else(|error| format!("unavailable:{error}"))
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn bytes_to_mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn escape_md(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn run_id() -> String {
    format!("p07-{}", unix_now())
}
