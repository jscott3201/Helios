use std::{
    collections::BTreeMap,
    env, fs,
    io::Write,
    num::{NonZeroU32, NonZeroU64},
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use gemma4d_adapters::{
    AdapterCompatibility, AdapterRegistry, Error as AdapterRegistryError, ImportedAdapter,
    TrustedPathPolicy,
};
use gemma4d_ffi::{
    Adapter, AdapterInfo, AdapterLoadConfig, Drafter, KvCache, KvPolicy, LoadConfig, Status,
    StepResult, Target, decode_one, prefill, runtime_version,
};
use gemma4d_kv::{
    Error as KvError, KvBlockKey, KvNamespace, RamPrefixBlock, RamPrefixCache,
    estimated_bf16_kv_bytes, fresh_prefill_fixture,
};
use gemma4d_tokenizer::{file_sha256, sha256_hex};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

const DEFAULT_MODEL: &str = "artifacts/models/gemma-4-12B-it-4bit";
const DEFAULT_OUT_DIR: &str = "benchmarks/out/P09-real-lora-adapter";
const MODE: &str = "native_lora_adapter_hot_path";
const ADAPTER_ID: &str = "rust-coding-r16-v1";
const BASE_MODEL_ID: &str = "mlx-community/gemma-4-12B-it-4bit";
const PROMPT_TOKEN_ID: i32 = 9259;
const RANK: u32 = 16;
const ALPHA: f32 = 32.0;
const OUTPUT_DIFF_EPSILON: f64 = 0.000_1;

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

    let record = if blockers.is_empty() {
        match run_p09(&args, &run_id, &model_identity) {
            Ok(record) => Some(record),
            Err(error) => {
                blockers.push(format!("P09 run failed: {error}"));
                None
            }
        }
    } else {
        None
    };

    if let Some(record) = &record {
        blockers.extend(record.blockers.iter().cloned());
    }
    blockers.sort();
    blockers.dedup();
    let status = if blockers.is_empty() {
        "passed"
    } else {
        "failed"
    };
    let records = record.into_iter().collect::<Vec<_>>();
    let summary = P09Summary {
        schema_version: 1,
        goal: "P09-real-lora-adapter-hot-path",
        status,
        run_id,
        timestamp_unix: unix_now(),
        mode: MODE,
        model_path: args.model_path.display().to_string(),
        out_dir: args.out_dir.display().to_string(),
        records_path: records_path.display().to_string(),
        summary_path: summary_path.display().to_string(),
        report_path: report_path.display().to_string(),
        blockers_path: blockers_path.display().to_string(),
        context_tokens: args.context_tokens,
        decode_tokens: args.decode_tokens,
        max_context_tokens: args.max_context_tokens,
        environment,
        relevant_environment: capture_relevant_environment(),
        model_identity,
        claims: claim_inventory(&records),
        records,
        blockers,
        measurement_notes: vec![
            "adapter fixture is a trusted local deterministic rank-16 PEFT LoRA safetensors payload with real Gemma 4 layer-0 q_proj/v_proj shapes",
            "adapter output difference is measured on native prefill/decode greedy tokens and greedy-logit deltas against the base target",
            "generation latency is measured with fresh KV caches for base, active-adapter, and post-clear base runs",
            "adapter-aware KV namespace checks exercise RAM prefix namespace/hash/block-id isolation outside the native tensor handle",
            "MTP is expected to fail closed while the standard LoRA adapter is active",
            "no remote adapter loading path is used or exposed by this benchmark",
        ],
    };

    write_jsonl(&records_path, &summary.records)?;
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)?;
    fs::write(&report_path, render_report(&summary))?;
    fs::write(&blockers_path, render_blockers(&summary))?;

    println!("P09 real LoRA adapter: {}", summary.status);
    println!("records: {}", records_path.display());
    println!("summary: {}", summary_path.display());
    println!("report: {}", report_path.display());
    println!("blockers: {}", blockers_path.display());

    if summary.status == "failed" {
        Err("P09 real LoRA adapter checks failed".into())
    } else {
        Ok(())
    }
}

#[derive(Debug)]
struct Args {
    out_dir: PathBuf,
    model_path: PathBuf,
    context_tokens: usize,
    decode_tokens: usize,
    max_context_tokens: usize,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut out_dir = PathBuf::from(DEFAULT_OUT_DIR);
        let mut model_path = PathBuf::from(DEFAULT_MODEL);
        let mut context_tokens = 128_usize;
        let mut decode_tokens = 2_usize;
        let mut max_context_tokens = 4096_usize;

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
                "--context-tokens" => {
                    let value = args.next().ok_or("--context-tokens requires a value")?;
                    context_tokens = parse_positive_usize(&value, "--context-tokens")?;
                }
                "--decode-tokens" => {
                    let value = args.next().ok_or("--decode-tokens requires a value")?;
                    decode_tokens = parse_positive_usize(&value, "--decode-tokens")?;
                }
                "--max-context-tokens" => {
                    let value = args.next().ok_or("--max-context-tokens requires a value")?;
                    max_context_tokens = parse_positive_usize(&value, "--max-context-tokens")?;
                }
                "-h" | "--help" => {
                    println!(
                        "usage: GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example p09_real_lora_adapter -- [--out-dir PATH] [--model-path PATH] [--context-tokens N] [--decode-tokens N] [--max-context-tokens N]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown option '{other}'").into()),
            }
        }

        if context_tokens == 0 || decode_tokens == 0 {
            return Err("--context-tokens and --decode-tokens must be > 0".into());
        }
        if context_tokens + decode_tokens > max_context_tokens {
            return Err(
                "--context-tokens + --decode-tokens cannot exceed --max-context-tokens".into(),
            );
        }

        Ok(Self {
            out_dir,
            model_path,
            context_tokens,
            decode_tokens,
            max_context_tokens,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct P09Summary {
    schema_version: u32,
    goal: &'static str,
    status: &'static str,
    run_id: String,
    timestamp_unix: u64,
    mode: &'static str,
    model_path: String,
    out_dir: String,
    records_path: String,
    summary_path: String,
    report_path: String,
    blockers_path: String,
    context_tokens: usize,
    decode_tokens: usize,
    max_context_tokens: usize,
    environment: Environment,
    relevant_environment: BTreeMap<String, Option<String>>,
    model_identity: ModelIdentity,
    claims: ClaimInventory,
    records: Vec<P09Record>,
    blockers: Vec<String>,
    measurement_notes: Vec<&'static str>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct P09Record {
    schema_version: u32,
    goal: &'static str,
    run_id: String,
    timestamp_unix: u64,
    mode: &'static str,
    adapter: AdapterEvidence,
    rejection: RejectionEvidence,
    kv_namespace: KvNamespaceEvidence,
    base: GenerationRun,
    adapter_run: GenerationRun,
    base_after_clear: GenerationRun,
    hotswap: HotswapEvidence,
    mtp_disabled_with_adapter: bool,
    gate: GateEvidence,
    blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AdapterEvidence {
    adapter_id: String,
    adapter_path: String,
    weights_path: String,
    weights_sha256: String,
    target_modules: Vec<String>,
    rank: u32,
    alpha: f32,
    registry_import_latency_us: u128,
    native_load_latency_us: u64,
    module_count: u64,
    resident_bytes: u64,
    shape_validation_result: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RejectionEvidence {
    wrong_base_rejected: bool,
    wrong_base_weight_hash_rejected: bool,
    wrong_tokenizer_rejected: bool,
    wrong_template_rejected: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KvNamespaceEvidence {
    base_namespace_hash: String,
    adapter_namespace_hash: String,
    namespace_hashes_unique_by_adapter: bool,
    block_ids_unique_by_adapter: bool,
    wrong_adapter_restore_rejected: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HotswapEvidence {
    activate_us: u128,
    clear_us: u128,
    activate_info: AdapterInfoRecord,
    clear_info: AdapterInfoRecord,
    base_after_clear_matches_base: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AdapterInfoRecord {
    module_count: u64,
    resident_bytes: u64,
    load_latency_us: u64,
    active: bool,
}

impl From<AdapterInfo> for AdapterInfoRecord {
    fn from(value: AdapterInfo) -> Self {
        Self {
            module_count: value.module_count,
            resident_bytes: value.resident_bytes,
            load_latency_us: value.load_latency_us,
            active: value.active,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GenerationRun {
    label: &'static str,
    context_tokens: usize,
    decode_tokens: usize,
    prefill_ms: f64,
    decode_ms: f64,
    total_generation_ms: f64,
    prefill: StepEvidence,
    decode_steps: Vec<StepEvidence>,
    generated_tokens: Vec<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StepEvidence {
    greedy_token: i32,
    greedy_logit: f32,
    sequence_len: u64,
    active_kv_bytes: u64,
    peak_memory_gb: f32,
}

impl From<StepResult> for StepEvidence {
    fn from(value: StepResult) -> Self {
        Self {
            greedy_token: value.greedy_token,
            greedy_logit: value.greedy_logit,
            sequence_len: value.sequence_len,
            active_kv_bytes: value.active_kv_bytes,
            peak_memory_gb: value.peak_memory_gb,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GateEvidence {
    adapter_output_differs_from_base: bool,
    prefill_greedy_token_differs: bool,
    prefill_greedy_logit_delta: f64,
    generated_tokens_differ: bool,
    wrong_manifest_rejections_pass: bool,
    kv_namespace_isolated: bool,
    native_adapter_loaded: bool,
    hotswap_measured: bool,
    mtp_disabled_with_adapter: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ClaimInventory {
    correctness: Vec<String>,
    latency: Vec<String>,
    memory: Vec<String>,
    defaults: Vec<String>,
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

fn run_p09(
    args: &Args,
    run_id: &str,
    model_identity: &ModelIdentity,
) -> Result<P09Record, Box<dyn std::error::Error>> {
    let fixture_root = args.out_dir.join("fixtures");
    let trusted_root = fixture_root.join("trusted");
    let registry_dir = args.out_dir.join("registry");
    reset_dir(&fixture_root)?;
    reset_dir(&registry_dir)?;
    fs::create_dir_all(&trusted_root)?;

    let compatibility = compatibility(model_identity);
    let adapter_dir = write_adapter_fixture(
        &trusted_root,
        ADAPTER_ID,
        &compatibility,
        AdapterOverride::default(),
        false,
    )?;
    let policy = TrustedPathPolicy::new(&trusted_root)?;
    let mut registry = AdapterRegistry::open(&registry_dir)?;
    let imported = registry.import_peft(&adapter_dir, &policy, &compatibility)?;

    let rejection = rejection_report(&trusted_root, &policy, &registry_dir, &compatibility)?;
    let kv_namespace = kv_namespace_report(
        args.context_tokens,
        &imported.manifest.adapter_id,
        &imported.manifest.adapter_weight_hash,
    )?;

    let mut target = Target::load(&target_config(args))?;
    let load_config = AdapterLoadConfig {
        adapter_path: imported.source_path.display().to_string(),
        adapter_id: imported.manifest.adapter_id.clone(),
        adapter_weight_hash: imported.manifest.adapter_weight_hash.clone(),
        target_modules: imported.manifest.target_modules.clone(),
        rank: NonZeroU32::new(imported.manifest.rank).expect("rank validated non-zero"),
        alpha: imported.manifest.alpha as f32,
    };
    let adapter = Adapter::load(&target, &load_config)?;
    let adapter_info = adapter.info();

    let base = run_generation(
        "base",
        &target,
        args.context_tokens,
        args.decode_tokens,
        PROMPT_TOKEN_ID,
    )?;
    let activate_started = Instant::now();
    let activate_info = target.set_adapter(&adapter)?;
    let activate_us = duration_us(activate_started.elapsed());
    let mtp_disabled_with_adapter = match Drafter::load(&target_config(args), &target) {
        Ok(_) => false,
        Err(error) => error.status() == Status::Adapter,
    };
    let adapter_run = run_generation(
        "adapter",
        &target,
        args.context_tokens,
        args.decode_tokens,
        PROMPT_TOKEN_ID,
    )?;
    let clear_started = Instant::now();
    let clear_info = target.clear_adapter()?;
    let clear_us = duration_us(clear_started.elapsed());
    let base_after_clear = run_generation(
        "base_after_clear",
        &target,
        args.context_tokens,
        args.decode_tokens,
        PROMPT_TOKEN_ID,
    )?;

    let prefill_greedy_logit_delta =
        (f64::from(adapter_run.prefill.greedy_logit) - f64::from(base.prefill.greedy_logit)).abs();
    let prefill_greedy_token_differs =
        adapter_run.prefill.greedy_token != base.prefill.greedy_token;
    let generated_tokens_differ = adapter_run.generated_tokens != base.generated_tokens;
    let adapter_output_differs_from_base = prefill_greedy_token_differs
        || generated_tokens_differ
        || prefill_greedy_logit_delta > OUTPUT_DIFF_EPSILON;
    let wrong_manifest_rejections_pass = rejection.wrong_base_rejected
        && rejection.wrong_base_weight_hash_rejected
        && rejection.wrong_tokenizer_rejected
        && rejection.wrong_template_rejected;
    let kv_namespace_isolated = kv_namespace.namespace_hashes_unique_by_adapter
        && kv_namespace.block_ids_unique_by_adapter
        && kv_namespace.wrong_adapter_restore_rejected;
    let native_adapter_loaded = adapter_info.module_count > 0 && adapter_info.resident_bytes > 0;
    let base_after_clear_matches_base = base.prefill.greedy_token
        == base_after_clear.prefill.greedy_token
        && (f64::from(base.prefill.greedy_logit)
            - f64::from(base_after_clear.prefill.greedy_logit))
        .abs()
            <= OUTPUT_DIFF_EPSILON
        && base.generated_tokens == base_after_clear.generated_tokens;
    let hotswap_measured =
        activate_us > 0 && clear_us > 0 && activate_info.active && !clear_info.active;

    let gate = GateEvidence {
        adapter_output_differs_from_base,
        prefill_greedy_token_differs,
        prefill_greedy_logit_delta,
        generated_tokens_differ,
        wrong_manifest_rejections_pass,
        kv_namespace_isolated,
        native_adapter_loaded,
        hotswap_measured,
        mtp_disabled_with_adapter,
    };
    let mut blockers = Vec::new();
    if !gate.adapter_output_differs_from_base {
        blockers.push(
            "adapter output did not differ from base by token or greedy-logit gate".to_owned(),
        );
    }
    if !gate.wrong_manifest_rejections_pass {
        blockers.push("wrong base/tokenizer/template/hash rejection gate failed".to_owned());
    }
    if !gate.kv_namespace_isolated {
        blockers.push("adapter-aware KV namespace isolation gate failed".to_owned());
    }
    if !gate.native_adapter_loaded {
        blockers.push("native adapter did not report loaded modules and resident bytes".to_owned());
    }
    if !gate.hotswap_measured {
        blockers.push("adapter hotswap activate/clear was not measured correctly".to_owned());
    }
    if !gate.mtp_disabled_with_adapter {
        blockers.push("MTP did not fail closed while adapter was active".to_owned());
    }
    if !base_after_clear_matches_base {
        blockers.push(
            "clearing adapter did not restore base generation for the deterministic prompt"
                .to_owned(),
        );
    }

    Ok(P09Record {
        schema_version: 1,
        goal: "P09-real-lora-adapter-hot-path",
        run_id: run_id.to_owned(),
        timestamp_unix: unix_now(),
        mode: MODE,
        adapter: adapter_evidence(&imported, adapter_info)?,
        rejection,
        kv_namespace,
        base,
        adapter_run,
        base_after_clear,
        hotswap: HotswapEvidence {
            activate_us,
            clear_us,
            activate_info: activate_info.into(),
            clear_info: clear_info.into(),
            base_after_clear_matches_base,
        },
        mtp_disabled_with_adapter,
        gate,
        blockers,
    })
}

fn adapter_evidence(
    imported: &ImportedAdapter,
    info: AdapterInfo,
) -> Result<AdapterEvidence, Box<dyn std::error::Error>> {
    Ok(AdapterEvidence {
        adapter_id: imported.manifest.adapter_id.clone(),
        adapter_path: imported.source_path.display().to_string(),
        weights_path: imported.weights_path.display().to_string(),
        weights_sha256: file_sha256(&imported.weights_path)?,
        target_modules: imported.manifest.target_modules.clone(),
        rank: imported.manifest.rank,
        alpha: imported.manifest.alpha as f32,
        registry_import_latency_us: imported.load_latency_us,
        native_load_latency_us: info.load_latency_us,
        module_count: info.module_count,
        resident_bytes: info.resident_bytes,
        shape_validation_result: "native_lora_shape_validated_against_loaded_gemma4_weights"
            .to_owned(),
    })
}

fn run_generation(
    label: &'static str,
    target: &Target,
    context_tokens: usize,
    decode_tokens: usize,
    prompt_token_id: i32,
) -> Result<GenerationRun, Box<dyn std::error::Error>> {
    let mut cache = KvCache::create(&KvPolicy::default())?;
    let prompt = vec![prompt_token_id; context_tokens];
    let prefill_started = Instant::now();
    let prefill_step = prefill(target, &mut cache, &prompt)?;
    let prefill_ms = duration_ms(prefill_started.elapsed());

    let mut current_token = prefill_step.greedy_token;
    let mut decode_steps = Vec::with_capacity(decode_tokens);
    let mut generated_tokens = vec![current_token];
    let decode_started = Instant::now();
    for _ in 0..decode_tokens {
        let step = decode_one(target, &mut cache, current_token)?;
        current_token = step.greedy_token;
        generated_tokens.push(current_token);
        decode_steps.push(step.into());
    }
    let decode_ms = duration_ms(decode_started.elapsed());
    Ok(GenerationRun {
        label,
        context_tokens,
        decode_tokens,
        prefill_ms,
        decode_ms,
        total_generation_ms: prefill_ms + decode_ms,
        prefill: prefill_step.into(),
        decode_steps,
        generated_tokens,
    })
}

fn rejection_report(
    trusted_root: &Path,
    policy: &TrustedPathPolicy,
    registry_dir: &Path,
    compatibility: &AdapterCompatibility,
) -> Result<RejectionEvidence, Box<dyn std::error::Error>> {
    Ok(RejectionEvidence {
        wrong_base_rejected: rejection_case(
            trusted_root,
            policy,
            registry_dir,
            compatibility,
            "wrong-base",
            AdapterOverride {
                base_model_id: Some("other-model".to_owned()),
                ..AdapterOverride::default()
            },
        )?,
        wrong_base_weight_hash_rejected: rejection_case(
            trusted_root,
            policy,
            registry_dir,
            compatibility,
            "wrong-base-hash",
            AdapterOverride {
                base_weight_hash: Some("other-base-weight-hash".to_owned()),
                ..AdapterOverride::default()
            },
        )?,
        wrong_tokenizer_rejected: rejection_case(
            trusted_root,
            policy,
            registry_dir,
            compatibility,
            "wrong-tokenizer",
            AdapterOverride {
                tokenizer_hash: Some("other-tokenizer".to_owned()),
                ..AdapterOverride::default()
            },
        )?,
        wrong_template_rejected: rejection_case(
            trusted_root,
            policy,
            registry_dir,
            compatibility,
            "wrong-template",
            AdapterOverride {
                chat_template_hash: Some("other-template".to_owned()),
                ..AdapterOverride::default()
            },
        )?,
    })
}

fn rejection_case(
    trusted_root: &Path,
    policy: &TrustedPathPolicy,
    registry_dir: &Path,
    compatibility: &AdapterCompatibility,
    adapter_id: &str,
    override_metadata: AdapterOverride,
) -> Result<bool, Box<dyn std::error::Error>> {
    let adapter_dir = write_adapter_fixture(
        trusted_root,
        adapter_id,
        compatibility,
        override_metadata,
        false,
    )?;
    let mut registry = AdapterRegistry::open(registry_dir)?;
    Ok(matches!(
        registry.import_peft(&adapter_dir, policy, compatibility),
        Err(AdapterRegistryError::InvalidManifest(_))
    ))
}

fn kv_namespace_report(
    context_tokens: usize,
    adapter_id: &str,
    adapter_weight_hash: &str,
) -> Result<KvNamespaceEvidence, Box<dyn std::error::Error>> {
    let block_size = NonZeroU64::new(1024).expect("non-zero");
    let mut base_namespace = KvNamespace::fixture(context_tokens as u64);
    base_namespace.adapter_id = None;
    base_namespace.adapter_weight_hash = None;
    let mut adapter_namespace = base_namespace.clone();
    adapter_namespace.adapter_id = Some(adapter_id.to_owned());
    adapter_namespace.adapter_weight_hash = Some(adapter_weight_hash.to_owned());

    let base_namespace_hash = base_namespace.namespace_hash()?;
    let adapter_namespace_hash = adapter_namespace.namespace_hash()?;
    let namespace_hashes_unique_by_adapter = base_namespace_hash != adapter_namespace_hash;
    let base_key = KvBlockKey::new(&base_namespace, 0, block_size, 0, context_tokens as u64)?;
    let adapter_key = KvBlockKey::new(&adapter_namespace, 0, block_size, 0, context_tokens as u64)?;
    let block_ids_unique_by_adapter = base_key.block_id != adapter_key.block_id;

    let block = RamPrefixBlock::from_observation(
        adapter_namespace.clone(),
        0,
        block_size,
        0,
        fresh_prefill_fixture(context_tokens as u64),
        estimated_bf16_kv_bytes(context_tokens as u64),
    )?;
    let key = block.key.clone();
    let mut cache = RamPrefixCache::new(NonZeroU64::new(block.byte_len * 2).expect("non-zero"));
    cache.insert(block)?;
    let wrong_adapter_restore_rejected = matches!(
        cache.restore(&key, &base_namespace),
        Err(KvError::NamespaceMismatch { .. })
    );

    Ok(KvNamespaceEvidence {
        base_namespace_hash: base_namespace_hash.0,
        adapter_namespace_hash: adapter_namespace_hash.0,
        namespace_hashes_unique_by_adapter,
        block_ids_unique_by_adapter,
        wrong_adapter_restore_rejected,
    })
}

#[derive(Debug, Clone, Default)]
struct AdapterOverride {
    base_model_id: Option<String>,
    base_weight_hash: Option<String>,
    tokenizer_hash: Option<String>,
    chat_template_hash: Option<String>,
}

fn write_adapter_fixture(
    trusted_root: &Path,
    adapter_id: &str,
    compatibility: &AdapterCompatibility,
    override_metadata: AdapterOverride,
    modules_to_save: bool,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let adapter_dir = trusted_root.join(adapter_id);
    fs::create_dir_all(&adapter_dir)?;
    write_adapter_config(
        &adapter_dir,
        adapter_id,
        compatibility,
        override_metadata,
        modules_to_save,
    )?;
    write_real_shape_lora_safetensors(&adapter_dir.join("adapter_model.safetensors"))?;
    Ok(adapter_dir)
}

fn write_adapter_config(
    adapter_dir: &Path,
    adapter_id: &str,
    compatibility: &AdapterCompatibility,
    override_metadata: AdapterOverride,
    modules_to_save: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let modules_to_save = if modules_to_save {
        r#",
  "modules_to_save": ["lm_head"]"#
    } else {
        ""
    };
    let raw = format!(
        r#"{{
  "peft_type": "LORA",
  "base_model_name_or_path": "{BASE_MODEL_ID}",
  "r": {RANK},
  "lora_alpha": {ALPHA},
  "lora_dropout": 0.0,
  "target_modules": ["q_proj", "v_proj"]{modules_to_save},
  "gemma4d": {{
    "adapter_id": "{adapter_id}",
    "base_model_id": "{}",
    "base_weight_hash": "{}",
    "tokenizer_hash": "{}",
    "chat_template_hash": "{}",
    "adapter_type": "lora",
    "dtype": "fp32",
    "supports_mtp": "false"
  }}
}}"#,
        override_metadata
            .base_model_id
            .unwrap_or_else(|| compatibility.base_model_id.clone()),
        override_metadata
            .base_weight_hash
            .unwrap_or_else(|| compatibility.base_weight_hash.clone()),
        override_metadata
            .tokenizer_hash
            .unwrap_or_else(|| compatibility.tokenizer_hash.clone()),
        override_metadata
            .chat_template_hash
            .unwrap_or_else(|| compatibility.chat_template_hash.clone()),
    );
    fs::write(adapter_dir.join("adapter_config.json"), raw)?;
    Ok(())
}

fn write_real_shape_lora_safetensors(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut header = Map::new();
    header.insert("__metadata__".to_owned(), json!({"format": "pt"}));
    let mut data = Vec::new();
    add_lora_a_tensor(
        &mut header,
        &mut data,
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
        RANK as usize,
        3840,
        0.8,
    );
    add_lora_b_tensor(
        &mut header,
        &mut data,
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight",
        4096,
        RANK as usize,
        0.35,
    );
    add_lora_a_tensor(
        &mut header,
        &mut data,
        "base_model.model.layers.0.self_attn.v_proj.lora_A.weight",
        RANK as usize,
        3840,
        0.8,
    );
    add_lora_b_tensor(
        &mut header,
        &mut data,
        "base_model.model.layers.0.self_attn.v_proj.lora_B.weight",
        2048,
        RANK as usize,
        0.25,
    );

    let header = serde_json::to_vec(&Value::Object(header))?;
    let mut bytes = Vec::with_capacity(8 + header.len() + data.len());
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&data);
    fs::write(path, bytes)?;
    Ok(())
}

fn add_lora_a_tensor(
    header: &mut Map<String, Value>,
    data: &mut Vec<u8>,
    name: &str,
    rank: usize,
    input_dim: usize,
    amplitude: f32,
) {
    let start = data.len();
    for r in 0..rank {
        for i in 0..input_dim {
            let value = if i == ((r * 193) % input_dim) {
                amplitude
            } else if (i + r * 17) % 997 == 0 {
                0.02
            } else {
                0.0
            };
            data.extend_from_slice(&value.to_le_bytes());
        }
    }
    let end = data.len();
    header.insert(
        name.to_owned(),
        json!({"dtype": "F32", "shape": [rank, input_dim], "data_offsets": [start, end]}),
    );
}

fn add_lora_b_tensor(
    header: &mut Map<String, Value>,
    data: &mut Vec<u8>,
    name: &str,
    output_dim: usize,
    rank: usize,
    amplitude: f32,
) {
    let start = data.len();
    for out in 0..output_dim {
        for r in 0..rank {
            let value = if out % rank == r {
                amplitude
            } else if (out + r * 31) % 409 == 0 {
                amplitude * 0.05
            } else {
                0.0
            };
            data.extend_from_slice(&value.to_le_bytes());
        }
    }
    let end = data.len();
    header.insert(
        name.to_owned(),
        json!({"dtype": "F32", "shape": [output_dim, rank], "data_offsets": [start, end]}),
    );
}

fn compatibility(model_identity: &ModelIdentity) -> AdapterCompatibility {
    AdapterCompatibility {
        base_model_id: BASE_MODEL_ID.to_owned(),
        base_weight_hash: model_identity.safetensors_inventory_sha256.clone(),
        tokenizer_hash: model_identity.tokenizer_sha256.clone(),
        chat_template_hash: model_identity.chat_template_sha256.clone(),
    }
}

fn target_config(args: &Args) -> LoadConfig {
    LoadConfig {
        model_path: args.model_path.display().to_string(),
        model_id: Some(BASE_MODEL_ID.to_owned()),
        model_revision: env::var("GEMMA4D_MODEL_REVISION").ok(),
        expected_architecture: Some("gemma4".to_owned()),
        max_context_tokens: NonZeroU32::new(args.max_context_tokens as u32)
            .expect("max_context_tokens is non-zero"),
        allow_unsupported_config: false,
    }
}

fn claim_inventory(records: &[P09Record]) -> ClaimInventory {
    let mut claims = ClaimInventory::default();
    for record in records {
        claims.correctness.push(format!(
            "adapter output differs={} token_delta={} logit_delta={:.6}",
            record.gate.adapter_output_differs_from_base,
            record.gate.prefill_greedy_token_differs,
            record.gate.prefill_greedy_logit_delta
        ));
        claims.correctness.push(format!(
            "manifest rejects wrong base={} base_hash={} tokenizer={} template={}",
            record.rejection.wrong_base_rejected,
            record.rejection.wrong_base_weight_hash_rejected,
            record.rejection.wrong_tokenizer_rejected,
            record.rejection.wrong_template_rejected
        ));
        claims.correctness.push(format!(
            "adapter KV namespace isolated={} wrong restore rejected={}",
            record.gate.kv_namespace_isolated, record.kv_namespace.wrong_adapter_restore_rejected
        ));
        claims.latency.push(format!(
            "base total {:.3} ms adapter total {:.3} ms base_after_clear total {:.3} ms",
            record.base.total_generation_ms,
            record.adapter_run.total_generation_ms,
            record.base_after_clear.total_generation_ms
        ));
        claims.latency.push(format!(
            "native load {} us activate {} us clear {} us",
            record.adapter.native_load_latency_us,
            record.hotswap.activate_us,
            record.hotswap.clear_us
        ));
        claims.memory.push(format!(
            "adapter resident bytes {} across {} modules",
            record.adapter.resident_bytes, record.adapter.module_count
        ));
        claims.defaults.push(format!(
            "MTP disabled with adapter={}",
            record.mtp_disabled_with_adapter
        ));
    }
    claims
}

fn render_report(summary: &P09Summary) -> String {
    let mut out = String::new();
    out.push_str("# P09 Real LoRA Adapter Hot Path\n\n");
    out.push_str(&format!("Status: `{}`\n\n", summary.status));
    out.push_str("## Run\n\n");
    out.push_str("| Field | Value |\n|---|---|\n");
    out.push_str(&format!("| Run ID | `{}` |\n", summary.run_id));
    out.push_str(&format!("| Mode | `{}` |\n", summary.mode));
    out.push_str(&format!("| Model path | `{}` |\n", summary.model_path));
    out.push_str(&format!(
        "| Context tokens | `{}` |\n",
        summary.context_tokens
    ));
    out.push_str(&format!(
        "| Decode tokens | `{}` |\n",
        summary.decode_tokens
    ));
    out.push_str(&format!("| Git | `{}` |\n", summary.environment.git_commit));
    out.push_str(&format!(
        "| Runtime | `{}` `{}` |\n",
        summary.environment.runtime_backend, summary.environment.runtime_backend_version
    ));

    if let Some(record) = summary.records.first() {
        out.push_str("\n## Adapter\n\n");
        out.push_str("| Field | Value |\n|---|---|\n");
        out.push_str(&format!(
            "| Adapter ID | `{}` |\n",
            record.adapter.adapter_id
        ));
        out.push_str(&format!(
            "| Target modules | `{}` |\n",
            record.adapter.target_modules.join(",")
        ));
        out.push_str(&format!("| Rank | `{}` |\n", record.adapter.rank));
        out.push_str(&format!("| Alpha | `{}` |\n", record.adapter.alpha));
        out.push_str(&format!(
            "| Module count | `{}` |\n",
            record.adapter.module_count
        ));
        out.push_str(&format!(
            "| Resident bytes | `{}` |\n",
            record.adapter.resident_bytes
        ));
        out.push_str(&format!(
            "| Native load latency us | `{}` |\n",
            record.adapter.native_load_latency_us
        ));
        out.push_str(&format!(
            "| Weights SHA256 | `{}` |\n",
            record.adapter.weights_sha256
        ));

        out.push_str("\n## Generation\n\n");
        out.push_str("| Run | Prefill ms | Decode ms | Total ms | Prefill token | Prefill logit | Generated tokens |\n");
        out.push_str("|---|---:|---:|---:|---:|---:|---|\n");
        for run in [&record.base, &record.adapter_run, &record.base_after_clear] {
            out.push_str(&format!(
                "| `{}` | {:.3} | {:.3} | {:.3} | {} | {:.6} | `{}` |\n",
                run.label,
                run.prefill_ms,
                run.decode_ms,
                run.total_generation_ms,
                run.prefill.greedy_token,
                run.prefill.greedy_logit,
                run.generated_tokens
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }

        out.push_str("\n## Gates\n\n");
        out.push_str("| Gate | Result |\n|---|---|\n");
        out.push_str(&format!(
            "| Adapter output differs | `{}` |\n",
            record.gate.adapter_output_differs_from_base
        ));
        out.push_str(&format!(
            "| Prefill greedy-logit delta | `{:.6}` |\n",
            record.gate.prefill_greedy_logit_delta
        ));
        out.push_str(&format!(
            "| Wrong manifest rejections | `{}` |\n",
            record.gate.wrong_manifest_rejections_pass
        ));
        out.push_str(&format!(
            "| KV namespace isolated | `{}` |\n",
            record.gate.kv_namespace_isolated
        ));
        out.push_str(&format!(
            "| Native adapter loaded | `{}` |\n",
            record.gate.native_adapter_loaded
        ));
        out.push_str(&format!(
            "| Hotswap measured | `{}` |\n",
            record.gate.hotswap_measured
        ));
        out.push_str(&format!(
            "| MTP disabled with adapter | `{}` |\n",
            record.gate.mtp_disabled_with_adapter
        ));
        out.push_str(&format!(
            "| Base restored after clear | `{}` |\n",
            record.hotswap.base_after_clear_matches_base
        ));

        out.push_str("\n## Hotswap\n\n");
        out.push_str("| Direction | Latency us | Active | Resident bytes |\n|---|---:|---|---:|\n");
        out.push_str(&format!(
            "| base-to-adapter | {} | `{}` | {} |\n",
            record.hotswap.activate_us,
            record.hotswap.activate_info.active,
            record.hotswap.activate_info.resident_bytes
        ));
        out.push_str(&format!(
            "| adapter-to-base | {} | `{}` | {} |\n",
            record.hotswap.clear_us,
            record.hotswap.clear_info.active,
            record.hotswap.clear_info.resident_bytes
        ));
    }

    out.push_str("\n## Verification Command\n\n```sh\n");
    out.push_str("GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example p09_real_lora_adapter -- --out-dir benchmarks/out/P09-real-lora-adapter --model-path artifacts/models/gemma-4-12B-it-4bit\n");
    out.push_str("```\n\n## Notes\n\n");
    for note in &summary.measurement_notes {
        out.push_str(&format!("- {note}.\n"));
    }
    out
}

fn render_blockers(summary: &P09Summary) -> String {
    if summary.blockers.is_empty() {
        return "# P09 Blockers\n\nNo blockers recorded.\n".to_owned();
    }
    let mut out = "# P09 Blockers\n\n".to_owned();
    for blocker in &summary.blockers {
        out.push_str(&format!("- {blocker}\n"));
    }
    out
}

fn write_jsonl(path: &Path, records: &[P09Record]) -> Result<(), Box<dyn std::error::Error>> {
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
        blockers.push("GEMMA4D_REQUIRE_MLX=1 is required for real native P09 evidence".to_owned());
    }
    if env::var("GEMMA4D_USE_NATIVE_GRAPH").ok().as_deref() != Some("1") {
        blockers
            .push("GEMMA4D_USE_NATIVE_GRAPH=1 is required for real native P09 evidence".to_owned());
    }
    blockers
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
    Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8_lossy(&output.stdout)
                    .trim()
                    .parse::<u64>()
                    .ok()
            } else {
                None
            }
        })
}

fn parse_positive_usize(value: &str, flag: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("{flag} must be an integer: {error}"))?;
    if parsed == 0 {
        return Err(format!("{flag} must be > 0").into());
    }
    Ok(parsed)
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn duration_us(duration: Duration) -> u128 {
    duration.as_micros()
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn run_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("p09-{millis}")
}

fn reset_dir(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    match fs::remove_dir_all(path) {
        Ok(()) => {}
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {}
        Err(source) => return Err(Box::new(source)),
    }
    fs::create_dir_all(path)?;
    Ok(())
}
