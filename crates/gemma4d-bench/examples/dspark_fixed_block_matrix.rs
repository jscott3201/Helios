use std::{
    env, fs,
    num::NonZeroU32,
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use gemma4d_bench::{BuildProvenance, capture_build_provenance, manifest};
use gemma4d_ffi::{
    DSparkDraftBlock, DSparkDrafter, DSparkTapConfig, KvCache, KvPolicy, LoadConfig, StepResult,
    Target, decode_one, prefill, runtime_version, verify_tokens,
};
use gemma4d_tokenizer::{file_sha256, sha256_hex};
use serde::{Deserialize, Serialize};

const GOAL: &str = "XR60-dspark-native-mlx";
const MODE: &str = "native_dspark_fixed_block_matrix";
const DEFAULT_OUT_DIR: &str = "benchmarks/out/XR60-dspark-native-mlx";
const DEFAULT_MODEL: &str = "artifacts/models/gemma-4-12B-it-4bit";
const DEFAULT_DRAFT: &str = "artifacts/drafts/dspark-gemma4-12b-block7";
const EXPECTED_DSPARK_REVISION: &str = "2fa72e765eec2965fc4d86a8663ce6769eba6218";
const EXPECTED_TARGET_LAYERS: &[u32] = &[5, 17, 29, 41, 46];
const DEFAULT_WORKLOADS: &[&str] = &["hello_smoke", "hello_reference_prefix"];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse(env::args().skip(1))?;
    fs::create_dir_all(&args.out_dir)?;

    let records_path = args.out_dir.join("records.jsonl");
    let summary_path = args.out_dir.join("summary.json");
    let report_path = args.out_dir.join("report.md");
    let blockers_path = args.out_dir.join("blockers.md");
    let decision_path = args.out_dir.join("decision.md");

    let build_provenance = capture_build_provenance()?;
    let model_identity =
        manifest::capture_artifact_identity(&args.model_path, "GEMMA4D_MODEL_REVISION");
    let draft_identity =
        manifest::capture_artifact_identity(&args.draft_path, "GEMMA4D_DSPARK_REVISION");
    let draft_config = inspect_dspark_config(&args.draft_path);
    let run_id = run_id();
    let mut blockers = startup_blockers(&args, &draft_config);
    let mut records = Vec::new();
    let mut native_tap_snapshots = Vec::new();
    let probes = match probes(&args) {
        Ok(probes) => probes,
        Err(err) => {
            blockers.push(format!("XR60 workload loading failed: {err}"));
            Vec::new()
        }
    };
    let selected_workload_ids = if args.workload_ids.is_empty() {
        probes.iter().map(|probe| probe.id.clone()).collect()
    } else {
        args.workload_ids.clone()
    };
    if blockers.is_empty() {
        eprintln!("XR60 DSpark loading target: {}", args.model_path.display());
        match Target::load(&target_config(&args)) {
            Ok(mut target) => match target.set_dspark_taps(&DSparkTapConfig::xr60_default()) {
                Ok(()) => {
                    eprintln!("XR60 DSpark loading drafter: {}", args.draft_path.display());
                    match DSparkDrafter::load(&dspark_config(&args), &target) {
                        Ok(drafter) => {
                            for probe in &probes {
                                if args.native_tap_snapshot_dir.is_some() {
                                    match dump_native_tap_snapshot(&target, &args, &run_id, probe) {
                                        Ok(snapshot) => native_tap_snapshots.push(snapshot),
                                        Err(err) => blockers.push(format!(
                                            "{} native tap snapshot failed: {}",
                                            probe.id, err
                                        )),
                                    }
                                }
                                eprintln!("XR60 DSpark baseline workload={}", probe.id);
                                let baseline = match run_baseline(&target, probe) {
                                    Ok(baseline) => baseline,
                                    Err(err) => {
                                        blockers
                                            .push(format!("{} baseline failed: {}", probe.id, err));
                                        continue;
                                    }
                                };
                                for block_size in &args.block_sizes {
                                    eprintln!(
                                        "XR60 DSpark record workload={} block_size={} max_new_tokens={}",
                                        probe.id, block_size, probe.max_new_tokens
                                    );
                                    match run_record(
                                        &target,
                                        &drafter,
                                        &run_id,
                                        probe,
                                        *block_size,
                                        &baseline,
                                    ) {
                                        Ok(record) => records.push(record),
                                        Err(err) => blockers.push(format!(
                                            "{} block_size={}: {}",
                                            probe.id, block_size, err
                                        )),
                                    }
                                }
                            }
                        }
                        Err(err) => {
                            blockers.push(format!("native DSpark drafter load failed: {err}"))
                        }
                    }
                }
                Err(err) => blockers.push(format!("native DSpark tap enable failed: {err}")),
            },
            Err(err) => blockers.push(format!("native DSpark target load failed: {err}")),
        }
    }
    if records.is_empty() {
        records = blocked_records(&args, &run_id, &blockers);
    }
    blockers.extend(
        records
            .iter()
            .filter(|record| record.measured && !record.exact)
            .map(|record| {
                format!(
                    "{} fixed-prefix {} did not match native greedy output",
                    record.workload_id, record.scheduled_len
                )
            }),
    );
    let status =
        if blockers.is_empty() && records.iter().all(|record| record.measured && record.exact) {
            "passed"
        } else if records.iter().any(|record| record.measured) {
            "failed"
        } else {
            "blocked"
        };
    let decision = if status == "passed" {
        "keep_disabled_pending_broader_evidence"
    } else {
        "blocked"
    };
    let summary = Summary {
        schema_version: 1,
        goal: GOAL,
        mode: MODE,
        status,
        decision,
        run_id,
        generated_at_unix_seconds: unix_now(),
        command: env::args().collect::<Vec<_>>().join(" "),
        build_provenance,
        model_identity,
        draft_identity,
        draft_config,
        records_path: records_path.display().to_string(),
        summary_path: summary_path.display().to_string(),
        report_path: report_path.display().to_string(),
        blockers_path: blockers_path.display().to_string(),
        decision_path: decision_path.display().to_string(),
        native_tap_snapshot_manifest_path: native_tap_snapshot_manifest_path(&args),
        block_sizes: args.block_sizes.clone(),
        workload_ids: selected_workload_ids,
        max_new_tokens: args.max_new_tokens,
        target_layer_ids: EXPECTED_TARGET_LAYERS.to_vec(),
        native_tap_snapshots: native_tap_snapshots.clone(),
        records: records.clone(),
        blockers: blockers.clone(),
        measurement_notes: vec![
            "This XR60 harness uses native target prefill/decode as the baseline and native DSpark draft plus gemma4_verify_tokens for speculative records.",
            "The DSpark path remains default-off regardless of this harness outcome; broader workload, parity, and memory gates are still required.",
            "The draft artifact is expected to be deepseek-ai/dspark_gemma4_12b_block7 at revision 2fa72e765eec2965fc4d86a8663ce6769eba6218.",
        ],
    };

    fs::create_dir_all(&args.out_dir)?;
    write_jsonl(&records_path, &records)?;
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)?;
    fs::write(&report_path, render_report(&summary))?;
    fs::write(&blockers_path, render_blockers(&summary))?;
    fs::write(&decision_path, format!("{decision}\n"))?;
    if let Some(snapshot_dir) = &args.native_tap_snapshot_dir {
        fs::create_dir_all(snapshot_dir)?;
        let manifest_path = snapshot_dir.join("native_tap_snapshot_manifest.json");
        let manifest = NativeTapSnapshotManifest {
            schema_version: 1,
            goal: GOAL,
            phase: "02-hidden-tap-parity",
            status: if native_tap_snapshots.is_empty() {
                "blocked"
            } else {
                "ready_for_reference_compare"
            },
            run_id: summary.run_id.clone(),
            generated_at_unix_seconds: summary.generated_at_unix_seconds,
            command: summary.command.clone(),
            target_layer_ids: EXPECTED_TARGET_LAYERS.to_vec(),
            snapshots: native_tap_snapshots,
            blockers: blockers.clone(),
        };
        fs::write(manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
    }

    println!("XR60 DSpark fixed-block matrix: {decision}");
    println!("records: {}", records_path.display());
    println!("summary: {}", summary_path.display());
    println!("report: {}", report_path.display());
    println!("blockers: {}", blockers_path.display());
    println!("decision: {}", decision_path.display());
    Ok(())
}

#[derive(Debug, Deserialize)]
struct TokenWorkloadRecord {
    workload_id: String,
    token_ids: Vec<i32>,
    #[serde(default)]
    max_new_tokens: Option<usize>,
    #[serde(default)]
    actual_context_tokens: Option<usize>,
    #[serde(default)]
    prompt_sha256: Option<String>,
}

#[derive(Debug)]
struct Args {
    out_dir: PathBuf,
    model_path: PathBuf,
    draft_path: PathBuf,
    block_sizes: Vec<usize>,
    workload_ids: Vec<String>,
    max_new_tokens: usize,
    max_context_tokens: usize,
    native_tap_snapshot_dir: Option<PathBuf>,
    token_workload_path: Option<PathBuf>,
}

impl Args {
    fn parse<I>(args: I) -> Result<Self, Box<dyn std::error::Error>>
    where
        I: IntoIterator<Item = String>,
    {
        let mut out_dir = PathBuf::from(DEFAULT_OUT_DIR);
        let mut model_path = PathBuf::from(DEFAULT_MODEL);
        let mut draft_path = PathBuf::from(DEFAULT_DRAFT);
        let mut block_sizes = vec![1, 2, 4, 7];
        let mut workload_ids = DEFAULT_WORKLOADS
            .iter()
            .map(|workload| (*workload).to_owned())
            .collect::<Vec<_>>();
        let mut workload_filter_explicit = false;
        let mut max_new_tokens = 32;
        let mut max_context_tokens = 8192;
        let mut native_tap_snapshot_dir = None;
        let mut token_workload_path = None;

        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--out-dir" => out_dir = PathBuf::from(required_value(&mut args, "--out-dir")?),
                "--model-path" => {
                    model_path = PathBuf::from(required_value(&mut args, "--model-path")?)
                }
                "--draft-path" => {
                    draft_path = PathBuf::from(required_value(&mut args, "--draft-path")?)
                }
                "--block-sizes" => {
                    block_sizes = parse_usize_list(&required_value(&mut args, "--block-sizes")?)?;
                }
                "--workloads" => {
                    workload_filter_explicit = true;
                    workload_ids = parse_string_list(&required_value(&mut args, "--workloads")?)?;
                }
                "--max-new-tokens" => {
                    max_new_tokens = required_value(&mut args, "--max-new-tokens")?
                        .parse()
                        .map_err(|_| "--max-new-tokens must be an integer")?;
                }
                "--max-context-tokens" => {
                    max_context_tokens = required_value(&mut args, "--max-context-tokens")?
                        .parse()
                        .map_err(|_| "--max-context-tokens must be an integer")?;
                }
                "--native-tap-snapshot-dir" => {
                    native_tap_snapshot_dir = Some(PathBuf::from(required_value(
                        &mut args,
                        "--native-tap-snapshot-dir",
                    )?));
                }
                "--token-workloads" => {
                    token_workload_path = Some(PathBuf::from(required_value(
                        &mut args,
                        "--token-workloads",
                    )?));
                }
                "-h" | "--help" => {
                    println!(
                        "usage: GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- [--out-dir PATH] [--model-path PATH] [--draft-path PATH] [--block-sizes 1,2,4,7] [--workloads hello_smoke,hello_reference_prefix] [--token-workloads PATH] [--max-new-tokens N] [--max-context-tokens N] [--native-tap-snapshot-dir PATH]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown option '{other}'").into()),
            }
        }
        if block_sizes.is_empty() {
            return Err("--block-sizes must not be empty".into());
        }
        if block_sizes
            .iter()
            .any(|size| !matches!(*size, 1 | 2 | 4 | 7))
        {
            return Err("XR60 fixed-prefix scaffold only accepts block sizes 1,2,4,7".into());
        }
        if workload_ids.is_empty() && token_workload_path.is_none() {
            return Err("--workloads must not be empty".into());
        }
        if token_workload_path.is_none() {
            for workload_id in &workload_ids {
                if !DEFAULT_WORKLOADS.contains(&workload_id.as_str()) {
                    return Err(format!(
                        "unknown XR60 workload '{workload_id}'; expected one of {}",
                        DEFAULT_WORKLOADS.join(",")
                    )
                    .into());
                }
            }
        }
        if token_workload_path.is_some() && !workload_filter_explicit {
            workload_ids.clear();
        }
        if max_new_tokens == 0 {
            return Err("--max-new-tokens must be greater than zero".into());
        }
        if max_context_tokens == 0 {
            return Err("--max-context-tokens must be greater than zero".into());
        }
        Ok(Self {
            out_dir,
            model_path,
            draft_path,
            block_sizes,
            workload_ids,
            max_new_tokens,
            max_context_tokens,
            native_tap_snapshot_dir,
            token_workload_path,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
struct Summary {
    schema_version: u32,
    goal: &'static str,
    mode: &'static str,
    status: &'static str,
    decision: &'static str,
    run_id: String,
    generated_at_unix_seconds: u64,
    command: String,
    build_provenance: BuildProvenance,
    model_identity: manifest::ArtifactIdentity,
    draft_identity: manifest::ArtifactIdentity,
    draft_config: DsparkConfigInspection,
    records_path: String,
    summary_path: String,
    report_path: String,
    blockers_path: String,
    decision_path: String,
    native_tap_snapshot_manifest_path: Option<String>,
    block_sizes: Vec<usize>,
    workload_ids: Vec<String>,
    max_new_tokens: usize,
    target_layer_ids: Vec<u32>,
    native_tap_snapshots: Vec<NativeTapSnapshot>,
    records: Vec<Record>,
    blockers: Vec<String>,
    measurement_notes: Vec<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
struct NativeTapSnapshotManifest {
    schema_version: u32,
    goal: &'static str,
    phase: &'static str,
    status: &'static str,
    run_id: String,
    generated_at_unix_seconds: u64,
    command: String,
    target_layer_ids: Vec<u32>,
    snapshots: Vec<NativeTapSnapshot>,
    blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct NativeTapSnapshot {
    workload_id: String,
    prompt_tokens: Vec<i32>,
    prompt_sha256: String,
    snapshot_path: String,
    prefill_greedy_token: i32,
    prefill_greedy_logit: f32,
    tap_layer_ids: Vec<u32>,
    tap_shapes: Vec<Vec<u64>>,
    tap_bytes: u64,
    hidden_present: bool,
    context_tokens: usize,
    active_kv_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
struct Record {
    schema_version: u32,
    goal: &'static str,
    mode: &'static str,
    run_id: String,
    status: &'static str,
    measured: bool,
    exact: bool,
    workload_id: String,
    context_tokens: usize,
    max_new_tokens: usize,
    scheduler: &'static str,
    scheduled_len: usize,
    warmup_target_tokens: usize,
    attempted_draft_tokens: usize,
    scheduled_draft_tokens: usize,
    accepted_draft_tokens: usize,
    accepted_tokens_per_verify: f64,
    acceptance_rate: f64,
    target_verify_passes: usize,
    rollback_count: usize,
    draft_ms: Option<f64>,
    scheduler_us: Option<u64>,
    verify_stage_ms: Option<f64>,
    verify_forward_ms: Option<f64>,
    repair_ms: Option<f64>,
    decode_tokens_per_second: Option<f64>,
    decode_phase_ms: Option<f64>,
    peak_memory_gb: Option<f64>,
    peak_rss_mb: Option<f64>,
    active_kv_bytes: Option<u64>,
    hidden_tap_bytes: Option<u64>,
    draft_resident_bytes: Option<u64>,
    baseline_token_sequence_sha256: Option<String>,
    dspark_token_sequence_sha256: Option<String>,
    verify_trace: Vec<VerifyTraceRecord>,
    auto_disable_reason: String,
}

#[derive(Debug, Clone, Serialize)]
struct VerifyTraceRecord {
    verify_index: usize,
    scheduled_len: usize,
    draft_tokens: Vec<i32>,
    draft_logits: Vec<f32>,
    draft_margins: Vec<f32>,
    draft_confidence: Vec<f32>,
    accepted_draft_count: u32,
    committed_tokens: Vec<i32>,
    target_tokens: Vec<i32>,
    target_logits: Vec<f32>,
    position_offsets: Vec<u64>,
    draft_in_top_k: Vec<bool>,
    target_top_token_ids: Vec<Vec<i32>>,
    target_top_logits: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize)]
struct DsparkConfigInspection {
    path: String,
    config_exists: bool,
    config_sha256: Option<String>,
    architecture: Option<String>,
    block_size: Option<u64>,
    target_layer_ids: Vec<u32>,
    num_hidden_layers: Option<u64>,
    markov_rank: Option<u64>,
    num_anchors: Option<u64>,
    mask_token_id: Option<u64>,
    dtype: Option<String>,
    checks_passed: bool,
    issues: Vec<String>,
}

#[derive(Debug, Clone)]
struct Probe {
    id: String,
    prompt_tokens: Vec<i32>,
    max_new_tokens: usize,
}

#[derive(Debug, Clone)]
struct NativeRun {
    generated_tokens: Vec<i32>,
}

#[derive(Debug, Clone)]
struct NativeDSparkRun {
    generated_tokens: Vec<i32>,
    warmup_target_tokens: usize,
    draft_ms: f64,
    scheduler_us: u64,
    verify_stage_ms: f64,
    verify_forward_ms: f64,
    repair_ms: f64,
    total_ms: f64,
    attempted_draft_tokens: usize,
    scheduled_draft_tokens: usize,
    accepted_draft_tokens: usize,
    target_verify_passes: usize,
    rollback_count: usize,
    peak_memory_gb: f32,
    peak_rss_mb: f32,
    active_kv_bytes: u64,
    hidden_tap_bytes: u64,
    verify_trace: Vec<VerifyTraceRecord>,
}

fn inspect_dspark_config(draft_path: &Path) -> DsparkConfigInspection {
    let config_path = draft_path.join("config.json");
    if !config_path.exists() {
        return DsparkConfigInspection {
            path: config_path.display().to_string(),
            config_exists: false,
            config_sha256: None,
            architecture: None,
            block_size: None,
            target_layer_ids: Vec::new(),
            num_hidden_layers: None,
            markov_rank: None,
            num_anchors: None,
            mask_token_id: None,
            dtype: None,
            checks_passed: false,
            issues: vec![format!("missing DSpark config: {}", config_path.display())],
        };
    }
    let bytes = fs::read(&config_path).unwrap_or_default();
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    let architecture = value
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .and_then(|items| items.first())
        .and_then(serde_json::Value::as_str)
        .map(ToOwned::to_owned);
    let target_layer_ids = value
        .get("target_layer_ids")
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(serde_json::Value::as_u64)
                .filter_map(|value| u32::try_from(value).ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let block_size = value.get("block_size").and_then(serde_json::Value::as_u64);
    let num_hidden_layers = value
        .get("num_hidden_layers")
        .and_then(serde_json::Value::as_u64);
    let markov_rank = value.get("markov_rank").and_then(serde_json::Value::as_u64);
    let num_anchors = value.get("num_anchors").and_then(serde_json::Value::as_u64);
    let mask_token_id = value
        .get("mask_token_id")
        .and_then(serde_json::Value::as_u64);
    let dtype = value
        .get("dtype")
        .and_then(serde_json::Value::as_str)
        .map(ToOwned::to_owned);
    let mut issues = Vec::new();
    if architecture.as_deref() != Some("Gemma4DSparkModel") {
        issues.push("architecture is not Gemma4DSparkModel".to_owned());
    }
    if block_size != Some(7) {
        issues.push("block_size is not 7".to_owned());
    }
    if target_layer_ids != EXPECTED_TARGET_LAYERS {
        issues.push("target_layer_ids are not [5,17,29,41,46]".to_owned());
    }
    if num_hidden_layers != Some(5) {
        issues.push("num_hidden_layers is not 5".to_owned());
    }
    if markov_rank != Some(256) {
        issues.push("markov_rank is not 256".to_owned());
    }
    if num_anchors != Some(512) {
        issues.push("num_anchors is not 512".to_owned());
    }
    if mask_token_id != Some(4) {
        issues.push("mask_token_id is not 4".to_owned());
    }
    if dtype.as_deref() != Some("bfloat16") {
        issues.push("dtype is not bfloat16".to_owned());
    }
    DsparkConfigInspection {
        path: config_path.display().to_string(),
        config_exists: true,
        config_sha256: file_sha256(&config_path).ok(),
        architecture,
        block_size,
        target_layer_ids,
        num_hidden_layers,
        markov_rank,
        num_anchors,
        mask_token_id,
        dtype,
        checks_passed: issues.is_empty(),
        issues,
    }
}

fn startup_blockers(args: &Args, draft_config: &DsparkConfigInspection) -> Vec<String> {
    let mut blockers = Vec::new();
    if runtime_version()
        .map(|version| version.backend_version == "m03-smoke-no-mlx")
        .unwrap_or(true)
    {
        blockers.push(
            "GEMMA4D_REQUIRE_MLX=1 is required at build time for XR60 native DSpark".to_owned(),
        );
    }
    if env::var_os("GEMMA4D_USE_NATIVE_GRAPH").is_none() {
        blockers.push("GEMMA4D_USE_NATIVE_GRAPH=1 is required for XR60 native DSpark".to_owned());
    }
    if !args.model_path.exists() {
        blockers.push(format!(
            "missing target model path: {}",
            args.model_path.display()
        ));
    }
    if !args.draft_path.exists() {
        blockers.push(format!(
            "missing DSpark draft path: {}",
            args.draft_path.display()
        ));
    }
    let weights_path = args.draft_path.join("model.safetensors");
    if !weights_path.exists() {
        blockers.push(format!(
            "missing DSpark draft weights: {}; download deepseek-ai/dspark_gemma4_12b_block7 before running fixed-prefix benchmarks",
            weights_path.display()
        ));
    }
    if !draft_config.checks_passed {
        blockers.extend(
            draft_config
                .issues
                .iter()
                .map(|issue| format!("DSpark config validation failed: {issue}")),
        );
    }
    blockers
}

fn probes(args: &Args) -> Result<Vec<Probe>, Box<dyn std::error::Error>> {
    if let Some(path) = &args.token_workload_path {
        load_token_workload_probes(path, args)
    } else {
        Ok(builtin_probes(args.max_new_tokens, &args.workload_ids))
    }
}

fn builtin_probes(max_new_tokens: usize, workload_ids: &[String]) -> Vec<Probe> {
    let all = vec![
        Probe {
            id: "hello_smoke".to_owned(),
            prompt_tokens: vec![9259],
            max_new_tokens,
        },
        Probe {
            id: "hello_reference_prefix".to_owned(),
            prompt_tokens: vec![9259, 236772, 236772],
            max_new_tokens,
        },
    ];
    all.into_iter()
        .filter(|probe| workload_ids.iter().any(|id| id == &probe.id))
        .collect()
}

fn load_token_workload_probes(
    path: &Path,
    args: &Args,
) -> Result<Vec<Probe>, Box<dyn std::error::Error>> {
    let body = fs::read_to_string(path)
        .map_err(|err| format!("could not read token workloads {}: {err}", path.display()))?;
    let mut probes = Vec::new();
    for (line_index, line) in body.lines().enumerate() {
        let line_number = line_index + 1;
        if line.trim().is_empty() {
            continue;
        }
        let record: TokenWorkloadRecord = serde_json::from_str(line)
            .map_err(|err| format!("could not parse {}:{line_number}: {err}", path.display()))?;
        if !args.workload_ids.is_empty()
            && !args
                .workload_ids
                .iter()
                .any(|workload_id| workload_id == &record.workload_id)
        {
            continue;
        }
        if record.token_ids.is_empty() {
            return Err(format!(
                "{}:{line_number} workload {} has no token_ids",
                path.display(),
                record.workload_id
            )
            .into());
        }
        if record.token_ids.iter().any(|token| *token < 0) {
            return Err(format!(
                "{}:{line_number} workload {} contains a negative token id",
                path.display(),
                record.workload_id
            )
            .into());
        }
        if let Some(actual_context_tokens) = record.actual_context_tokens {
            if actual_context_tokens != record.token_ids.len() {
                return Err(format!(
                    "{}:{line_number} workload {} actual_context_tokens={} but token_ids has {} entries",
                    path.display(),
                    record.workload_id,
                    actual_context_tokens,
                    record.token_ids.len()
                )
                .into());
            }
        }
        if record.token_ids.len() > args.max_context_tokens {
            return Err(format!(
                "{}:{line_number} workload {} has {} prompt tokens, exceeding --max-context-tokens {}",
                path.display(),
                record.workload_id,
                record.token_ids.len(),
                args.max_context_tokens
            )
            .into());
        }
        if matches!(record.prompt_sha256.as_deref(), Some("")) {
            return Err(format!(
                "{}:{line_number} workload {} has an empty prompt_sha256",
                path.display(),
                record.workload_id
            )
            .into());
        }
        let max_new_tokens = record
            .max_new_tokens
            .unwrap_or(args.max_new_tokens)
            .min(args.max_new_tokens);
        if max_new_tokens == 0 {
            return Err(format!(
                "{}:{line_number} workload {} resolves to zero max_new_tokens",
                path.display(),
                record.workload_id
            )
            .into());
        }
        probes.push(Probe {
            id: record.workload_id,
            prompt_tokens: record.token_ids,
            max_new_tokens,
        });
    }
    if probes.is_empty() {
        let filter = if args.workload_ids.is_empty() {
            "<all>".to_owned()
        } else {
            args.workload_ids.join(",")
        };
        return Err(format!(
            "no token workload records selected from {} with filter {}",
            path.display(),
            filter
        )
        .into());
    }
    Ok(probes)
}

fn run_record(
    target: &Target,
    drafter: &DSparkDrafter,
    run_id: &str,
    probe: &Probe,
    block_size: usize,
    baseline: &NativeRun,
) -> Result<Record, Box<dyn std::error::Error>> {
    let dspark = run_dspark(target, drafter, probe, block_size)?;
    let exact = baseline.generated_tokens == dspark.generated_tokens;
    Ok(Record {
        schema_version: 1,
        goal: GOAL,
        mode: MODE,
        run_id: run_id.to_owned(),
        status: if exact { "passed" } else { "failed" },
        measured: true,
        exact,
        workload_id: probe.id.clone(),
        context_tokens: probe.prompt_tokens.len(),
        max_new_tokens: probe.max_new_tokens,
        scheduler: "fixed",
        scheduled_len: block_size,
        warmup_target_tokens: dspark.warmup_target_tokens,
        attempted_draft_tokens: dspark.attempted_draft_tokens,
        scheduled_draft_tokens: dspark.scheduled_draft_tokens,
        accepted_draft_tokens: dspark.accepted_draft_tokens,
        accepted_tokens_per_verify: if dspark.target_verify_passes == 0 {
            0.0
        } else {
            dspark.accepted_draft_tokens as f64 / dspark.target_verify_passes as f64
        },
        acceptance_rate: if dspark.attempted_draft_tokens == 0 {
            0.0
        } else {
            dspark.accepted_draft_tokens as f64 / dspark.attempted_draft_tokens as f64
        },
        target_verify_passes: dspark.target_verify_passes,
        rollback_count: dspark.rollback_count,
        draft_ms: Some(dspark.draft_ms),
        scheduler_us: Some(dspark.scheduler_us),
        verify_stage_ms: Some(dspark.verify_stage_ms),
        verify_forward_ms: Some(dspark.verify_forward_ms),
        repair_ms: Some(dspark.repair_ms),
        decode_tokens_per_second: Some(tps(probe.max_new_tokens, dspark.total_ms)),
        decode_phase_ms: Some(dspark.total_ms),
        peak_memory_gb: Some(dspark.peak_memory_gb as f64),
        peak_rss_mb: Some(dspark.peak_rss_mb as f64),
        active_kv_bytes: Some(dspark.active_kv_bytes),
        hidden_tap_bytes: Some(dspark.hidden_tap_bytes),
        draft_resident_bytes: None,
        baseline_token_sequence_sha256: Some(checksum_tokens(&baseline.generated_tokens)),
        dspark_token_sequence_sha256: Some(checksum_tokens(&dspark.generated_tokens)),
        verify_trace: dspark.verify_trace,
        auto_disable_reason: if exact {
            String::new()
        } else {
            format!(
                "baseline tokens {:?} != dspark tokens {:?}",
                baseline.generated_tokens, dspark.generated_tokens
            )
        },
    })
}

fn run_baseline(target: &Target, probe: &Probe) -> Result<NativeRun, Box<dyn std::error::Error>> {
    let mut cache = KvCache::create(&KvPolicy::default())?;
    let first = prefill(target, &mut cache, &probe.prompt_tokens)?;
    let mut generated = Vec::with_capacity(probe.max_new_tokens);
    generated.push(first.greedy_token);

    while generated.len() < probe.max_new_tokens {
        let token = *generated.last().expect("generated has token");
        let step = decode_one(target, &mut cache, token)?;
        generated.push(step.greedy_token);
    }

    Ok(NativeRun {
        generated_tokens: generated,
    })
}

fn dump_native_tap_snapshot(
    target: &Target,
    args: &Args,
    run_id: &str,
    probe: &Probe,
) -> Result<NativeTapSnapshot, Box<dyn std::error::Error>> {
    let snapshot_dir = args
        .native_tap_snapshot_dir
        .as_ref()
        .ok_or("native tap snapshot dir is not configured")?;
    fs::create_dir_all(snapshot_dir)?;

    let mut cache = KvCache::create(&KvPolicy::default())?;
    let first = prefill(target, &mut cache, &probe.prompt_tokens)?;
    let tap_info = cache.dspark_tap_info()?;
    let snapshot = cache.export_snapshot()?;
    let snapshot_path = snapshot_dir.join(format!("{}-{}.safetensors", run_id, probe.id));
    snapshot.save_to_path(&snapshot_path)?;

    Ok(NativeTapSnapshot {
        workload_id: probe.id.clone(),
        prompt_tokens: probe.prompt_tokens.clone(),
        prompt_sha256: checksum_tokens(&probe.prompt_tokens),
        snapshot_path: snapshot_path.display().to_string(),
        prefill_greedy_token: first.greedy_token,
        prefill_greedy_logit: first.greedy_logit,
        tap_layer_ids: tap_info.layer_ids,
        tap_shapes: tap_info.tap_shapes,
        tap_bytes: tap_info.tap_bytes,
        hidden_present: tap_info.has_last_hidden,
        context_tokens: probe.prompt_tokens.len(),
        active_kv_bytes: first.active_kv_bytes,
    })
}

fn run_dspark(
    target: &Target,
    drafter: &DSparkDrafter,
    probe: &Probe,
    block_size: usize,
) -> Result<NativeDSparkRun, Box<dyn std::error::Error>> {
    let started = Instant::now();
    let mut cache = KvCache::create(&KvPolicy::default())?;

    let prefill_started = Instant::now();
    let first = prefill(target, &mut cache, &probe.prompt_tokens)?;
    let _prefill_ms = duration_ms(prefill_started.elapsed());
    let mut generated = Vec::with_capacity(probe.max_new_tokens);
    let mut peak_memory_gb = first.peak_memory_gb;
    let mut peak_rss_mb = first.peak_rss_mb;
    let mut active_kv_bytes = first.active_kv_bytes;
    let warmup_target_tokens = 1_usize;
    let mut draft_ms = 0.0;
    let mut scheduler_us = 0_u64;
    let mut verify_stage_ms = 0.0;
    let mut verify_forward_ms = 0.0;
    let mut repair_ms = 0.0;
    let mut attempted_draft_tokens = 0_usize;
    let mut scheduled_draft_tokens = 0_usize;
    let mut accepted_draft_tokens = 0_usize;
    let mut target_verify_passes = 0_usize;
    let mut rollback_count = 0_usize;
    let mut verify_trace = Vec::new();

    generated.push(first.greedy_token);
    if generated.len() < probe.max_new_tokens {
        let warmup = decode_one(target, &mut cache, first.greedy_token)?;
        peak_memory_gb = peak_memory_gb.max(warmup.peak_memory_gb);
        peak_rss_mb = peak_rss_mb.max(warmup.peak_rss_mb);
        active_kv_bytes = active_kv_bytes.max(warmup.active_kv_bytes);
    }
    let tap_info = cache.dspark_tap_info()?;
    let hidden_tap_bytes = tap_info.tap_bytes;

    while generated.len() < probe.max_new_tokens {
        let remaining = probe.max_new_tokens - generated.len();
        let scheduled = block_size.min(remaining);
        let draft = drafter.draft_block(
            &mut cache,
            NonZeroU32::new(scheduled as u32).expect("scheduled block is non-zero"),
        )?;
        let draft_tokens = dspark_tokens(&draft);
        if draft_tokens.is_empty() {
            return Err("native DSpark drafter returned no tokens".into());
        }
        draft_ms += draft.draft_ms;
        scheduler_us = scheduler_us.saturating_add(draft.scheduler_us);
        scheduled_draft_tokens += scheduled;
        attempted_draft_tokens += draft_tokens.len();
        target_verify_passes += 1;

        let step = verify_tokens(target, &mut cache, &draft_tokens)?;
        verify_stage_ms += step.verify_stage_ms;
        verify_forward_ms += step.verify_forward_ms;
        repair_ms += step.verify_repair_ms;
        peak_memory_gb = peak_memory_gb.max(step.peak_memory_gb);
        peak_rss_mb = peak_rss_mb.max(step.peak_rss_mb);
        active_kv_bytes = active_kv_bytes.max(step.active_kv_bytes);
        accepted_draft_tokens += usize::try_from(step.accepted_draft_count)
            .unwrap_or(usize::MAX)
            .min(draft_tokens.len());
        if usize::try_from(step.accepted_draft_count).unwrap_or(usize::MAX) < draft_tokens.len() {
            rollback_count += 1;
        }
        verify_trace.push(verify_trace_record(
            target_verify_passes - 1,
            scheduled,
            &draft,
            &step,
        ));
        let committed = step.committed_tokens();
        if committed.is_empty() {
            return Err("gemma4_verify_tokens committed no tokens".into());
        }
        for token in committed {
            if generated.len() < probe.max_new_tokens {
                generated.push(*token);
            }
        }
    }

    Ok(NativeDSparkRun {
        generated_tokens: generated,
        warmup_target_tokens,
        draft_ms,
        scheduler_us,
        verify_stage_ms,
        verify_forward_ms,
        repair_ms,
        total_ms: duration_ms(started.elapsed()),
        attempted_draft_tokens,
        scheduled_draft_tokens,
        accepted_draft_tokens,
        target_verify_passes,
        rollback_count,
        peak_memory_gb,
        peak_rss_mb,
        active_kv_bytes,
        hidden_tap_bytes,
        verify_trace,
    })
}

fn target_config(args: &Args) -> LoadConfig {
    LoadConfig {
        model_path: args.model_path.display().to_string(),
        model_id: Some("mlx-community/gemma-4-12B-it-4bit".to_owned()),
        model_revision: None,
        expected_architecture: Some("gemma4".to_owned()),
        max_context_tokens: NonZeroU32::new(args.max_context_tokens as u32)
            .expect("max context is non-zero"),
        allow_unsupported_config: false,
    }
}

fn dspark_config(args: &Args) -> LoadConfig {
    LoadConfig {
        model_path: args.draft_path.display().to_string(),
        model_id: Some("deepseek-ai/dspark_gemma4_12b_block7".to_owned()),
        model_revision: Some(EXPECTED_DSPARK_REVISION.to_owned()),
        expected_architecture: Some("Gemma4DSparkModel".to_owned()),
        max_context_tokens: NonZeroU32::new(args.max_context_tokens as u32)
            .expect("max context is non-zero"),
        allow_unsupported_config: false,
    }
}

fn dspark_tokens(draft: &DSparkDraftBlock) -> Vec<i32> {
    draft.tokens.iter().map(|token| token.token).collect()
}

fn blocked_records(args: &Args, run_id: &str, blockers: &[String]) -> Vec<Record> {
    let reason = blockers.join("; ");
    args.block_sizes
        .iter()
        .map(|scheduled_len| Record {
            schema_version: 1,
            goal: GOAL,
            mode: MODE,
            run_id: run_id.to_owned(),
            status: "blocked",
            measured: false,
            exact: false,
            workload_id: "startup".to_owned(),
            context_tokens: 0,
            max_new_tokens: args.max_new_tokens,
            scheduler: "fixed",
            scheduled_len: *scheduled_len,
            warmup_target_tokens: 0,
            attempted_draft_tokens: 0,
            scheduled_draft_tokens: 0,
            accepted_draft_tokens: 0,
            accepted_tokens_per_verify: 0.0,
            acceptance_rate: 0.0,
            target_verify_passes: 0,
            rollback_count: 0,
            draft_ms: None,
            scheduler_us: None,
            verify_stage_ms: None,
            verify_forward_ms: None,
            repair_ms: None,
            decode_tokens_per_second: None,
            decode_phase_ms: None,
            peak_memory_gb: None,
            peak_rss_mb: None,
            active_kv_bytes: None,
            hidden_tap_bytes: None,
            draft_resident_bytes: None,
            baseline_token_sequence_sha256: None,
            dspark_token_sequence_sha256: None,
            verify_trace: Vec::new(),
            auto_disable_reason: reason.clone(),
        })
        .collect()
}

fn verify_trace_record(
    verify_index: usize,
    scheduled_len: usize,
    draft: &DSparkDraftBlock,
    step: &StepResult,
) -> VerifyTraceRecord {
    VerifyTraceRecord {
        verify_index,
        scheduled_len,
        draft_tokens: draft.tokens.iter().map(|token| token.token).collect(),
        draft_logits: draft.tokens.iter().map(|token| token.logit).collect(),
        draft_margins: draft.tokens.iter().map(|token| token.margin).collect(),
        draft_confidence: draft.tokens.iter().map(|token| token.confidence).collect(),
        accepted_draft_count: step.accepted_draft_count,
        committed_tokens: step.committed_tokens().to_vec(),
        target_tokens: step.mtp_trace.target_tokens.clone(),
        target_logits: step.mtp_trace.target_logits.clone(),
        position_offsets: step.mtp_trace.position_offsets.clone(),
        draft_in_top_k: step.mtp_trace.draft_in_top_k.clone(),
        target_top_token_ids: step.mtp_trace.top_token_ids.clone(),
        target_top_logits: step.mtp_trace.top_logits.clone(),
    }
}

fn render_report(summary: &Summary) -> String {
    let mut out = String::new();
    out.push_str("# XR60 DSpark native MLX report\n\n");
    out.push_str("## Decision\n");
    out.push_str(summary.decision);
    out.push_str("\n\n## Git and environment\n\n");
    out.push_str(&format!(
        "- Git SHA: `{}`\n",
        escape_md(&summary.build_provenance.git_sha)
    ));
    out.push_str(&format!(
        "- Git status: `{}`\n",
        escape_md(&summary.build_provenance.git_status_short)
    ));
    out.push_str(&format!("- Mode: `{}`\n", summary.mode));
    out.push_str(&format!(
        "- Workloads: `{}`\n",
        escape_md(&summary.workload_ids.join(","))
    ));
    out.push_str(&format!(
        "- Draft path: `{}`\n",
        escape_md(&summary.draft_identity.path)
    ));
    out.push_str(&format!(
        "- Expected DSpark revision: `{EXPECTED_DSPARK_REVISION}`\n\n"
    ));
    out.push_str("## What changed\n\n");
    out.push_str(
        "This artifact was generated by the XR60 fixed-prefix harness. When startup gates pass, records warm-start with the first target greedy token, then use native DSpark draft blocks and `gemma4_verify_tokens` commit/rollback semantics. The warm-start matches DeepSpec's evaluator flow where the prompt prefill commits one target token before DSpark proposes from that current-token anchor.\n\n",
    );
    out.push_str("## Correctness results\n\n");
    if summary.records.iter().any(|record| record.measured) {
        for record in &summary.records {
            out.push_str(&format!(
                "- {} block_size={}: exact={} baseline_sha={} dspark_sha={}\n",
                record.workload_id,
                record.scheduled_len,
                record.exact,
                record
                    .baseline_token_sequence_sha256
                    .as_deref()
                    .unwrap_or("n/a"),
                record
                    .dspark_token_sequence_sha256
                    .as_deref()
                    .unwrap_or("n/a")
            ));
        }
        out.push('\n');
    } else {
        out.push_str("No DSpark exactness records were measured because startup blockers fired before native drafting.\n\n");
    }
    out.push_str("## Benchmark summary\n\n");
    out.push_str("| workload | scheduler | block/max | warmup | exact | decode tok/s | speedup | acceptance | accepted/verify | draft ms | verify fwd ms | peak GB | active KV bytes |\n");
    out.push_str("|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for record in &summary.records {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | n/a | {:.3} | {:.3} | {} | {} | {} | {} |\n",
            record.workload_id,
            record.scheduler,
            record.scheduled_len,
            record.warmup_target_tokens,
            record.exact,
            fmt_f64(record.decode_tokens_per_second),
            record.acceptance_rate,
            record.accepted_tokens_per_verify,
            fmt_f64(record.draft_ms),
            fmt_f64(record.verify_forward_ms),
            fmt_f64(record.peak_memory_gb),
            fmt_u64(record.active_kv_bytes)
        ));
    }
    out.push_str("\n## Verify trace\n\n");
    if summary
        .records
        .iter()
        .any(|record| !record.verify_trace.is_empty())
    {
        for record in &summary.records {
            if let Some(first_trace) = record.verify_trace.first() {
                out.push_str(&format!(
                    "- {} block_size={}: first_draft={:?} target={:?} committed={:?} draft_in_top_k={:?}\n",
                    record.workload_id,
                    record.scheduled_len,
                    first_trace.draft_tokens,
                    first_trace.target_tokens,
                    first_trace.committed_tokens,
                    first_trace.draft_in_top_k
                ));
            }
        }
        out.push('\n');
    } else {
        out.push_str("No verifier trace records were emitted.\n\n");
    }
    out.push_str("\n## Hidden tap parity\n\n");
    if summary.native_tap_snapshots.is_empty() {
        out.push_str("Not measured. Required target tap ids are `[5, 17, 29, 41, 46]`.\n\n");
    } else {
        out.push_str("Native tap snapshots were emitted for reference comparison; DeepSpec/PyTorch numeric parity is still not measured.\n\n");
        if let Some(path) = &summary.native_tap_snapshot_manifest_path {
            out.push_str(&format!("- Manifest: `{}`\n", escape_md(path)));
        }
        for snapshot in &summary.native_tap_snapshots {
            out.push_str(&format!(
                "- {}: path=`{}` layers={:?} shapes={:?} tap_bytes={} prefill_greedy={}\n",
                snapshot.workload_id,
                escape_md(&snapshot.snapshot_path),
                snapshot.tap_layer_ids,
                snapshot.tap_shapes,
                snapshot.tap_bytes,
                snapshot.prefill_greedy_token
            ));
        }
        out.push('\n');
    }
    out.push_str("## MLX parity\n\n");
    out.push_str("Not measured. See `tools/dspark/convert_to_mlx.py` and `tools/dspark/compare_mlx_parity.py`.\n\n");
    out.push_str("## Blockers\n\n");
    if summary.blockers.is_empty() {
        out.push_str("No blockers recorded.\n");
    } else {
        for blocker in &summary.blockers {
            out.push_str(&format!("- {}\n", escape_md(blocker)));
        }
    }
    out
}

fn render_blockers(summary: &Summary) -> String {
    if summary.blockers.is_empty() {
        return "# XR60 blockers\n\nNo blockers recorded.\n".to_owned();
    }
    let mut out = String::from("# XR60 blockers\n\n");
    for blocker in &summary.blockers {
        out.push_str(&format!(
            "## Blocker: {}\n\n- Time: {}\n- Git SHA: `{}`\n- Phase/Gate: G0-G4 startup\n- Command: `{}`\n- Expected: DSpark fixed-prefix benchmark can run through native Helios path\n- Observed: {}\n- Next input needed: provide missing artifact/dependency or implement the named native integration slice\n\n",
            escape_md(blocker),
            summary.generated_at_unix_seconds,
            escape_md(&summary.build_provenance.git_sha),
            escape_md(&summary.command),
            escape_md(blocker)
        ));
    }
    out
}

fn write_jsonl(path: &Path, records: &[Record]) -> Result<(), Box<dyn std::error::Error>> {
    let mut body = String::new();
    for record in records {
        body.push_str(&serde_json::to_string(record)?);
        body.push('\n');
    }
    fs::write(path, body)?;
    Ok(())
}

fn required_value<I>(args: &mut I, flag: &str) -> Result<String, Box<dyn std::error::Error>>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn parse_usize_list(value: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    value
        .split(',')
        .filter(|item| !item.trim().is_empty())
        .map(|item| {
            item.trim()
                .parse()
                .map_err(|_| format!("invalid integer in --block-sizes: {item}").into())
        })
        .collect()
}

fn parse_string_list(value: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    Ok(value
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(ToOwned::to_owned)
        .collect())
}

fn run_id() -> String {
    format!("xr60-{}", unix_now())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn escape_md(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn tps(tokens: usize, total_ms: f64) -> f64 {
    if total_ms <= 0.0 {
        0.0
    } else {
        tokens as f64 / (total_ms / 1000.0)
    }
}

fn fmt_f64(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "n/a".to_owned())
}

fn fmt_u64(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "n/a".to_owned())
}

fn native_tap_snapshot_manifest_path(args: &Args) -> Option<String> {
    args.native_tap_snapshot_dir.as_ref().map(|path| {
        path.join("native_tap_snapshot_manifest.json")
            .display()
            .to_string()
    })
}

#[allow(dead_code)]
fn checksum_tokens(tokens: &[i32]) -> String {
    sha256_hex(
        tokens
            .iter()
            .map(i32::to_string)
            .collect::<Vec<_>>()
            .join(",")
            .as_bytes(),
    )
}
