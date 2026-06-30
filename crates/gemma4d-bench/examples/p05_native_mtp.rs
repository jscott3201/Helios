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
    Drafter, KvCache, KvPolicy, LoadConfig, StepResult, Target, decode_one, draft_block, prefill,
    verify_tokens,
};
use serde::Serialize;

const DEFAULT_MODEL: &str = "artifacts/models/gemma-4-12B-it-4bit";
const DEFAULT_ASSISTANT_MODEL: &str = "artifacts/models/gemma-4-12B-it-qat-assistant-4bit";
const DEFAULT_OUT_DIR: &str = "benchmarks/out/P05-native-mtp";
const MODE: &str = "native_target_and_native_mtp_ffi";
const MIN_ACCEPTANCE_RATE: f64 = 0.35;
const MEMORY_CLIFF_GB: f64 = 14.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    fs::create_dir_all(&args.out_dir)?;

    let run_id = run_id();
    let records_path = args.out_dir.join("records.jsonl");
    let summary_path = args.out_dir.join("summary.json");
    let report_path = args.out_dir.join("report.md");
    let blockers_path = args.out_dir.join("blockers.md");
    let environment = capture_environment();
    let probes = probes(args.max_new_tokens);
    let mut records = Vec::new();
    let mut blockers = startup_blockers(&args);

    if blockers.is_empty() {
        for probe in &probes {
            let baseline = run_baseline(&args, probe)?;
            for block_size in &args.block_sizes {
                records.push(run_record(
                    &args,
                    &run_id,
                    probe,
                    *block_size,
                    baseline.clone(),
                )?);
            }
        }
    }

    blockers.extend(blockers_for_records(&records));
    let status = if blockers.is_empty() {
        "passed"
    } else {
        "failed"
    };
    let default_recommendation = default_recommendation(&records);

    let summary = P05Summary {
        schema_version: 1,
        goal: "P05-native-mtp",
        status,
        run_id,
        timestamp_unix: unix_now(),
        mode: MODE,
        model_path: args.model_path.display().to_string(),
        assistant_model_path: args.assistant_model_path.display().to_string(),
        records_path: records_path.display().to_string(),
        summary_path: summary_path.display().to_string(),
        report_path: report_path.display().to_string(),
        blockers_path: blockers_path.display().to_string(),
        environment,
        relevant_environment: capture_relevant_environment(),
        max_context_tokens: args.max_context_tokens,
        block_sizes: args.block_sizes.clone(),
        min_acceptance_rate: MIN_ACCEPTANCE_RATE,
        memory_cliff_gb: MEMORY_CLIFF_GB,
        default_recommendation,
        claims: claim_inventory(&records),
        records,
        blockers,
        measurement_notes: vec![
            "This harness requires GEMMA4D_REQUIRE_MLX=1 at build time and GEMMA4D_USE_NATIVE_GRAPH=1 at runtime.",
            "Baseline and MTP runs both use gemma4d-ffi safe wrappers over the native C ABI.",
            "MTP runs load the real Gemma 4 assistant artifact through Drafter::load and draft through gemma4_mtp_draft_block.",
            "Verification and rollback use gemma4_verify_tokens; emitted tokens are reconstructed from committed-token metadata returned by the verifier.",
            "If acceptance or memory gates fail, the harness auto-disables MTP for the rest of that run and fills the tail with native non-MTP decode_one.",
            "Peak memory is the maximum peak_memory_gb surfaced by native target prefill/decode/verify StepResult values.",
        ],
    };

    write_jsonl(&records_path, &summary.records)?;
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)?;
    fs::write(&report_path, render_report(&summary))?;
    fs::write(&blockers_path, render_blockers(&summary))?;

    println!("P05 native MTP: {}", summary.status);
    println!("records: {}", records_path.display());
    println!("summary: {}", summary_path.display());
    println!("report: {}", report_path.display());
    println!("blockers: {}", blockers_path.display());

    if summary.status == "failed" {
        Err("P05 native MTP checks failed".into())
    } else {
        Ok(())
    }
}

#[derive(Debug)]
struct Args {
    out_dir: PathBuf,
    model_path: PathBuf,
    assistant_model_path: PathBuf,
    max_context_tokens: usize,
    max_new_tokens: usize,
    block_sizes: Vec<usize>,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut out_dir = PathBuf::from(DEFAULT_OUT_DIR);
        let mut model_path = PathBuf::from(DEFAULT_MODEL);
        let mut assistant_model_path = PathBuf::from(DEFAULT_ASSISTANT_MODEL);
        let mut max_context_tokens = 8192;
        let mut max_new_tokens = 8;
        let mut block_sizes = vec![1, 2];

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
                "--assistant-model-path" => {
                    assistant_model_path = args
                        .next()
                        .map(PathBuf::from)
                        .ok_or("--assistant-model-path requires a path")?;
                }
                "--max-context-tokens" => {
                    let value = args.next().ok_or("--max-context-tokens requires a value")?;
                    max_context_tokens = parse_positive_usize(&value, "--max-context-tokens")?;
                }
                "--max-new-tokens" => {
                    let value = args.next().ok_or("--max-new-tokens requires a value")?;
                    max_new_tokens = parse_positive_usize(&value, "--max-new-tokens")?;
                }
                "--block-sizes" => {
                    let value = args.next().ok_or("--block-sizes requires a comma list")?;
                    block_sizes = parse_contexts(&value)?;
                }
                "-h" | "--help" => {
                    println!(
                        "usage: GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example p05_native_mtp -- [--out-dir PATH] [--model-path PATH] [--assistant-model-path PATH] [--max-new-tokens N] [--block-sizes 1,2]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown option '{other}'").into()),
            }
        }

        if block_sizes.is_empty() {
            return Err("--block-sizes must include at least one value".into());
        }
        block_sizes.sort_unstable();
        block_sizes.dedup();
        if block_sizes.iter().any(|size| *size > 2) {
            return Err("P05 native MTP currently supports --block-sizes up to 2".into());
        }

        Ok(Self {
            out_dir,
            model_path,
            assistant_model_path,
            max_context_tokens,
            max_new_tokens,
            block_sizes,
        })
    }
}

#[derive(Debug, Clone)]
struct Probe {
    id: &'static str,
    description: &'static str,
    prompt_tokens: Vec<i32>,
    max_new_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
struct P05Summary {
    schema_version: u32,
    goal: &'static str,
    status: &'static str,
    run_id: String,
    timestamp_unix: u64,
    mode: &'static str,
    model_path: String,
    assistant_model_path: String,
    records_path: String,
    summary_path: String,
    report_path: String,
    blockers_path: String,
    environment: Environment,
    relevant_environment: BTreeMap<String, Option<String>>,
    max_context_tokens: usize,
    block_sizes: Vec<usize>,
    min_acceptance_rate: f64,
    memory_cliff_gb: f64,
    default_recommendation: String,
    claims: ClaimInventory,
    records: Vec<P05Record>,
    blockers: Vec<String>,
    measurement_notes: Vec<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
struct P05Record {
    schema_version: u32,
    goal: &'static str,
    run_id: String,
    timestamp_unix: u64,
    probe_id: String,
    description: String,
    prompt_tokens: Vec<i32>,
    max_new_tokens: usize,
    block_size: usize,
    mode: &'static str,
    baseline: NativeGreedyRun,
    mtp: NativeMtpRun,
    comparison: Comparison,
    gate: GateOutcome,
    blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct NativeGreedyRun {
    generated_tokens: Vec<i32>,
    model_load_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    total_ms: f64,
    decode_tps: f64,
    peak_memory_gb: f32,
    active_kv_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
struct NativeMtpRun {
    generated_tokens: Vec<i32>,
    model_load_ms: f64,
    drafter_load_ms: f64,
    prefill_ms: f64,
    draft_ms: f64,
    verify_ms: f64,
    fallback_decode_ms: f64,
    total_ms: f64,
    decode_tps: f64,
    draft_block_size: usize,
    attempted_draft_tokens: u64,
    accepted_draft_tokens: u64,
    acceptance_rate: f64,
    accepted_tokens_per_verify: f64,
    target_verify_passes: u64,
    rollback_count: u64,
    auto_disabled: bool,
    auto_disable_reason: Option<String>,
    peak_memory_gb: f32,
    active_kv_bytes: u64,
    events: Vec<MtpEvent>,
}

#[derive(Debug, Clone, Serialize)]
struct MtpEvent {
    pass_index: u64,
    draft_tokens: Vec<i32>,
    committed_tokens: Vec<i32>,
    accepted_draft_count: u32,
    rejected: bool,
    sequence_len: u64,
    verify_ms: f64,
    peak_memory_gb: f32,
}

#[derive(Debug, Clone, Serialize)]
struct Comparison {
    byte_identical: bool,
    first_mismatch: Option<TokenMismatch>,
}

#[derive(Debug, Clone, Serialize)]
struct TokenMismatch {
    index: usize,
    baseline_token: Option<i32>,
    mtp_token: Option<i32>,
}

#[derive(Debug, Clone, Serialize)]
struct GateOutcome {
    correctness_passed: bool,
    acceptance_gate_passed: bool,
    memory_gate_passed: bool,
    default_enabled: bool,
    recommendation: String,
}

#[derive(Debug, Clone, Serialize)]
struct ClaimInventory {
    exactness: Vec<String>,
    acceptance: Vec<String>,
    speed: Vec<String>,
    memory: Vec<String>,
    auto_disable: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct Environment {
    machine: String,
    macos: String,
    rustc: String,
    cargo: String,
    mlx_version: String,
    git_commit: String,
    git_status_short: String,
    hw_memsize_bytes: Option<u64>,
}

fn probes(max_new_tokens: usize) -> Vec<Probe> {
    vec![
        Probe {
            id: "hello_smoke",
            description: "M04 tokenizer-controlled Hello prompt.",
            prompt_tokens: vec![9259],
            max_new_tokens,
        },
        Probe {
            id: "hello_reference_prefix",
            description: "Hello plus two reference target tokens.",
            prompt_tokens: vec![9259, 236772, 236772],
            max_new_tokens,
        },
    ]
}

fn startup_blockers(args: &Args) -> Vec<String> {
    let mut blockers = Vec::new();
    if !args.model_path.exists() {
        blockers.push(format!(
            "target model path does not exist: {}",
            args.model_path.display()
        ));
    }
    if !args.assistant_model_path.exists() {
        blockers.push(format!(
            "assistant model path does not exist: {}",
            args.assistant_model_path.display()
        ));
    }
    if env::var_os("GEMMA4D_USE_NATIVE_GRAPH").is_none() {
        blockers.push("GEMMA4D_USE_NATIVE_GRAPH=1 is required for P05 native MTP".to_owned());
    }
    if env::var_os("GEMMA4D_REQUIRE_MLX").is_none() {
        blockers
            .push("GEMMA4D_REQUIRE_MLX=1 is required so gemma4d-ffi builds with MLX".to_owned());
    }
    blockers
}

fn run_record(
    args: &Args,
    run_id: &str,
    probe: &Probe,
    block_size: usize,
    baseline: NativeGreedyRun,
) -> Result<P05Record, Box<dyn std::error::Error>> {
    let mtp = run_mtp(args, probe, block_size)?;
    let comparison = compare_tokens(&baseline.generated_tokens, &mtp.generated_tokens);
    let gate = gate_outcome(&comparison, &mtp);
    let mut blockers = Vec::new();
    if !comparison.byte_identical {
        blockers.push(format!(
            "{} block_size={} native MTP output differs from non-MTP native baseline",
            probe.id, block_size
        ));
    }

    Ok(P05Record {
        schema_version: 1,
        goal: "P05-native-mtp",
        run_id: run_id.to_owned(),
        timestamp_unix: unix_now(),
        probe_id: probe.id.to_owned(),
        description: probe.description.to_owned(),
        prompt_tokens: probe.prompt_tokens.clone(),
        max_new_tokens: probe.max_new_tokens,
        block_size,
        mode: MODE,
        baseline,
        mtp,
        comparison,
        gate,
        blockers,
    })
}

fn run_baseline(args: &Args, probe: &Probe) -> Result<NativeGreedyRun, Box<dyn std::error::Error>> {
    let started = Instant::now();
    let target_started = Instant::now();
    let target = Target::load(&target_config(args))?;
    let model_load = target_started.elapsed();
    let mut cache = KvCache::create(&KvPolicy::default())?;

    let prefill_started = Instant::now();
    let first = prefill(&target, &mut cache, &probe.prompt_tokens)?;
    let prefill_duration = prefill_started.elapsed();

    let mut generated = vec![first.greedy_token];
    let mut decode_duration = Duration::ZERO;
    let mut peak_memory_gb = first.peak_memory_gb;
    let mut active_kv_bytes = first.active_kv_bytes;

    while generated.len() < probe.max_new_tokens {
        let token = *generated.last().expect("generated has first token");
        let token_started = Instant::now();
        let step = decode_one(&target, &mut cache, token)?;
        decode_duration += token_started.elapsed();
        peak_memory_gb = peak_memory_gb.max(step.peak_memory_gb);
        active_kv_bytes = active_kv_bytes.max(step.active_kv_bytes);
        generated.push(step.greedy_token);
    }

    let total = started.elapsed();
    Ok(NativeGreedyRun {
        generated_tokens: generated,
        model_load_ms: duration_ms(model_load),
        prefill_ms: duration_ms(prefill_duration),
        decode_ms: duration_ms(decode_duration),
        total_ms: duration_ms(total),
        decode_tps: tps(probe.max_new_tokens, total),
        peak_memory_gb,
        active_kv_bytes,
    })
}

fn run_mtp(
    args: &Args,
    probe: &Probe,
    block_size: usize,
) -> Result<NativeMtpRun, Box<dyn std::error::Error>> {
    let started = Instant::now();
    let target_started = Instant::now();
    let target = Target::load(&target_config(args))?;
    let model_load = target_started.elapsed();
    let drafter_started = Instant::now();
    let drafter = Drafter::load(&assistant_config(args), &target)?;
    let drafter_load = drafter_started.elapsed();
    let mut cache = KvCache::create(&KvPolicy::default())?;

    let prefill_started = Instant::now();
    let first = prefill(&target, &mut cache, &probe.prompt_tokens)?;
    let prefill_duration = prefill_started.elapsed();

    let mut generated = Vec::with_capacity(probe.max_new_tokens);
    let mut draft_duration = Duration::ZERO;
    let mut verify_duration = Duration::ZERO;
    let mut fallback_decode_duration = Duration::ZERO;
    let mut peak_memory_gb = first.peak_memory_gb;
    let mut active_kv_bytes = first.active_kv_bytes;
    let mut attempted_draft_tokens = 0_u64;
    let mut accepted_draft_tokens = 0_u64;
    let mut target_verify_passes = 0_u64;
    let mut rollback_count = 0_u64;
    let mut auto_disabled = false;
    let mut auto_disable_reason = None;
    let mut pending_greedy = Some(first.greedy_token);
    let mut events = Vec::new();

    while generated.len() < probe.max_new_tokens {
        if auto_disabled {
            if let Some(token) = pending_greedy.take() {
                generated.push(token);
                continue;
            }
            let token = *generated
                .last()
                .ok_or("auto-disabled MTP has no committed token")?;
            let token_started = Instant::now();
            let step = decode_one(&target, &mut cache, token)?;
            fallback_decode_duration += token_started.elapsed();
            peak_memory_gb = peak_memory_gb.max(step.peak_memory_gb);
            active_kv_bytes = active_kv_bytes.max(step.active_kv_bytes);
            generated.push(step.greedy_token);
            continue;
        }

        let remaining = probe.max_new_tokens - generated.len();
        let current_block_size = block_size.min(remaining);
        let draft_started = Instant::now();
        let draft = draft_block(
            &drafter,
            &mut cache,
            NonZeroU32::new(current_block_size as u32).expect("block size is non-zero"),
        )?;
        draft_duration += draft_started.elapsed();
        if draft.is_empty() {
            auto_disabled = true;
            auto_disable_reason = Some("native drafter returned no tokens".to_owned());
            continue;
        }
        pending_greedy = None;

        attempted_draft_tokens += draft.len() as u64;
        target_verify_passes += 1;
        let verify_started = Instant::now();
        let step = verify_tokens(&target, &mut cache, &draft)?;
        let verify_elapsed = verify_started.elapsed();
        verify_duration += verify_elapsed;
        peak_memory_gb = peak_memory_gb.max(step.peak_memory_gb);
        active_kv_bytes = active_kv_bytes.max(step.active_kv_bytes);

        let committed = committed_tokens(&step);
        if committed.is_empty() {
            return Err("gemma4_verify_tokens committed no tokens".into());
        }
        let accepted = u64::from(step.accepted_draft_count);
        accepted_draft_tokens += accepted;
        let rejected =
            usize::try_from(step.accepted_draft_count).unwrap_or(usize::MAX) < draft.len();
        if rejected {
            rollback_count += 1;
        }
        for token in &committed {
            if generated.len() < probe.max_new_tokens {
                generated.push(*token);
            }
        }

        events.push(MtpEvent {
            pass_index: target_verify_passes,
            draft_tokens: draft,
            committed_tokens: committed,
            accepted_draft_count: step.accepted_draft_count,
            rejected,
            sequence_len: step.sequence_len,
            verify_ms: duration_ms(verify_elapsed),
            peak_memory_gb: step.peak_memory_gb,
        });

        let acceptance_rate = acceptance_rate(accepted_draft_tokens, attempted_draft_tokens);
        if rejected && acceptance_rate < MIN_ACCEPTANCE_RATE {
            auto_disabled = true;
            auto_disable_reason = Some(format!(
                "acceptance rate {:.3} fell below threshold {:.3}",
                acceptance_rate, MIN_ACCEPTANCE_RATE
            ));
            pending_greedy = Some(step.greedy_token);
        } else if peak_memory_gb as f64 >= MEMORY_CLIFF_GB {
            auto_disabled = true;
            auto_disable_reason = Some(format!(
                "peak memory {:.3} GB crossed {:.1} GB threshold",
                peak_memory_gb, MEMORY_CLIFF_GB
            ));
            pending_greedy = Some(step.greedy_token);
        }
    }

    let total = started.elapsed();
    let acceptance_rate = acceptance_rate(accepted_draft_tokens, attempted_draft_tokens);
    Ok(NativeMtpRun {
        generated_tokens: generated,
        model_load_ms: duration_ms(model_load),
        drafter_load_ms: duration_ms(drafter_load),
        prefill_ms: duration_ms(prefill_duration),
        draft_ms: duration_ms(draft_duration),
        verify_ms: duration_ms(verify_duration),
        fallback_decode_ms: duration_ms(fallback_decode_duration),
        total_ms: duration_ms(total),
        decode_tps: tps(probe.max_new_tokens, total),
        draft_block_size: block_size,
        attempted_draft_tokens,
        accepted_draft_tokens,
        acceptance_rate,
        accepted_tokens_per_verify: if target_verify_passes == 0 {
            0.0
        } else {
            accepted_draft_tokens as f64 / target_verify_passes as f64
        },
        target_verify_passes,
        rollback_count,
        auto_disabled,
        auto_disable_reason,
        peak_memory_gb,
        active_kv_bytes,
        events,
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

fn assistant_config(args: &Args) -> LoadConfig {
    LoadConfig {
        model_path: args.assistant_model_path.display().to_string(),
        model_id: Some("mlx-community/gemma-4-12B-it-qat-assistant-4bit".to_owned()),
        model_revision: None,
        expected_architecture: Some("gemma4_unified_assistant".to_owned()),
        max_context_tokens: NonZeroU32::new(args.max_context_tokens as u32)
            .expect("max context is non-zero"),
        allow_unsupported_config: false,
    }
}

fn committed_tokens(step: &StepResult) -> Vec<i32> {
    step.committed_tokens().to_vec()
}

fn compare_tokens(baseline: &[i32], mtp: &[i32]) -> Comparison {
    let max_len = baseline.len().max(mtp.len());
    let first_mismatch = (0..max_len).find_map(|index| {
        let baseline_token = baseline.get(index).copied();
        let mtp_token = mtp.get(index).copied();
        (baseline_token != mtp_token).then_some(TokenMismatch {
            index,
            baseline_token,
            mtp_token,
        })
    });
    Comparison {
        byte_identical: first_mismatch.is_none() && baseline.len() == mtp.len(),
        first_mismatch,
    }
}

fn gate_outcome(comparison: &Comparison, mtp: &NativeMtpRun) -> GateOutcome {
    let correctness_passed = comparison.byte_identical;
    let acceptance_gate_passed = mtp.acceptance_rate >= MIN_ACCEPTANCE_RATE;
    let memory_gate_passed = mtp.peak_memory_gb as f64 <= MEMORY_CLIFF_GB;
    let default_enabled = false;
    let recommendation = if !correctness_passed {
        "keep_disabled_correctness_failed"
    } else if !acceptance_gate_passed || !memory_gate_passed {
        "keep_disabled_auto_disable_gate"
    } else {
        "eligible_for_manual_opt_in_only"
    }
    .to_owned();
    GateOutcome {
        correctness_passed,
        acceptance_gate_passed,
        memory_gate_passed,
        default_enabled,
        recommendation,
    }
}

fn default_recommendation(records: &[P05Record]) -> String {
    if records.is_empty() {
        return "keep_disabled_no_evidence".to_owned();
    }
    if records.iter().all(|record| {
        record.gate.correctness_passed
            && record.gate.acceptance_gate_passed
            && record.gate.memory_gate_passed
    }) {
        "eligible_for_manual_opt_in_only".to_owned()
    } else {
        "keep_disabled_by_default".to_owned()
    }
}

fn claim_inventory(records: &[P05Record]) -> ClaimInventory {
    let mut exactness = Vec::new();
    let mut acceptance = Vec::new();
    let mut speed = Vec::new();
    let mut memory = Vec::new();
    let mut auto_disable = Vec::new();

    for record in records {
        if record.comparison.byte_identical {
            exactness.push(format!(
                "{} block_size={}: native MTP output matched non-MTP native baseline",
                record.probe_id, record.block_size
            ));
        }
        acceptance.push(format!(
            "{} block_size={}: attempted={} accepted={} rate={:.3} accepted/verify={:.3} rollbacks={}",
            record.probe_id,
            record.block_size,
            record.mtp.attempted_draft_tokens,
            record.mtp.accepted_draft_tokens,
            record.mtp.acceptance_rate,
            record.mtp.accepted_tokens_per_verify,
            record.mtp.rollback_count,
        ));
        speed.push(format!(
            "{} block_size={}: baseline {:.3} tok/s vs MTP {:.3} tok/s",
            record.probe_id, record.block_size, record.baseline.decode_tps, record.mtp.decode_tps
        ));
        memory.push(format!(
            "{} block_size={}: MTP peak {:.3} GB active KV {:.3} MiB",
            record.probe_id,
            record.block_size,
            record.mtp.peak_memory_gb,
            record.mtp.active_kv_bytes as f64 / 1024.0 / 1024.0
        ));
        if record.mtp.auto_disabled {
            auto_disable.push(format!(
                "{} block_size={}: {}",
                record.probe_id,
                record.block_size,
                record
                    .mtp
                    .auto_disable_reason
                    .as_deref()
                    .unwrap_or("auto-disabled")
            ));
        }
    }

    ClaimInventory {
        exactness,
        acceptance,
        speed,
        memory,
        auto_disable,
    }
}

fn blockers_for_records(records: &[P05Record]) -> Vec<String> {
    records
        .iter()
        .flat_map(|record| record.blockers.iter().cloned())
        .collect()
}

fn write_jsonl(path: &Path, records: &[P05Record]) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = fs::File::create(path)?;
    for record in records {
        writeln!(file, "{}", serde_json::to_string(record)?)?;
    }
    Ok(())
}

fn render_report(summary: &P05Summary) -> String {
    let mut out = String::new();
    out.push_str("# P05 Native MTP\n\n");
    out.push_str("## Status\n\n");
    out.push_str(&format!(
        "- Status: `{}`\n- Mode: `{}`\n- Records: `{}`\n- Summary: `{}`\n- Blockers: `{}`\n- Default recommendation: `{}`\n- Min acceptance rate: `{:.3}`\n- Memory cliff: `{:.1} GB`\n\n",
        summary.status,
        summary.mode,
        summary.records_path,
        summary.summary_path,
        summary.blockers_path,
        summary.default_recommendation,
        summary.min_acceptance_rate,
        summary.memory_cliff_gb,
    ));

    out.push_str("## Claim Inventory\n\n");
    render_claim_list(&mut out, "Exactness", &summary.claims.exactness);
    render_claim_list(&mut out, "Acceptance", &summary.claims.acceptance);
    render_claim_list(&mut out, "Speed", &summary.claims.speed);
    render_claim_list(&mut out, "Memory", &summary.claims.memory);
    render_claim_list(&mut out, "Auto Disable", &summary.claims.auto_disable);

    out.push_str("## Results\n\n");
    out.push_str("| Probe | Block | Exact | Attempted | Accepted | Rate | Accepted/Verify | Verify Passes | Rollbacks | Auto Disabled | Baseline tok/s | MTP tok/s | MTP Peak GB | Recommendation |\n");
    out.push_str("|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|\n");
    for record in &summary.records {
        out.push_str(&format!(
            "| `{}` | {} | `{}` | {} | {} | {:.3} | {:.3} | {} | {} | `{}` | {:.3} | {:.3} | {:.3} | `{}` |\n",
            record.probe_id,
            record.block_size,
            record.comparison.byte_identical,
            record.mtp.attempted_draft_tokens,
            record.mtp.accepted_draft_tokens,
            record.mtp.acceptance_rate,
            record.mtp.accepted_tokens_per_verify,
            record.mtp.target_verify_passes,
            record.mtp.rollback_count,
            record.mtp.auto_disabled,
            record.baseline.decode_tps,
            record.mtp.decode_tps,
            record.mtp.peak_memory_gb,
            record.gate.recommendation,
        ));
    }

    out.push_str("\n## Token Detail\n\n");
    out.push_str("| Probe | Block | Baseline Tokens | MTP Tokens | First Mismatch |\n");
    out.push_str("|---|---:|---|---|---|\n");
    for record in &summary.records {
        out.push_str(&format!(
            "| `{}` | {} | `{}` | `{}` | `{}` |\n",
            record.probe_id,
            record.block_size,
            tokens_short(&record.baseline.generated_tokens),
            tokens_short(&record.mtp.generated_tokens),
            record
                .comparison
                .first_mismatch
                .as_ref()
                .map(|mismatch| format!(
                    "index={} baseline={:?} mtp={:?}",
                    mismatch.index, mismatch.baseline_token, mismatch.mtp_token
                ))
                .unwrap_or_else(|| "none".to_owned())
        ));
    }

    out.push_str("\n## Environment\n\n");
    out.push_str("| Item | Value |\n|---|---|\n");
    out.push_str(&format!(
        "| Machine | `{}` |\n",
        escape_md(&summary.environment.machine)
    ));
    out.push_str(&format!(
        "| macOS | `{}` |\n",
        escape_md(&summary.environment.macos)
    ));
    out.push_str(&format!(
        "| Rust | `{}` |\n",
        escape_md(&summary.environment.rustc)
    ));
    out.push_str(&format!(
        "| Cargo | `{}` |\n",
        escape_md(&summary.environment.cargo)
    ));
    out.push_str(&format!(
        "| MLX | `{}` |\n",
        escape_md(&summary.environment.mlx_version)
    ));
    out.push_str(&format!(
        "| Git commit | `{}` |\n",
        escape_md(&summary.environment.git_commit)
    ));
    out.push_str(&format!(
        "| Git status | `{}` |\n",
        escape_md(&summary.environment.git_status_short)
    ));

    out.push_str("\n## Commands\n\n```text\n");
    out.push_str(&format!(
        "GEMMA4D_REQUIRE_MLX=1 GEMMA4D_USE_NATIVE_GRAPH=1 cargo run -p gemma4d-bench --example p05_native_mtp -- --out-dir {} --model-path {} --assistant-model-path {}\n",
        summary.records_path
            .trim_end_matches("/records.jsonl"),
        summary.model_path,
        summary.assistant_model_path
    ));
    out.push_str("```\n\n");

    out.push_str("## Notes\n\n");
    for note in &summary.measurement_notes {
        out.push_str(&format!("- {note}\n"));
    }
    if !summary.blockers.is_empty() {
        out.push_str("\n## Blockers\n\n");
        for blocker in &summary.blockers {
            out.push_str(&format!("- {blocker}\n"));
        }
    }
    out
}

fn render_claim_list(out: &mut String, title: &str, claims: &[String]) {
    out.push_str(&format!("### {title}\n\n"));
    if claims.is_empty() {
        out.push_str("- None recorded in this run.\n\n");
    } else {
        for claim in claims {
            out.push_str(&format!("- {claim}\n"));
        }
        out.push('\n');
    }
}

fn render_blockers(summary: &P05Summary) -> String {
    if summary.blockers.is_empty() {
        return "No blockers recorded.\n".to_owned();
    }
    let mut out = String::new();
    out.push_str("# P05 Blockers\n\n");
    for blocker in &summary.blockers {
        out.push_str(&format!("- {blocker}\n"));
    }
    out
}

fn acceptance_rate(accepted: u64, attempted: u64) -> f64 {
    if attempted == 0 {
        0.0
    } else {
        accepted as f64 / attempted as f64
    }
}

fn parse_contexts(value: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    value
        .split(',')
        .map(|part| parse_positive_usize(part.trim(), "--block-sizes"))
        .collect()
}

fn parse_positive_usize(value: &str, flag: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("{flag} must be a positive integer: {error}"))?;
    if parsed == 0 {
        return Err(format!("{flag} must be greater than zero").into());
    }
    Ok(parsed)
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn tps(tokens: usize, duration: Duration) -> f64 {
    if duration.is_zero() {
        0.0
    } else {
        tokens as f64 / duration.as_secs_f64()
    }
}

fn capture_environment() -> Environment {
    Environment {
        machine: command_stdout("uname", &["-a"]).unwrap_or_else(|| "unknown".to_owned()),
        macos: command_stdout("sw_vers", &[]).unwrap_or_else(|| "unknown".to_owned()),
        rustc: command_stdout("rustc", &["-Vv"]).unwrap_or_else(|| "unknown".to_owned()),
        cargo: command_stdout("cargo", &["-V"]).unwrap_or_else(|| "unknown".to_owned()),
        mlx_version: mlx_version(),
        git_commit: command_stdout("git", &["rev-parse", "HEAD"])
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
        "GEMMA4D_ASSISTANT_MODEL_PATH",
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

fn mlx_version() -> String {
    let python = env::var("GEMMA4D_MLX_LM_PYTHON")
        .unwrap_or_else(|_| "/opt/homebrew/opt/mlx-lm/libexec/bin/python".to_owned());
    command_stdout(
        &python,
        &[
            "-c",
            "import mlx.core as mx; import mlx_lm; print(f'mlx={mx.__version__} mlx_lm={getattr(mlx_lm, \"__version__\", \"unknown\")}')",
        ],
    )
    .or_else(|| {
        command_stdout(
            "python3",
            &[
                "-c",
                "import mlx.core as mx; import mlx_lm; print(f'mlx={mx.__version__} mlx_lm={getattr(mlx_lm, \"__version__\", \"unknown\")}')",
            ],
        )
    })
    .unwrap_or_else(|| "unknown".to_owned())
}

fn command_stdout(command: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(command).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn tokens_short(tokens: &[i32]) -> String {
    const LIMIT: usize = 12;
    if tokens.len() <= LIMIT {
        return format!("{tokens:?}");
    }
    let head = tokens
        .iter()
        .take(LIMIT)
        .map(i32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!("[{head}, ...; len={}]", tokens.len())
}

fn escape_md(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', "<br>")
}

fn run_id() -> String {
    format!("p05-{}", unix_now())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is after Unix epoch")
        .as_secs()
}
