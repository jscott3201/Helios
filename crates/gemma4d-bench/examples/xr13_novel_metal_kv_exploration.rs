#[cfg(not(feature = "xr13-prototypes"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err("XR13 prototype runner requires --features xr13-prototypes".into())
}

#[cfg(feature = "xr13-prototypes")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    xr13::run()
}

#[cfg(feature = "xr13-prototypes")]
mod xr13 {
    use std::{
        cmp::Ordering,
        collections::{BTreeMap, HashMap},
        env,
        fs::{self, File},
        hint::black_box,
        io::{BufRead, BufReader, Write},
        path::{Path, PathBuf},
        process::Command,
        time::{Instant, SystemTime, UNIX_EPOCH},
    };

    use gemma4d_bench::manifest;
    use serde::{Deserialize, Serialize};

    const GOAL: &str = "XR13-novel-metal-kv-exploration";
    const MODE: &str = "feature_gated_l1_compressed_k_score_microbenchmark";
    const FEATURE_FLAG: &str = "xr13-prototypes";
    const DEFAULT_OUT_DIR: &str = "benchmarks/out/XR13-novel-metal-kv-exploration";
    const DEFAULT_MODEL: &str = "artifacts/models/gemma-4-12B-it-4bit";
    const DEFAULT_XR09_SUMMARY: &str =
        "benchmarks/out/XR09-kv-compression-real-quality-ab/summary.json";
    const DEFAULT_XR09_RECORDS: &str =
        "benchmarks/out/XR09-kv-compression-real-quality-ab/records.jsonl";
    const DEFAULT_HEAD_DIM: usize = 64;
    const DEFAULT_PROJECTION_DIM: usize = 8;
    const DEFAULT_TRIALS: usize = 3;

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let args = Args::parse()?;
        fs::create_dir_all(&args.out_dir)?;

        let records_path = args.out_dir.join("records.jsonl");
        let summary_path = args.out_dir.join("summary.json");
        let report_path = args.out_dir.join("report.md");
        let blockers_path = args.out_dir.join("blockers.md");
        let decision_path = args.out_dir.join("decision.md");
        let run_id = run_id();
        let timestamp_unix = unix_now();
        let command = command_display();
        let reproduction_command = reproduction_command(&args);
        let environment = capture_environment();
        let relevant_environment = capture_relevant_environment();
        let model_identity =
            manifest::capture_artifact_identity(&args.model_path, "GEMMA4D_MODEL_REVISION");

        let xr09_summary = load_xr09_summary(&args.xr09_summary_path)?;
        let xr09_records = load_xr09_references(&args.xr09_records_path, &xr09_summary)?;
        let mut selected_cases = xr09_summary.selected_cases.clone();
        if let Some(max_cases) = args.max_cases {
            selected_cases.truncate(max_cases);
        }
        if selected_cases.is_empty() {
            return Err("XR09 selected_cases is empty; cannot build XR13 corpus".into());
        }

        let mut records = Vec::new();
        for case in &selected_cases {
            let xr09_reference = xr09_records
                .get(&case.case_id)
                .cloned()
                .unwrap_or_else(|| Xr09Reference::missing(&xr09_summary, case));
            for trial_index in 0..args.trials {
                let seed = derived_seed(case, trial_index, args.head_dim, args.projection_dim);
                records.push(run_case(
                    case,
                    xr09_reference.clone(),
                    &args,
                    &run_id,
                    timestamp_unix,
                    seed,
                    trial_index,
                    &environment,
                    &model_identity,
                )?);
            }
        }

        let aggregates = aggregate_variants(&records);
        let blockers = hard_blockers(&records);
        let failed_hypotheses = failed_hypotheses(&aggregates);
        let candidate_failed = aggregates.values().any(|aggregate| {
            aggregate.role == "candidate" && aggregate.correctness_passes < aggregate.samples
        });
        let decision = if !blockers.is_empty() {
            "blocked_with_evidence"
        } else if candidate_failed {
            "reject_candidate"
        } else {
            "keep_experimental"
        };
        let status = if blockers.is_empty() {
            "completed"
        } else {
            "blocked"
        };

        let generated_files = vec![
            records_path.display().to_string(),
            summary_path.display().to_string(),
            report_path.display().to_string(),
            blockers_path.display().to_string(),
            decision_path.display().to_string(),
        ];
        let selected_case_evidence = selected_cases
            .iter()
            .map(|case| SelectedCaseEvidence {
                case_id: case.case_id.clone(),
                workload_id: case.workload_id.clone(),
                family: case.family.clone(),
                prompt_path: case.prompt_path.clone(),
                prompt_sha256: case.prompt_sha256.clone(),
                source_deterministic_seed: case.source_deterministic_seed,
                target_context_tokens: case.target_context_tokens,
                actual_context_tokens: case.actual_context_tokens,
                context_tokens: case.context_tokens,
                prefix_token_hash: case.prefix_token_hash.clone(),
                derived_trial_seeds: (0..args.trials)
                    .map(|trial_index| {
                        derived_seed(case, trial_index, args.head_dim, args.projection_dim)
                    })
                    .collect(),
            })
            .collect();
        let xr09_references = selected_cases
            .iter()
            .map(|case| {
                xr09_records
                    .get(&case.case_id)
                    .cloned()
                    .unwrap_or_else(|| Xr09Reference::missing(&xr09_summary, case))
            })
            .collect();

        let summary = Summary {
            schema_version: 1,
            goal: GOAL.to_owned(),
            status: status.to_owned(),
            decision: decision.to_owned(),
            run_id,
            timestamp_unix,
            mode: MODE.to_owned(),
            feature_flag: FEATURE_FLAG.to_owned(),
            command,
            reproduction_command,
            git_sha: environment.git_sha.clone(),
            git_status_short: environment.git_status_short.clone(),
            model_path: args.model_path.display().to_string(),
            model_identity,
            xr09_summary_path: args.xr09_summary_path.display().to_string(),
            xr09_records_path: args.xr09_records_path.display().to_string(),
            xr09_run_id: xr09_summary.run_id.clone(),
            xr09_decision: xr09_summary.decision.clone(),
            xr09_git_sha: xr09_summary.git_sha.clone(),
            xr09_policy_decision: xr09_summary.policy_decision.clone(),
            xr09_model_identity: xr09_summary.model_identity.clone(),
            xr09_references,
            out_dir: args.out_dir.display().to_string(),
            trials: args.trials,
            head_dim: args.head_dim,
            projection_dim: args.projection_dim,
            selected_cases: selected_case_evidence,
            generated_files,
            records_count: records.len(),
            environment,
            relevant_environment,
            aggregates,
            blockers,
            failed_hypotheses,
            measurement_notes: measurement_notes(),
        };

        write_jsonl(&records_path, &records)?;
        fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)?;
        fs::write(&report_path, render_report(&summary))?;
        fs::write(&blockers_path, render_blockers(&summary))?;
        fs::write(&decision_path, render_decision(&summary))?;

        println!("XR13 novel Metal/KV exploration: {}", summary.status);
        println!("decision: {}", summary.decision);
        println!("records: {}", records_path.display());
        println!("summary: {}", summary_path.display());
        println!("report: {}", report_path.display());
        println!("blockers: {}", blockers_path.display());
        println!("decision path: {}", decision_path.display());

        Ok(())
    }

    #[derive(Debug, Clone)]
    struct Args {
        out_dir: PathBuf,
        model_path: PathBuf,
        xr09_summary_path: PathBuf,
        xr09_records_path: PathBuf,
        trials: usize,
        head_dim: usize,
        projection_dim: usize,
        max_cases: Option<usize>,
    }

    impl Args {
        fn parse() -> Result<Self, Box<dyn std::error::Error>> {
            let mut out_dir = PathBuf::from(DEFAULT_OUT_DIR);
            let mut model_path = PathBuf::from(DEFAULT_MODEL);
            let mut xr09_summary_path = PathBuf::from(DEFAULT_XR09_SUMMARY);
            let mut xr09_records_path = PathBuf::from(DEFAULT_XR09_RECORDS);
            let mut trials = DEFAULT_TRIALS;
            let mut head_dim = DEFAULT_HEAD_DIM;
            let mut projection_dim = DEFAULT_PROJECTION_DIM;
            let mut max_cases = None;

            let mut args = env::args().skip(1);
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--out-dir" => out_dir = required_path(&mut args, "--out-dir")?,
                    "--model-path" => model_path = required_path(&mut args, "--model-path")?,
                    "--xr09-summary" => {
                        xr09_summary_path = required_path(&mut args, "--xr09-summary")?
                    }
                    "--xr09-records" => {
                        xr09_records_path = required_path(&mut args, "--xr09-records")?
                    }
                    "--trials" => {
                        trials =
                            parse_positive_usize(&required(&mut args, "--trials")?, "--trials")?
                    }
                    "--head-dim" => {
                        head_dim =
                            parse_positive_usize(&required(&mut args, "--head-dim")?, "--head-dim")?
                    }
                    "--projection-dim" => {
                        projection_dim = parse_positive_usize(
                            &required(&mut args, "--projection-dim")?,
                            "--projection-dim",
                        )?
                    }
                    "--max-cases" => {
                        max_cases = Some(parse_positive_usize(
                            &required(&mut args, "--max-cases")?,
                            "--max-cases",
                        )?)
                    }
                    "-h" | "--help" => {
                        println!(
                            "usage: cargo run -p gemma4d-bench --features xr13-prototypes --example xr13_novel_metal_kv_exploration -- [--out-dir PATH] [--model-path PATH] [--xr09-summary PATH] [--xr09-records PATH] [--trials N] [--head-dim N] [--projection-dim N] [--max-cases N]"
                        );
                        std::process::exit(0);
                    }
                    other => return Err(format!("unknown option '{other}'").into()),
                }
            }

            if projection_dim > head_dim {
                return Err("--projection-dim must be <= --head-dim".into());
            }

            Ok(Self {
                out_dir,
                model_path,
                xr09_summary_path,
                xr09_records_path,
                trials,
                head_dim,
                projection_dim,
                max_cases,
            })
        }
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct Xr09Summary {
        run_id: String,
        decision: String,
        git_sha: String,
        selected_cases: Vec<Xr09Case>,
        #[serde(default)]
        policy_decision: serde_json::Value,
        #[serde(default)]
        model_identity: serde_json::Value,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct Xr09Case {
        case_id: String,
        workload_id: String,
        family: String,
        prompt_path: String,
        prompt_sha256: String,
        source_deterministic_seed: u64,
        target_context_tokens: usize,
        actual_context_tokens: usize,
        context_tokens: usize,
        prefix_token_hash: String,
    }

    #[derive(Debug, Clone, Deserialize)]
    struct Xr09RecordLine {
        #[serde(default)]
        run_id: Option<String>,
        #[serde(default)]
        git_sha: Option<String>,
        case_id: String,
        workload_id: String,
        #[serde(default)]
        modes: Vec<Xr09ModeLine>,
    }

    #[derive(Debug, Clone, Deserialize)]
    struct Xr09ModeLine {
        cache_mode: String,
        #[serde(default)]
        warm_restore_ms: Option<f64>,
        #[serde(default)]
        memory: Option<Xr09MemoryLine>,
        #[serde(default)]
        quality_gate: Option<Xr09QualityLine>,
    }

    #[derive(Debug, Clone, Deserialize)]
    struct Xr09MemoryLine {
        #[serde(default)]
        payload_memory_reduction: Option<f64>,
        #[serde(default)]
        active_kv_memory_reduction: Option<f64>,
        #[serde(default)]
        compressed_payload_bytes: Option<u64>,
        #[serde(default)]
        restored_active_kv_bytes: Option<u64>,
    }

    #[derive(Debug, Clone, Deserialize)]
    struct Xr09QualityLine {
        #[serde(default)]
        passed: Option<bool>,
    }

    #[derive(Debug, Clone, Serialize)]
    struct Xr13Record {
        schema_version: u32,
        goal: String,
        run_id: String,
        timestamp_unix: u64,
        git_sha: String,
        git_status_short: String,
        mode: String,
        feature_flag: String,
        case_id: String,
        workload_id: String,
        family: String,
        prompt_path: String,
        prompt_sha256: String,
        source_deterministic_seed: u64,
        deterministic_seed: u64,
        target_context_tokens: usize,
        actual_context_tokens: usize,
        context_tokens: usize,
        prefix_token_hash: String,
        trial_index: usize,
        head_dim: usize,
        projection_dim: usize,
        xr09_reference: Xr09Reference,
        variants: Vec<VariantRecord>,
        gate: RecordGate,
        blockers: Vec<String>,
    }

    #[derive(Debug, Clone, Serialize)]
    struct VariantRecord {
        variant: String,
        role: String,
        track: String,
        status: String,
        correctness: CorrectnessRecord,
        memory: MemoryRecord,
        latency: LatencyRecord,
        score_checksum: f64,
    }

    #[derive(Debug, Clone, Serialize)]
    struct CorrectnessRecord {
        reference_top1: usize,
        variant_top1: usize,
        top1_match: bool,
        reference_top1_in_variant_top8: bool,
        top8_overlap: usize,
        score_rmse: f64,
        mean_abs_score_delta: f64,
        max_abs_score_delta: f64,
        correctness_gate_passed: bool,
        gate_notes: String,
    }

    #[derive(Debug, Clone, Serialize)]
    struct MemoryRecord {
        active_k_bytes: u64,
        bf16_reference_k_bytes: u64,
        q8_reference_k_bytes: u64,
        active_reduction_vs_bf16: f64,
        active_reduction_vs_q8: f64,
    }

    #[derive(Debug, Clone, Serialize)]
    struct LatencyRecord {
        encode_ms: f64,
        score_ms: f64,
        total_ms: f64,
    }

    #[derive(Debug, Clone, Serialize)]
    struct RecordGate {
        all_baselines_measured: bool,
        candidate_correctness_passed: bool,
        candidate_memory_reduced_vs_q8: bool,
        speed_claim_allowed: bool,
    }

    #[derive(Debug, Clone, Serialize)]
    struct Xr09Reference {
        run_id: String,
        decision: String,
        git_sha: String,
        case_id: String,
        workload_id: String,
        q8_quality_gate_passed: Option<bool>,
        q8_payload_memory_reduction: Option<f64>,
        q8_active_memory_reduction: Option<f64>,
        q8_warm_restore_ms: Option<f64>,
        q8_payload_bytes: Option<u64>,
        q8_restored_active_kv_bytes: Option<u64>,
        q4_quality_gate_passed: Option<bool>,
        q4_payload_memory_reduction: Option<f64>,
        q4_active_memory_reduction: Option<f64>,
        q4_warm_restore_ms: Option<f64>,
    }

    impl Xr09Reference {
        fn missing(summary: &Xr09Summary, case: &Xr09Case) -> Self {
            Self {
                run_id: summary.run_id.clone(),
                decision: summary.decision.clone(),
                git_sha: summary.git_sha.clone(),
                case_id: case.case_id.clone(),
                workload_id: case.workload_id.clone(),
                q8_quality_gate_passed: None,
                q8_payload_memory_reduction: None,
                q8_active_memory_reduction: None,
                q8_warm_restore_ms: None,
                q8_payload_bytes: None,
                q8_restored_active_kv_bytes: None,
                q4_quality_gate_passed: None,
                q4_payload_memory_reduction: None,
                q4_active_memory_reduction: None,
                q4_warm_restore_ms: None,
            }
        }
    }

    #[derive(Debug, Clone, Serialize)]
    struct Summary {
        schema_version: u32,
        goal: String,
        status: String,
        decision: String,
        run_id: String,
        timestamp_unix: u64,
        mode: String,
        feature_flag: String,
        command: String,
        reproduction_command: String,
        git_sha: String,
        git_status_short: String,
        model_path: String,
        model_identity: manifest::ArtifactIdentity,
        xr09_summary_path: String,
        xr09_records_path: String,
        xr09_run_id: String,
        xr09_decision: String,
        xr09_git_sha: String,
        xr09_policy_decision: serde_json::Value,
        xr09_model_identity: serde_json::Value,
        xr09_references: Vec<Xr09Reference>,
        out_dir: String,
        trials: usize,
        head_dim: usize,
        projection_dim: usize,
        selected_cases: Vec<SelectedCaseEvidence>,
        generated_files: Vec<String>,
        records_count: usize,
        environment: Environment,
        relevant_environment: BTreeMap<String, Option<String>>,
        aggregates: BTreeMap<String, VariantAggregate>,
        blockers: Vec<String>,
        failed_hypotheses: Vec<String>,
        measurement_notes: Vec<&'static str>,
    }

    #[derive(Debug, Clone, Serialize)]
    struct SelectedCaseEvidence {
        case_id: String,
        workload_id: String,
        family: String,
        prompt_path: String,
        prompt_sha256: String,
        source_deterministic_seed: u64,
        target_context_tokens: usize,
        actual_context_tokens: usize,
        context_tokens: usize,
        prefix_token_hash: String,
        derived_trial_seeds: Vec<u64>,
    }

    #[derive(Debug, Clone, Serialize)]
    struct VariantAggregate {
        variant: String,
        role: String,
        track: String,
        samples: usize,
        correctness_passes: usize,
        top1_match_rate: f64,
        top8_recall_rate: f64,
        top8_overlap_p50: f64,
        score_rmse_p50: f64,
        score_rmse_p95: f64,
        max_abs_score_delta_p95: f64,
        active_reduction_vs_bf16_p50: f64,
        active_reduction_vs_q8_p50: f64,
        score_ms_p50: f64,
        score_ms_p95: f64,
        encode_ms_p50: f64,
        total_ms_p50: f64,
        total_ms_p95: f64,
    }

    #[derive(Debug, Clone, Serialize)]
    struct Environment {
        machine: String,
        macos: String,
        rustc: String,
        cargo: String,
        git_sha: String,
        git_status_short: String,
        hw_memsize_bytes: Option<u64>,
    }

    #[derive(Debug, Clone, Copy)]
    struct VariantSpec {
        variant: &'static str,
        role: &'static str,
        track: &'static str,
        kind: EncodingKind,
    }

    #[derive(Debug, Clone, Copy)]
    enum EncodingKind {
        Bf16,
        Q8,
        Planar4,
        Turbo,
    }

    const VARIANTS: [VariantSpec; 4] = [
        VariantSpec {
            variant: "bf16_reference",
            role: "baseline",
            track: "bf16_reference",
            kind: EncodingKind::Bf16,
        },
        VariantSpec {
            variant: "mlx_affine_q8_reference",
            role: "baseline",
            track: "xr09_q8_reference",
            kind: EncodingKind::Q8,
        },
        VariantSpec {
            variant: "planar4_k_only_candidate",
            role: "candidate",
            track: "planar_iso_k_only_compression",
            kind: EncodingKind::Planar4,
        },
        VariantSpec {
            variant: "turbo_score_estimation_candidate",
            role: "candidate",
            track: "turbo_score_estimation",
            kind: EncodingKind::Turbo,
        },
    ];

    struct CaseData {
        query: Vec<f32>,
        keys: Vec<f32>,
    }

    enum EncodedKeys {
        Bf16 {
            keys: Vec<f32>,
        },
        Quantized {
            values: Vec<i8>,
            scales: Vec<f32>,
            bits: u8,
        },
        Turbo {
            values: Vec<i8>,
            scales: Vec<f32>,
            projected_query: Vec<f32>,
            projection_dim: usize,
        },
    }

    impl EncodedKeys {
        fn score(&self, query: &[f32], context_tokens: usize, head_dim: usize) -> Vec<f32> {
            match self {
                Self::Bf16 { keys } => score_bf16(keys, query, context_tokens, head_dim),
                Self::Quantized {
                    values,
                    scales,
                    bits,
                    ..
                } => score_quantized(values, scales, *bits, query, context_tokens, head_dim),
                Self::Turbo {
                    values,
                    scales,
                    projected_query,
                    projection_dim,
                } => score_turbo(
                    values,
                    scales,
                    projected_query,
                    context_tokens,
                    head_dim,
                    *projection_dim,
                ),
            }
        }

        fn estimated_active_k_bytes(&self, context_tokens: usize, head_dim: usize) -> u64 {
            match self {
                Self::Bf16 { .. } => bf16_k_bytes(context_tokens, head_dim),
                Self::Quantized { bits, .. } if *bits == 8 => q8_k_bytes(context_tokens, head_dim),
                Self::Quantized { bits, .. } if *bits == 4 => q4_k_bytes(context_tokens, head_dim),
                Self::Turbo { projection_dim, .. } => {
                    turbo_k_bytes(context_tokens, *projection_dim)
                }
                Self::Quantized { .. } => 0,
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_case(
        case: &Xr09Case,
        xr09_reference: Xr09Reference,
        args: &Args,
        run_id: &str,
        timestamp_unix: u64,
        deterministic_seed: u64,
        trial_index: usize,
        environment: &Environment,
        _model_identity: &manifest::ArtifactIdentity,
    ) -> Result<Xr13Record, Box<dyn std::error::Error>> {
        let data = build_case_data(case.context_tokens, args.head_dim, deterministic_seed);
        let reference_scores =
            score_bf16(&data.keys, &data.query, case.context_tokens, args.head_dim);
        let reference_top8 = top_k_indices(&reference_scores, 8);
        let bf16_bytes = bf16_k_bytes(case.context_tokens, args.head_dim);
        let q8_bytes = q8_k_bytes(case.context_tokens, args.head_dim);

        let mut variants = Vec::new();
        for spec in VARIANTS {
            let encode_started = Instant::now();
            let encoded = encode_keys(
                spec.kind,
                &data,
                case.context_tokens,
                args.head_dim,
                args.projection_dim,
            )?;
            let encode_ms = duration_ms(encode_started.elapsed());

            let score_started = Instant::now();
            let scores = encoded.score(&data.query, case.context_tokens, args.head_dim);
            let checksum = score_checksum(&scores);
            black_box(checksum);
            let score_ms = duration_ms(score_started.elapsed());
            let total_ms = encode_ms + score_ms;
            let memory_bytes = encoded.estimated_active_k_bytes(case.context_tokens, args.head_dim);
            let correctness =
                evaluate_correctness(spec, &reference_scores, &reference_top8, &scores);
            let status = if correctness.correctness_gate_passed {
                "passed"
            } else {
                "failed"
            };

            variants.push(VariantRecord {
                variant: spec.variant.to_owned(),
                role: spec.role.to_owned(),
                track: spec.track.to_owned(),
                status: status.to_owned(),
                correctness,
                memory: MemoryRecord {
                    active_k_bytes: memory_bytes,
                    bf16_reference_k_bytes: bf16_bytes,
                    q8_reference_k_bytes: q8_bytes,
                    active_reduction_vs_bf16: reduction(memory_bytes, bf16_bytes),
                    active_reduction_vs_q8: reduction(memory_bytes, q8_bytes),
                },
                latency: LatencyRecord {
                    encode_ms,
                    score_ms,
                    total_ms,
                },
                score_checksum: checksum,
            });
        }

        let candidate_variants = variants
            .iter()
            .filter(|variant| variant.role == "candidate")
            .collect::<Vec<_>>();
        let candidate_correctness_passed = candidate_variants
            .iter()
            .all(|variant| variant.correctness.correctness_gate_passed);
        let candidate_memory_reduced_vs_q8 = candidate_variants
            .iter()
            .all(|variant| variant.memory.active_reduction_vs_q8 > 0.0);
        let all_baselines_measured = variants
            .iter()
            .filter(|variant| variant.role == "baseline")
            .all(|variant| variant.latency.score_ms >= 0.0);

        Ok(Xr13Record {
            schema_version: 1,
            goal: GOAL.to_owned(),
            run_id: run_id.to_owned(),
            timestamp_unix,
            git_sha: environment.git_sha.clone(),
            git_status_short: environment.git_status_short.clone(),
            mode: MODE.to_owned(),
            feature_flag: FEATURE_FLAG.to_owned(),
            case_id: case.case_id.clone(),
            workload_id: case.workload_id.clone(),
            family: case.family.clone(),
            prompt_path: case.prompt_path.clone(),
            prompt_sha256: case.prompt_sha256.clone(),
            source_deterministic_seed: case.source_deterministic_seed,
            deterministic_seed,
            target_context_tokens: case.target_context_tokens,
            actual_context_tokens: case.actual_context_tokens,
            context_tokens: case.context_tokens,
            prefix_token_hash: case.prefix_token_hash.clone(),
            trial_index,
            head_dim: args.head_dim,
            projection_dim: args.projection_dim,
            xr09_reference,
            variants,
            gate: RecordGate {
                all_baselines_measured,
                candidate_correctness_passed,
                candidate_memory_reduced_vs_q8,
                speed_claim_allowed: all_baselines_measured
                    && candidate_correctness_passed
                    && candidate_memory_reduced_vs_q8,
            },
            blockers: Vec::new(),
        })
    }

    fn encode_keys(
        kind: EncodingKind,
        data: &CaseData,
        context_tokens: usize,
        head_dim: usize,
        projection_dim: usize,
    ) -> Result<EncodedKeys, Box<dyn std::error::Error>> {
        match kind {
            EncodingKind::Bf16 => Ok(EncodedKeys::Bf16 {
                keys: data.keys.clone(),
            }),
            EncodingKind::Q8 => Ok(encode_quantized(&data.keys, context_tokens, head_dim, 8)),
            EncodingKind::Planar4 => Ok(encode_quantized(&data.keys, context_tokens, head_dim, 4)),
            EncodingKind::Turbo => {
                if projection_dim == 0 || projection_dim > head_dim {
                    return Err("projection_dim must be in 1..=head_dim".into());
                }
                Ok(encode_turbo(
                    &data.keys,
                    &data.query,
                    context_tokens,
                    head_dim,
                    projection_dim,
                ))
            }
        }
    }

    fn build_case_data(context_tokens: usize, head_dim: usize, seed: u64) -> CaseData {
        let mut rng = SplitMix64::new(seed ^ 0x9e37_79b9_7f4a_7c15);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut query = Vec::with_capacity(head_dim);
        for _ in 0..head_dim {
            query.push(round_to_bf16(rng.next_signed_unit_f32() * scale));
        }
        let mut keys = Vec::with_capacity(context_tokens * head_dim);
        for token_index in 0..context_tokens {
            let token_bias =
                ((((token_index as u64).wrapping_mul(131)) & 0xff) as f32 - 127.5) / 4096.0;
            for _ in 0..head_dim {
                keys.push(round_to_bf16(
                    rng.next_signed_unit_f32() * scale + token_bias,
                ));
            }
        }
        CaseData { query, keys }
    }

    fn score_bf16(keys: &[f32], query: &[f32], context_tokens: usize, head_dim: usize) -> Vec<f32> {
        let mut scores = Vec::with_capacity(context_tokens);
        for token_index in 0..context_tokens {
            let start = token_index * head_dim;
            let mut sum = 0.0f32;
            for dim in 0..head_dim {
                sum += keys[start + dim] * query[dim];
            }
            scores.push(sum);
        }
        scores
    }

    fn encode_quantized(
        keys: &[f32],
        context_tokens: usize,
        head_dim: usize,
        bits: u8,
    ) -> EncodedKeys {
        let max_level = if bits == 8 { 127.0 } else { 7.0 };
        let mut values = Vec::with_capacity(context_tokens * head_dim);
        let mut scales = Vec::with_capacity(context_tokens);
        for token_index in 0..context_tokens {
            let start = token_index * head_dim;
            let token = &keys[start..start + head_dim];
            let max_abs = token.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
            let scale = if max_abs == 0.0 {
                1.0
            } else {
                max_abs / max_level
            };
            scales.push(scale);
            for value in token {
                let quantized = (*value / scale).round().clamp(-max_level, max_level) as i8;
                values.push(quantized);
            }
        }
        EncodedKeys::Quantized {
            values,
            scales,
            bits,
        }
    }

    fn score_quantized(
        values: &[i8],
        scales: &[f32],
        bits: u8,
        query: &[f32],
        context_tokens: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let clamp = if bits == 8 { 127.0 } else { 7.0 };
        let mut scores = Vec::with_capacity(context_tokens);
        for (token_index, scale) in scales.iter().copied().enumerate().take(context_tokens) {
            let start = token_index * head_dim;
            let mut sum = 0.0f32;
            for dim in 0..head_dim {
                let dequant = (values[start + dim] as f32).clamp(-clamp, clamp) * scale;
                sum += dequant * query[dim];
            }
            scores.push(sum);
        }
        scores
    }

    fn encode_turbo(
        keys: &[f32],
        query: &[f32],
        context_tokens: usize,
        head_dim: usize,
        projection_dim: usize,
    ) -> EncodedKeys {
        let projection_scale = (head_dim as f32 / projection_dim as f32).sqrt();
        let projected_query = (0..projection_dim)
            .map(|projection_index| {
                query[turbo_projection_index(projection_index, head_dim)] * projection_scale
            })
            .collect::<Vec<_>>();
        let mut values = Vec::with_capacity(context_tokens * projection_dim);
        let mut scales = Vec::with_capacity(context_tokens);
        for token_index in 0..context_tokens {
            let start = token_index * head_dim;
            let mut projected = Vec::with_capacity(projection_dim);
            for projection_index in 0..projection_dim {
                let key_index = start + turbo_projection_index(projection_index, head_dim);
                projected.push(keys[key_index] * projection_scale);
            }
            let max_abs = projected
                .iter()
                .fold(0.0f32, |acc, value| acc.max(value.abs()));
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
            scales.push(scale);
            for value in projected {
                values.push((value / scale).round().clamp(-127.0, 127.0) as i8);
            }
        }
        EncodedKeys::Turbo {
            values,
            scales,
            projected_query,
            projection_dim,
        }
    }

    fn score_turbo(
        values: &[i8],
        scales: &[f32],
        projected_query: &[f32],
        context_tokens: usize,
        _head_dim: usize,
        projection_dim: usize,
    ) -> Vec<f32> {
        let mut scores = Vec::with_capacity(context_tokens);
        for (token_index, scale) in scales.iter().copied().enumerate().take(context_tokens) {
            let start = token_index * projection_dim;
            let mut sum = 0.0f32;
            for dim in 0..projection_dim {
                sum += values[start + dim] as f32 * scale * projected_query[dim];
            }
            scores.push(sum);
        }
        scores
    }

    fn evaluate_correctness(
        spec: VariantSpec,
        reference_scores: &[f32],
        reference_top8: &[usize],
        scores: &[f32],
    ) -> CorrectnessRecord {
        let reference_top1 = reference_top8[0];
        let variant_top8 = top_k_indices(scores, 8);
        let variant_top1 = variant_top8[0];
        let top1_match = reference_top1 == variant_top1;
        let reference_top1_in_variant_top8 = variant_top8.contains(&reference_top1);
        let top8_overlap = reference_top8
            .iter()
            .filter(|candidate| variant_top8.contains(candidate))
            .count();
        let (score_rmse, mean_abs_score_delta, max_abs_score_delta) =
            score_error(reference_scores, scores);
        let (correctness_gate_passed, gate_notes) = match spec.kind {
            EncodingKind::Bf16 => (
                top1_match && score_rmse == 0.0,
                "self-reference gate requires exact score reuse".to_owned(),
            ),
            EncodingKind::Q8 => (
                top1_match && score_rmse <= 0.003 && max_abs_score_delta <= 0.020,
                "q8 reference gate: top1 match, RMSE <= 0.003, max delta <= 0.020".to_owned(),
            ),
            EncodingKind::Planar4 => (
                top1_match
                    && reference_top1_in_variant_top8
                    && score_rmse <= 0.008
                    && max_abs_score_delta <= 0.040,
                "Planar4 gate: top1 match, reference top1 in top8, RMSE <= 0.008, max delta <= 0.040".to_owned(),
            ),
            EncodingKind::Turbo => (
                reference_top1_in_variant_top8 && top8_overlap >= 4 && score_rmse <= 0.020,
                "Turbo gate: reference top1 in top8, top8 overlap >= 4, RMSE <= 0.020".to_owned(),
            ),
        };

        CorrectnessRecord {
            reference_top1,
            variant_top1,
            top1_match,
            reference_top1_in_variant_top8,
            top8_overlap,
            score_rmse,
            mean_abs_score_delta,
            max_abs_score_delta,
            correctness_gate_passed,
            gate_notes,
        }
    }

    fn score_error(reference_scores: &[f32], scores: &[f32]) -> (f64, f64, f64) {
        let mut squared = 0.0f64;
        let mut abs_sum = 0.0f64;
        let mut max_abs = 0.0f64;
        for (reference, candidate) in reference_scores.iter().zip(scores.iter()) {
            let delta = (*candidate as f64) - (*reference as f64);
            let abs = delta.abs();
            squared += delta * delta;
            abs_sum += abs;
            max_abs = max_abs.max(abs);
        }
        let len = reference_scores.len().max(1) as f64;
        ((squared / len).sqrt(), abs_sum / len, max_abs)
    }

    fn aggregate_variants(records: &[Xr13Record]) -> BTreeMap<String, VariantAggregate> {
        let mut grouped: BTreeMap<String, Vec<&VariantRecord>> = BTreeMap::new();
        for record in records {
            for variant in &record.variants {
                grouped
                    .entry(variant.variant.clone())
                    .or_default()
                    .push(variant);
            }
        }

        grouped
            .into_iter()
            .map(|(variant, items)| {
                let samples = items.len();
                let correctness_passes = items
                    .iter()
                    .filter(|item| item.correctness.correctness_gate_passed)
                    .count();
                let top1_matches = items
                    .iter()
                    .filter(|item| item.correctness.top1_match)
                    .count();
                let top8_recalls = items
                    .iter()
                    .filter(|item| item.correctness.reference_top1_in_variant_top8)
                    .count();
                let score_rmse = items
                    .iter()
                    .map(|item| item.correctness.score_rmse)
                    .collect::<Vec<_>>();
                let max_delta = items
                    .iter()
                    .map(|item| item.correctness.max_abs_score_delta)
                    .collect::<Vec<_>>();
                let top8_overlap = items
                    .iter()
                    .map(|item| item.correctness.top8_overlap as f64)
                    .collect::<Vec<_>>();
                let reduction_bf16 = items
                    .iter()
                    .map(|item| item.memory.active_reduction_vs_bf16)
                    .collect::<Vec<_>>();
                let reduction_q8 = items
                    .iter()
                    .map(|item| item.memory.active_reduction_vs_q8)
                    .collect::<Vec<_>>();
                let score_ms = items
                    .iter()
                    .map(|item| item.latency.score_ms)
                    .collect::<Vec<_>>();
                let encode_ms = items
                    .iter()
                    .map(|item| item.latency.encode_ms)
                    .collect::<Vec<_>>();
                let total_ms = items
                    .iter()
                    .map(|item| item.latency.total_ms)
                    .collect::<Vec<_>>();
                let first = items[0];
                (
                    variant.clone(),
                    VariantAggregate {
                        variant,
                        role: first.role.clone(),
                        track: first.track.clone(),
                        samples,
                        correctness_passes,
                        top1_match_rate: ratio(top1_matches, samples),
                        top8_recall_rate: ratio(top8_recalls, samples),
                        top8_overlap_p50: percentile(top8_overlap, 0.50),
                        score_rmse_p50: percentile(score_rmse.clone(), 0.50),
                        score_rmse_p95: percentile(score_rmse, 0.95),
                        max_abs_score_delta_p95: percentile(max_delta, 0.95),
                        active_reduction_vs_bf16_p50: percentile(reduction_bf16, 0.50),
                        active_reduction_vs_q8_p50: percentile(reduction_q8, 0.50),
                        score_ms_p50: percentile(score_ms.clone(), 0.50),
                        score_ms_p95: percentile(score_ms, 0.95),
                        encode_ms_p50: percentile(encode_ms, 0.50),
                        total_ms_p50: percentile(total_ms.clone(), 0.50),
                        total_ms_p95: percentile(total_ms, 0.95),
                    },
                )
            })
            .collect()
    }

    fn hard_blockers(records: &[Xr13Record]) -> Vec<String> {
        let mut blockers = records
            .iter()
            .flat_map(|record| record.blockers.iter().cloned())
            .collect::<Vec<_>>();
        blockers.sort();
        blockers.dedup();
        blockers
    }

    fn failed_hypotheses(aggregates: &BTreeMap<String, VariantAggregate>) -> Vec<String> {
        let mut failures = Vec::new();
        for aggregate in aggregates.values() {
            if aggregate.role == "candidate" && aggregate.correctness_passes < aggregate.samples {
                failures.push(format!(
                    "{} failed correctness on {}/{} samples; top1_match_rate={:.3}, top8_recall_rate={:.3}, rmse_p95={:.6}",
                    aggregate.variant,
                    aggregate.samples - aggregate.correctness_passes,
                    aggregate.samples,
                    aggregate.top1_match_rate,
                    aggregate.top8_recall_rate,
                    aggregate.score_rmse_p95
                ));
            }
        }
        failures.push(
            "MLX custom Metal kernel integration remains deferred: XR13 measured the isolated score path only and did not add a C ABI or runtime decode path."
                .to_owned(),
        );
        failures.push(
            "Active compressed KV decode remains disabled by default; XR13 did not import compressed K into the production native KV cache."
                .to_owned(),
        );
        failures
    }

    fn load_xr09_summary(path: &Path) -> Result<Xr09Summary, Box<dyn std::error::Error>> {
        let bytes = fs::read(path)?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    fn load_xr09_references(
        path: &Path,
        summary: &Xr09Summary,
    ) -> Result<HashMap<String, Xr09Reference>, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut out = HashMap::new();
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let record: Xr09RecordLine = serde_json::from_str(&line)?;
            let mut reference = Xr09Reference {
                run_id: record
                    .run_id
                    .clone()
                    .unwrap_or_else(|| summary.run_id.clone()),
                decision: summary.decision.clone(),
                git_sha: record
                    .git_sha
                    .clone()
                    .unwrap_or_else(|| summary.git_sha.clone()),
                case_id: record.case_id.clone(),
                workload_id: record.workload_id.clone(),
                q8_quality_gate_passed: None,
                q8_payload_memory_reduction: None,
                q8_active_memory_reduction: None,
                q8_warm_restore_ms: None,
                q8_payload_bytes: None,
                q8_restored_active_kv_bytes: None,
                q4_quality_gate_passed: None,
                q4_payload_memory_reduction: None,
                q4_active_memory_reduction: None,
                q4_warm_restore_ms: None,
            };
            for mode in record.modes {
                match mode.cache_mode.as_str() {
                    "mlx_affine_q8" => {
                        reference.q8_quality_gate_passed =
                            mode.quality_gate.as_ref().and_then(|gate| gate.passed);
                        reference.q8_payload_memory_reduction = mode
                            .memory
                            .as_ref()
                            .and_then(|memory| memory.payload_memory_reduction);
                        reference.q8_active_memory_reduction = mode
                            .memory
                            .as_ref()
                            .and_then(|memory| memory.active_kv_memory_reduction);
                        reference.q8_payload_bytes = mode
                            .memory
                            .as_ref()
                            .and_then(|memory| memory.compressed_payload_bytes);
                        reference.q8_restored_active_kv_bytes = mode
                            .memory
                            .as_ref()
                            .and_then(|memory| memory.restored_active_kv_bytes);
                        reference.q8_warm_restore_ms = mode.warm_restore_ms;
                    }
                    "mlx_affine_q4" => {
                        reference.q4_quality_gate_passed =
                            mode.quality_gate.as_ref().and_then(|gate| gate.passed);
                        reference.q4_payload_memory_reduction = mode
                            .memory
                            .as_ref()
                            .and_then(|memory| memory.payload_memory_reduction);
                        reference.q4_active_memory_reduction = mode
                            .memory
                            .as_ref()
                            .and_then(|memory| memory.active_kv_memory_reduction);
                        reference.q4_warm_restore_ms = mode.warm_restore_ms;
                    }
                    _ => {}
                }
            }
            out.insert(record.case_id, reference);
        }
        Ok(out)
    }

    fn write_jsonl<T: Serialize>(
        path: &Path,
        records: &[T],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::create(path)?;
        for record in records {
            serde_json::to_writer(&mut file, record)?;
            file.write_all(b"\n")?;
        }
        Ok(())
    }

    fn render_report(summary: &Summary) -> String {
        let mut out = String::new();
        out.push_str("# XR13 Novel Metal/KV Exploration\n\n");
        out.push_str("## Summary\n\n");
        out.push_str(&format!("- Status: `{}`\n", summary.status));
        out.push_str(&format!("- Decision: `{}`\n", summary.decision));
        out.push_str(&format!("- Run ID: `{}`\n", summary.run_id));
        out.push_str(&format!("- Feature flag: `{}`\n", summary.feature_flag));
        out.push_str(&format!("- Mode: `{}`\n", summary.mode));
        out.push_str(&format!("- Git SHA: `{}`\n", summary.git_sha));
        out.push_str(&format!(
            "- XR09 baseline: `{}` at `{}` decision `{}`\n",
            summary.xr09_run_id, summary.xr09_git_sha, summary.xr09_decision
        ));
        out.push_str(&format!(
            "- Head/projection dims: `{}` / `{}`\n",
            summary.head_dim, summary.projection_dim
        ));
        out.push_str(&format!(
            "- Records: `{}` across `{}` selected cases and `{}` trials\n\n",
            summary.records_count,
            summary.selected_cases.len(),
            summary.trials
        ));

        out.push_str("## Commands\n\n");
        out.push_str("```text\n");
        out.push_str(&summary.command);
        out.push('\n');
        out.push_str(&summary.reproduction_command);
        out.push_str("\n```\n\n");

        out.push_str("## Generated Files\n\n");
        for path in &summary.generated_files {
            out.push_str(&format!("- `{path}`\n"));
        }
        out.push('\n');

        out.push_str("## XR09 BF16/q8 Baseline\n\n");
        out.push_str("| Workload | XR09 q8 gate | q8 payload reduction | q8 active reduction | q8 warm restore ms |\n");
        out.push_str("|---|---:|---:|---:|---:|\n");
        for reference in &summary.xr09_references {
            out.push_str(&format!(
                "| `{}` | `{}` | {} | {} | {} |\n",
                reference.workload_id,
                display_option_bool(reference.q8_quality_gate_passed),
                display_option_percent(reference.q8_payload_memory_reduction),
                display_option_percent(reference.q8_active_memory_reduction),
                display_option_f64(reference.q8_warm_restore_ms),
            ));
        }
        out.push('\n');

        out.push_str("## Variant Aggregates\n\n");
        out.push_str("| Variant | Role | Correctness | Top1 | Top8 | RMSE p95 | Active K vs BF16 | Active K vs q8 | Score p50 ms | Score p95 ms |\n");
        out.push_str("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");
        for aggregate in summary.aggregates.values() {
            out.push_str(&format!(
                "| `{}` | `{}` | {}/{} | {:.3} | {:.3} | {:.6} | {:.2}% | {:.2}% | {:.3} | {:.3} |\n",
                aggregate.variant,
                aggregate.role,
                aggregate.correctness_passes,
                aggregate.samples,
                aggregate.top1_match_rate,
                aggregate.top8_recall_rate,
                aggregate.score_rmse_p95,
                aggregate.active_reduction_vs_bf16_p50 * 100.0,
                aggregate.active_reduction_vs_q8_p50 * 100.0,
                aggregate.score_ms_p50,
                aggregate.score_ms_p95,
            ));
        }
        out.push('\n');

        out.push_str("## Selected Cases\n\n");
        out.push_str("| Workload | Tokens | Source Seed | Trial Seeds |\n");
        out.push_str("|---|---:|---:|---|\n");
        for case in &summary.selected_cases {
            let seeds = case
                .derived_trial_seeds
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&format!(
                "| `{}` | {} | {} | `{}` |\n",
                case.workload_id, case.context_tokens, case.source_deterministic_seed, seeds
            ));
        }
        out.push('\n');

        out.push_str("## Measurement Notes\n\n");
        for note in &summary.measurement_notes {
            out.push_str(&format!("- {note}\n"));
        }
        out.push('\n');

        out.push_str("## Failed Or Deferred Hypotheses\n\n");
        for failure in &summary.failed_hypotheses {
            out.push_str(&format!("- {failure}\n"));
        }

        out
    }

    fn render_blockers(summary: &Summary) -> String {
        let mut out = String::new();
        out.push_str("# XR13 Novel Metal/KV Exploration Blockers\n\n");
        out.push_str("## Hard Blockers\n\n");
        if summary.blockers.is_empty() {
            out.push_str("- None. The prototype ran to completion and generated raw records.\n");
        } else {
            for blocker in &summary.blockers {
                out.push_str(&format!("- {blocker}\n"));
            }
        }
        out.push_str("\n## Failed Or Deferred Hypotheses\n\n");
        for failure in &summary.failed_hypotheses {
            out.push_str(&format!("- {failure}\n"));
        }
        out.push_str("\n## No-Go Boundary\n\n");
        out.push_str("- Do not enable active compressed KV decode from XR13 evidence alone.\n");
        out.push_str("- Do not merge a custom Metal kernel without a later narrow C ABI design and real-model correctness gate.\n");
        out
    }

    fn render_decision(summary: &Summary) -> String {
        let mut out = String::new();
        out.push_str("# XR13 Novel Metal/KV Exploration Decision\n\n");
        out.push_str(&format!("- Decision: `{}`\n", summary.decision));
        out.push_str(&format!("- Status: `{}`\n", summary.status));
        out.push_str(&format!("- Run ID: `{}`\n", summary.run_id));
        out.push_str(&format!("- Feature flag: `{}`\n", summary.feature_flag));
        out.push_str(&format!(
            "- Raw records: `{}/records.jsonl`\n\n",
            summary.out_dir
        ));

        out.push_str("## Rationale\n\n");
        if summary.decision == "reject_candidate" {
            out.push_str("One or more candidate tracks failed the isolated correctness gate, so no XR13 candidate is promoted toward runtime or Metal integration.\n\n");
        } else if summary.decision == "keep_experimental" {
            out.push_str("Candidates passed the isolated gate, but evidence remains synthetic and runtime/Metal integration is still out of scope for this goal.\n\n");
        } else {
            out.push_str("The run is blocked; see blockers.md for the required next input.\n\n");
        }

        out.push_str("## Evidence\n\n");
        for file in &summary.generated_files {
            out.push_str(&format!("- `{file}`\n"));
        }
        out.push('\n');
        out.push_str("## Policy\n\n");
        out.push_str("- No default runtime path changed.\n");
        out.push_str("- Active compressed KV decode remains disabled.\n");
        out.push_str(
            "- Any speed claim is limited to the feature-gated isolated score microbenchmark.\n",
        );
        out
    }

    fn measurement_notes() -> Vec<&'static str> {
        vec![
            "XR13 reads XR09 selected cases, context token lengths, source seeds, and prefix hashes.",
            "The benchmark uses deterministic synthetic KV-like vectors; it does not execute the Gemma model.",
            "BF16 and q8 are comparison baselines; Planar4 and Turbo are candidate tracks.",
            "Active memory values are estimated active K bytes for the isolated score path, not full model KV memory.",
            "Latency is CPU-side prototype loop timing from the feature-gated example, not an MLX/Metal kernel claim.",
            "No default runtime path, server path, cache import path, or production decode path is modified.",
        ]
    }

    fn display_option_bool(value: Option<bool>) -> String {
        value
            .map(|value| value.to_string())
            .unwrap_or_else(|| "n/a".to_owned())
    }

    fn display_option_percent(value: Option<f64>) -> String {
        value
            .map(|value| format!("{:.2}%", value * 100.0))
            .unwrap_or_else(|| "n/a".to_owned())
    }

    fn display_option_f64(value: Option<f64>) -> String {
        value
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "n/a".to_owned())
    }

    fn capture_environment() -> Environment {
        Environment {
            machine: command_stdout("uname", &["-m"]).unwrap_or_else(|| "unknown".to_owned()),
            macos: command_stdout("sw_vers", &["-productVersion"])
                .unwrap_or_else(|| "unknown".to_owned()),
            rustc: command_stdout("rustc", &["-V"]).unwrap_or_else(|| "unknown".to_owned()),
            cargo: command_stdout("cargo", &["-V"]).unwrap_or_else(|| "unknown".to_owned()),
            git_sha: command_stdout("git", &["rev-parse", "HEAD"])
                .unwrap_or_else(|| "unknown".to_owned()),
            git_status_short: command_stdout("git", &["status", "--short"])
                .unwrap_or_else(|| "unknown".to_owned()),
            hw_memsize_bytes: command_stdout("sysctl", &["-n", "hw.memsize"])
                .and_then(|value| value.parse::<u64>().ok()),
        }
    }

    fn capture_relevant_environment() -> BTreeMap<String, Option<String>> {
        [
            "GEMMA4D_REQUIRE_MLX",
            "GEMMA4D_USE_NATIVE_GRAPH",
            "GEMMA4D_MODEL_PATH",
            "GEMMA4D_MODEL_REVISION",
            "RUSTFLAGS",
        ]
        .into_iter()
        .map(|key| (key.to_owned(), env::var(key).ok()))
        .collect()
    }

    fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
        let output = Command::new(program).args(args).output().ok()?;
        if !output.status.success() {
            return None;
        }
        Some(String::from_utf8_lossy(&output.stdout).trim().to_owned())
    }

    fn command_display() -> String {
        env::args()
            .map(|arg| {
                if arg.contains(' ') {
                    format!("'{}'", arg.replace('\'', "'\\''"))
                } else {
                    arg
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn reproduction_command(args: &Args) -> String {
        let mut command = format!(
            "cargo run -p gemma4d-bench --features xr13-prototypes --example xr13_novel_metal_kv_exploration -- --out-dir {} --model-path {} --xr09-summary {} --xr09-records {} --trials {} --head-dim {} --projection-dim {}",
            args.out_dir.display(),
            args.model_path.display(),
            args.xr09_summary_path.display(),
            args.xr09_records_path.display(),
            args.trials,
            args.head_dim,
            args.projection_dim,
        );
        if let Some(max_cases) = args.max_cases {
            command.push_str(&format!(" --max-cases {max_cases}"));
        }
        command
    }

    fn run_id() -> String {
        format!("xr13-{}", unix_now())
    }

    fn unix_now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or(0)
    }

    fn duration_ms(duration: std::time::Duration) -> f64 {
        duration.as_secs_f64() * 1000.0
    }

    fn required(
        args: &mut impl Iterator<Item = String>,
        option: &str,
    ) -> Result<String, Box<dyn std::error::Error>> {
        args.next()
            .ok_or_else(|| format!("{option} requires a value").into())
    }

    fn required_path(
        args: &mut impl Iterator<Item = String>,
        option: &str,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        Ok(PathBuf::from(required(args, option)?))
    }

    fn parse_positive_usize(
        value: &str,
        option: &str,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let parsed = value
            .parse::<usize>()
            .map_err(|error| format!("{option} must be an integer: {error}"))?;
        if parsed == 0 {
            return Err(format!("{option} must be greater than zero").into());
        }
        Ok(parsed)
    }

    fn derived_seed(
        case: &Xr09Case,
        trial_index: usize,
        head_dim: usize,
        projection_dim: usize,
    ) -> u64 {
        mix64(
            case.source_deterministic_seed
                ^ hash_hex_prefix(&case.prefix_token_hash)
                ^ ((trial_index as u64) << 32)
                ^ ((head_dim as u64) << 16)
                ^ projection_dim as u64,
        )
    }

    fn hash_hex_prefix(value: &str) -> u64 {
        let prefix = value.chars().take(16).collect::<String>();
        u64::from_str_radix(&prefix, 16).unwrap_or_else(|_| fnv1a64(value.as_bytes()))
    }

    fn fnv1a64(bytes: &[u8]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for byte in bytes {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x1000_0000_01b3);
        }
        hash
    }

    fn mix64(mut value: u64) -> u64 {
        value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    struct SplitMix64 {
        state: u64,
    }

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            mix64(self.state)
        }

        fn next_signed_unit_f32(&mut self) -> f32 {
            let unit = ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32);
            (unit * 2.0) - 1.0
        }
    }

    fn round_to_bf16(value: f32) -> f32 {
        let bits = value.to_bits();
        let rounding_bias = ((bits >> 16) & 1) + 0x7fff;
        f32::from_bits((bits + rounding_bias) & 0xffff_0000)
    }

    fn turbo_projection_index(projection_index: usize, head_dim: usize) -> usize {
        (projection_index.wrapping_mul(37).wrapping_add(11)) % head_dim
    }

    fn top_k_indices(scores: &[f32], k: usize) -> Vec<usize> {
        let mut indices = (0..scores.len()).collect::<Vec<_>>();
        indices.sort_unstable_by(|left, right| {
            scores[*right]
                .partial_cmp(&scores[*left])
                .unwrap_or(Ordering::Equal)
        });
        indices.truncate(k.min(indices.len()));
        indices
    }

    fn score_checksum(scores: &[f32]) -> f64 {
        scores
            .iter()
            .enumerate()
            .fold(0.0f64, |acc, (index, score)| {
                acc + (*score as f64) * ((index as f64 % 17.0) + 1.0)
            })
    }

    fn bf16_k_bytes(context_tokens: usize, head_dim: usize) -> u64 {
        (context_tokens * head_dim * 2) as u64
    }

    fn q8_k_bytes(context_tokens: usize, head_dim: usize) -> u64 {
        (context_tokens * head_dim + context_tokens * std::mem::size_of::<f32>()) as u64
    }

    fn q4_k_bytes(context_tokens: usize, head_dim: usize) -> u64 {
        let packed_values = (context_tokens * head_dim).div_ceil(2);
        (packed_values + context_tokens * std::mem::size_of::<f32>()) as u64
    }

    fn turbo_k_bytes(context_tokens: usize, projection_dim: usize) -> u64 {
        (context_tokens * projection_dim + context_tokens * std::mem::size_of::<f32>()) as u64
    }

    fn reduction(active: u64, reference: u64) -> f64 {
        if reference == 0 {
            return 0.0;
        }
        1.0 - (active as f64 / reference as f64)
    }

    fn ratio(numerator: usize, denominator: usize) -> f64 {
        if denominator == 0 {
            0.0
        } else {
            numerator as f64 / denominator as f64
        }
    }

    fn percentile(mut values: Vec<f64>, percentile: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
        let index = ((values.len() - 1) as f64 * percentile).round() as usize;
        values[index.min(values.len() - 1)]
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn compressed_memory_estimates_are_smaller_than_bf16() {
            let tokens = 4096;
            let head_dim = 64;
            let bf16 = bf16_k_bytes(tokens, head_dim);
            let q8 = q8_k_bytes(tokens, head_dim);
            let q4 = q4_k_bytes(tokens, head_dim);
            let turbo = turbo_k_bytes(tokens, 8);

            assert!(q8 < bf16);
            assert!(q4 < q8);
            assert!(turbo < q4);
            assert!(reduction(q4, q8) > 0.0);
        }

        #[test]
        fn bf16_reference_scores_are_exact_self_comparison() {
            let case = Xr09Case {
                case_id: "case".to_owned(),
                workload_id: "workload".to_owned(),
                family: "family".to_owned(),
                prompt_path: "prompt".to_owned(),
                prompt_sha256: "sha".to_owned(),
                source_deterministic_seed: 20260630,
                target_context_tokens: 128,
                actual_context_tokens: 128,
                context_tokens: 128,
                prefix_token_hash: "0123456789abcdef".to_owned(),
            };
            let seed = derived_seed(&case, 0, 32, 8);
            let data = build_case_data(case.context_tokens, 32, seed);
            let scores = score_bf16(&data.keys, &data.query, case.context_tokens, 32);
            let top8 = top_k_indices(&scores, 8);
            let spec = VariantSpec {
                variant: "bf16_reference",
                role: "baseline",
                track: "bf16_reference",
                kind: EncodingKind::Bf16,
            };
            let correctness = evaluate_correctness(spec, &scores, &top8, &scores);
            assert!(correctness.correctness_gate_passed);
            assert_eq!(correctness.score_rmse, 0.0);
        }

        #[test]
        fn derived_seed_changes_by_trial_and_shape() {
            let case = Xr09Case {
                case_id: "case".to_owned(),
                workload_id: "workload".to_owned(),
                family: "family".to_owned(),
                prompt_path: "prompt".to_owned(),
                prompt_sha256: "sha".to_owned(),
                source_deterministic_seed: 20260630,
                target_context_tokens: 128,
                actual_context_tokens: 128,
                context_tokens: 128,
                prefix_token_hash: "0123456789abcdef".to_owned(),
            };

            let a = derived_seed(&case, 0, 64, 8);
            let b = derived_seed(&case, 1, 64, 8);
            let c = derived_seed(&case, 0, 32, 8);
            assert_ne!(a, b);
            assert_ne!(a, c);
        }
    }
}
