use std::{
    env, fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use gemma4d_bench::{BuildProvenance, capture_build_provenance, manifest};
use gemma4d_tokenizer::{file_sha256, sha256_hex};
use serde::Serialize;

const GOAL: &str = "XR60-dspark-native-mlx";
const MODE: &str = "native_dspark_fixed_block_matrix";
const DEFAULT_OUT_DIR: &str = "benchmarks/out/XR60-dspark-native-mlx";
const DEFAULT_MODEL: &str = "artifacts/models/gemma-4-12B-it-4bit";
const DEFAULT_DRAFT: &str = "artifacts/drafts/dspark-gemma4-12b-block7";
const EXPECTED_DSPARK_REVISION: &str = "2fa72e765eec2965fc4d86a8663ce6769eba6218";
const EXPECTED_TARGET_LAYERS: &[u32] = &[5, 17, 29, 41, 46];

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
    let mut blockers = startup_blockers(&args, &draft_config);
    if blockers.is_empty() {
        blockers.push(
            "XR60 fixed-prefix workload execution is not wired into this scaffold yet; invoke the native DSpark FFI draft path before making speed or exactness claims"
                .to_owned(),
        );
    }
    let run_id = run_id();
    let records = blocked_records(&args, &run_id, &blockers);
    let decision = "blocked";
    let summary = Summary {
        schema_version: 1,
        goal: GOAL,
        mode: MODE,
        status: "blocked",
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
        block_sizes: args.block_sizes.clone(),
        max_new_tokens: args.max_new_tokens,
        target_layer_ids: EXPECTED_TARGET_LAYERS.to_vec(),
        records: records.clone(),
        blockers: blockers.clone(),
        measurement_notes: vec![
            "This XR60 harness slice is fail-closed scaffolding. It emits the required artifact shape and blocks until startup dependencies and workload execution are available.",
            "No DSpark benchmark speed or exactness claim is made by blocked records.",
            "The draft artifact is expected to be deepseek-ai/dspark_gemma4_12b_block7 at revision 2fa72e765eec2965fc4d86a8663ce6769eba6218.",
        ],
    };

    write_jsonl(&records_path, &records)?;
    fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)?;
    fs::write(&report_path, render_report(&summary))?;
    fs::write(&blockers_path, render_blockers(&summary))?;
    fs::write(&decision_path, format!("{decision}\n"))?;

    println!("XR60 DSpark fixed-block matrix: {decision}");
    println!("records: {}", records_path.display());
    println!("summary: {}", summary_path.display());
    println!("report: {}", report_path.display());
    println!("blockers: {}", blockers_path.display());
    println!("decision: {}", decision_path.display());
    Ok(())
}

#[derive(Debug)]
struct Args {
    out_dir: PathBuf,
    model_path: PathBuf,
    draft_path: PathBuf,
    block_sizes: Vec<usize>,
    max_new_tokens: usize,
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
        let mut max_new_tokens = 32;

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
                "--max-new-tokens" => {
                    max_new_tokens = required_value(&mut args, "--max-new-tokens")?
                        .parse()
                        .map_err(|_| "--max-new-tokens must be an integer")?;
                }
                "-h" | "--help" => {
                    println!(
                        "usage: cargo run -p gemma4d-bench --example dspark_fixed_block_matrix -- [--out-dir PATH] [--model-path PATH] [--draft-path PATH] [--block-sizes 1,2,4,7] [--max-new-tokens N]"
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
        if max_new_tokens == 0 {
            return Err("--max-new-tokens must be greater than zero".into());
        }
        Ok(Self {
            out_dir,
            model_path,
            draft_path,
            block_sizes,
            max_new_tokens,
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
    block_sizes: Vec<usize>,
    max_new_tokens: usize,
    target_layer_ids: Vec<u32>,
    records: Vec<Record>,
    blockers: Vec<String>,
    measurement_notes: Vec<&'static str>,
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
    workload_id: &'static str,
    context_tokens: usize,
    max_new_tokens: usize,
    scheduler: &'static str,
    scheduled_len: usize,
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
    auto_disable_reason: String,
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
            workload_id: "startup",
            context_tokens: 0,
            max_new_tokens: args.max_new_tokens,
            scheduler: "fixed",
            scheduled_len: *scheduled_len,
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
            auto_disable_reason: reason.clone(),
        })
        .collect()
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
        "- Draft path: `{}`\n",
        escape_md(&summary.draft_identity.path)
    ));
    out.push_str(&format!(
        "- Expected DSpark revision: `{EXPECTED_DSPARK_REVISION}`\n\n"
    ));
    out.push_str("## What changed\n\n");
    out.push_str(
        "This artifact was generated by the XR60 fixed-prefix harness scaffold. The native DSpark FFI draft path is present, but this scaffold did not produce workload records, so no speed claim is made.\n\n",
    );
    out.push_str("## Correctness results\n\n");
    out.push_str("No DSpark exactness records were measured because startup blockers fired before native drafting.\n\n");
    out.push_str("## Benchmark summary\n\n");
    out.push_str("| workload | scheduler | block/max | exact | decode tok/s | speedup | acceptance | accepted/verify | draft ms | verify ms | peak GB | active KV bytes |\n");
    out.push_str("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for record in &summary.records {
        out.push_str(&format!(
            "| {} | {} | {} | {} | n/a | n/a | {:.3} | {:.3} | n/a | n/a | n/a | n/a |\n",
            record.workload_id,
            record.scheduler,
            record.scheduled_len,
            record.exact,
            record.acceptance_rate,
            record.accepted_tokens_per_verify
        ));
    }
    out.push_str("\n## Hidden tap parity\n\n");
    out.push_str("Not measured. Required target tap ids are `[5, 17, 29, 41, 46]`.\n\n");
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
