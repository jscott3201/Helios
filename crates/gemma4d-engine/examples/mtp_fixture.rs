use std::{
    env, fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use gemma4d_engine::{
    Drafter, GreedyTarget, MtpConfig, MtpError, MtpMetrics, TargetStep, non_mtp_greedy,
    speculative_greedy,
};

fn main() {
    match run() {
        Ok(report) => {
            println!("{}", report.summary);
            if let Some(path) = report.path {
                println!("evidence: {}", path.display());
            }
        }
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}

struct ReportResult {
    summary: String,
    path: Option<PathBuf>,
}

fn run() -> Result<ReportResult, String> {
    let mut out = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--out requires a file path".to_owned())?;
                out = Some(PathBuf::from(value));
            }
            "-h" | "--help" => {
                return Ok(ReportResult {
                    summary: usage(),
                    path: None,
                });
            }
            other => return Err(format!("unknown option '{other}'\n{}", usage())),
        }
    }

    let cases = vec![
        exact_case(
            "block_size_1_exact",
            1,
            vec![236772, 236772, 236772, 236772],
            PerfectDrafter::new(vec![236772, 236772, 236772, 236772]),
            MtpConfig::block_size(1),
        )?,
        exact_case(
            "block_size_2_rollback_exact",
            2,
            vec![10, 11, 12, 13],
            BlockDrafter::new(vec![vec![10, 99], vec![12, 13]]),
            MtpConfig::block_size(2),
        )?,
        exact_case(
            "block_size_2_auto_disable",
            2,
            vec![10, 11, 12, 13],
            BlockDrafter::new(vec![vec![99, 98]]),
            MtpConfig::block_size(2),
        )?,
        exact_case(
            "adapter_active_auto_disable",
            1,
            vec![20, 21],
            PerfectDrafter::new(vec![20, 21]),
            MtpConfig {
                adapter_id: Some("rust-expert".to_owned()),
                ..MtpConfig::block_size(1)
            },
        )?,
        exact_case(
            "compressed_active_kv_auto_disable",
            1,
            vec![30, 31],
            PerfectDrafter::new(vec![30, 31]),
            MtpConfig {
                active_kv_compressed: true,
                ..MtpConfig::block_size(1)
            },
        )?,
    ];

    let body = render_report(&cases);
    if let Some(path) = &out {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        fs::write(path, &body).map_err(|error| error.to_string())?;
    }

    Ok(ReportResult {
        summary: format!("M06 MTP fixture report: {} cases passed", cases.len()),
        path: out,
    })
}

fn usage() -> String {
    "usage: cargo run -p gemma4d-engine --example mtp_fixture -- [--out PATH]".to_owned()
}

fn exact_case<D: Drafter>(
    name: &str,
    block_size: usize,
    expected: Vec<i32>,
    mut drafter: D,
    config: MtpConfig,
) -> Result<FixtureCase, String> {
    let prompt = vec![9259];
    let mut baseline_target = ScriptedTarget::new(expected.clone());
    let baseline = non_mtp_greedy(&mut baseline_target, &prompt, expected.len())
        .map_err(|error| error.to_string())?;

    let mut target = ScriptedTarget::new(expected);
    let run = speculative_greedy(&mut target, &mut drafter, &prompt, baseline.len(), &config)
        .map_err(|error| error.to_string())?;

    if run.generated_tokens != baseline {
        return Err(format!(
            "{name} failed exactness: baseline={baseline:?} mtp={:?}",
            run.generated_tokens
        ));
    }

    Ok(FixtureCase {
        name: name.to_owned(),
        block_size,
        baseline_tokens: baseline,
        mtp_tokens: run.generated_tokens,
        metrics: run.metrics,
    })
}

#[derive(Debug, Clone)]
struct FixtureCase {
    name: String,
    block_size: usize,
    baseline_tokens: Vec<i32>,
    mtp_tokens: Vec<i32>,
    metrics: MtpMetrics,
}

fn render_report(cases: &[FixtureCase]) -> String {
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str("  \"schema_version\": 1,\n");
    json.push_str("  \"milestone\": \"M06\",\n");
    json.push_str(&format!("  \"timestamp_unix\": {},\n", unix_now()));
    json.push_str("  \"status\": \"passed\",\n");
    json.push_str("  \"adapter_policy\": \"MTP auto-disables when adapter_id != none\",\n");
    json.push_str("  \"active_kv_policy\": \"MTP requires bf16 active KV in M06; compressed active KV auto-disables\",\n");
    json.push_str("  \"cases\": [\n");
    for (index, case) in cases.iter().enumerate() {
        if index != 0 {
            json.push_str(",\n");
        }
        json.push_str(&render_case(case));
    }
    json.push_str("\n  ]\n");
    json.push_str("}\n");
    json
}

fn render_case(case: &FixtureCase) -> String {
    format!(
        "    {{\n      \"name\": \"{}\",\n      \"block_size\": {},\n      \"exact\": {},\n      \"baseline_tokens\": {},\n      \"mtp_tokens\": {},\n      \"metrics\": {}\n    }}",
        escape_json(&case.name),
        case.block_size,
        case.baseline_tokens == case.mtp_tokens,
        int_array_json(&case.baseline_tokens),
        int_array_json(&case.mtp_tokens),
        metrics_json(&case.metrics)
    )
}

fn metrics_json(metrics: &MtpMetrics) -> String {
    format!(
        "{{\"draft_block_size\":{},\"attempted_draft_tokens\":{},\"accepted_draft_tokens\":{},\"acceptance_rate\":{:.6},\"accepted_tokens_per_verify\":{:.6},\"target_verify_passes\":{},\"decode_tokens_per_second\":{:.6},\"peak_memory_gb\":{:.6},\"rollback_count\":{},\"auto_disabled\":{},\"auto_disable_reason\":{}}}",
        metrics.draft_block_size,
        metrics.attempted_draft_tokens,
        metrics.accepted_draft_tokens,
        metrics.acceptance_rate,
        metrics.accepted_tokens_per_verify,
        metrics.target_verify_passes,
        metrics.decode_tokens_per_second,
        metrics.peak_memory_gb,
        metrics.rollback_count,
        metrics.auto_disabled,
        option_string_json(metrics.auto_disable_reason.as_deref())
    )
}

fn int_array_json(tokens: &[i32]) -> String {
    let mut out = String::from("[");
    for (index, token) in tokens.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push_str(&token.to_string());
    }
    out.push(']');
    out
}

fn option_string_json(value: Option<&str>) -> String {
    value
        .map(|value| format!("\"{}\"", escape_json(value)))
        .unwrap_or_else(|| "null".to_owned())
}

fn escape_json(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[derive(Debug, Clone)]
struct ScriptedTarget {
    expected: Vec<i32>,
}

impl ScriptedTarget {
    fn new(expected: Vec<i32>) -> Self {
        Self { expected }
    }
}

impl GreedyTarget for ScriptedTarget {
    fn next_greedy(
        &mut self,
        _prompt_tokens: &[i32],
        accepted_tokens: &[i32],
    ) -> Result<TargetStep, MtpError> {
        self.expected
            .get(accepted_tokens.len())
            .copied()
            .map(TargetStep::new)
            .ok_or_else(|| {
                MtpError::Target(format!(
                    "fixture target exhausted at generated length {}",
                    accepted_tokens.len()
                ))
            })
    }
}

#[derive(Debug, Clone)]
struct PerfectDrafter {
    expected: Vec<i32>,
}

impl PerfectDrafter {
    fn new(expected: Vec<i32>) -> Self {
        Self { expected }
    }
}

impl Drafter for PerfectDrafter {
    fn draft(
        &mut self,
        _prompt_tokens: &[i32],
        accepted_tokens: &[i32],
        block_size: usize,
    ) -> Result<Vec<i32>, MtpError> {
        Ok(self
            .expected
            .iter()
            .skip(accepted_tokens.len())
            .take(block_size)
            .copied()
            .collect())
    }
}

#[derive(Debug, Clone)]
struct BlockDrafter {
    blocks: Vec<Vec<i32>>,
    next: usize,
}

impl BlockDrafter {
    fn new(blocks: Vec<Vec<i32>>) -> Self {
        Self { blocks, next: 0 }
    }
}

impl Drafter for BlockDrafter {
    fn draft(
        &mut self,
        _prompt_tokens: &[i32],
        _accepted_tokens: &[i32],
        _block_size: usize,
    ) -> Result<Vec<i32>, MtpError> {
        let block = self.blocks.get(self.next).cloned().ok_or_else(|| {
            MtpError::Drafter(format!("fixture drafter exhausted at block {}", self.next))
        })?;
        self.next += 1;
        Ok(block)
    }
}
