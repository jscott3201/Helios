use std::{
    collections::BTreeSet,
    env, fs,
    num::NonZeroU64,
    path::PathBuf,
    time::{Duration, Instant},
};

use gemma4d_kv::{
    CacheMode, CompressionQualityResult, CompressionWorkload, KvNamespace,
    evaluate_compression_fixture, fixture_block_with_mode,
};
use serde::Serialize;

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    milestone: &'static str,
    status: &'static str,
    bf16_fallback_default: bool,
    namespace_hashes_unique_by_mode: bool,
    block_ids_unique_by_mode: bool,
    modes: Vec<&'static str>,
    workloads: Vec<&'static str>,
    sequence_lengths: Vec<u64>,
    cases: Vec<CaseReport>,
    summary: Summary,
    planar_iso: PlanarIsoReport,
}

#[derive(Debug, Serialize)]
struct CaseReport {
    mode: CacheMode,
    workload: CompressionWorkload,
    sequence_len: u64,
    logit_cosine: f64,
    greedy_agreement: bool,
    quality_gate_passed: bool,
    bf16_bytes: u64,
    compressed_bytes: u64,
    memory_delta_bytes: i64,
    memory_reduction: f64,
    eval_us: u128,
}

#[derive(Debug, Serialize)]
struct Summary {
    q8_min_logit_cosine: f64,
    q4_min_logit_cosine: f64,
    q8_all_greedy_agree: bool,
    q4_all_greedy_agree: bool,
    q8_average_memory_reduction: f64,
    q4_average_memory_reduction: f64,
    max_eval_us: u128,
}

#[derive(Debug, Serialize)]
struct PlanarIsoReport {
    feature_enabled: bool,
    accepted_by_default: bool,
    status: &'static str,
    candidates: Vec<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_path = parse_out_path()?;
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let modes = [CacheMode::MlxAffineQ8, CacheMode::MlxAffineQ4];
    let workloads = [
        CompressionWorkload::SimpleChat,
        CompressionWorkload::JsonToolFixture,
        CompressionWorkload::CodeReview,
    ];
    let sequence_lengths = [16 * 1024, 32 * 1024, 64 * 1024];
    let mut cases = Vec::new();

    for sequence_len in sequence_lengths {
        for workload in workloads {
            for mode in modes {
                let started = Instant::now();
                let result = evaluate_compression_fixture(sequence_len, workload, mode);
                cases.push(CaseReport::from_result(result, started.elapsed()));
            }
        }
    }

    let bf16_fallback_default = KvNamespace::fixture(16 * 1024).cache_mode == CacheMode::Bf16;
    let (namespace_hashes_unique_by_mode, block_ids_unique_by_mode) =
        namespace_semantics_are_mode_scoped()?;
    let summary = summarize(&cases);
    let planar_iso = planar_iso_report();
    let passed = bf16_fallback_default
        && namespace_hashes_unique_by_mode
        && block_ids_unique_by_mode
        && cases
            .iter()
            .all(|case| case.greedy_agreement && case.memory_delta_bytes < 0);

    let report = Report {
        schema_version: 1,
        milestone: "M09",
        status: if passed { "passed" } else { "failed" },
        bf16_fallback_default,
        namespace_hashes_unique_by_mode,
        block_ids_unique_by_mode,
        modes: modes.iter().map(|mode| mode.label()).collect(),
        workloads: workloads.iter().map(|workload| workload.label()).collect(),
        sequence_lengths: sequence_lengths.to_vec(),
        cases,
        summary,
        planar_iso,
    };

    fs::write(&out_path, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "M09 compression eval: {} cases {}",
        report.cases.len(),
        report.status
    );
    println!("evidence: {}", out_path.display());
    if passed {
        Ok(())
    } else {
        Err("M09 compression eval failed".into())
    }
}

impl CaseReport {
    fn from_result(result: CompressionQualityResult, elapsed: Duration) -> Self {
        Self {
            mode: result.mode,
            workload: result.workload,
            sequence_len: result.sequence_len,
            logit_cosine: result.logit_cosine,
            greedy_agreement: result.greedy_agreement,
            quality_gate_passed: result.accepted,
            bf16_bytes: result.bf16_bytes,
            compressed_bytes: result.compressed_bytes,
            memory_delta_bytes: result.memory_delta_bytes,
            memory_reduction: result.memory_reduction,
            eval_us: elapsed.as_micros(),
        }
    }
}

fn namespace_semantics_are_mode_scoped() -> Result<(bool, bool), Box<dyn std::error::Error>> {
    let block_size = NonZeroU64::new(16 * 1024).expect("non-zero");
    let blocks = [
        fixture_block_with_mode(16 * 1024, block_size, CacheMode::Bf16)?,
        fixture_block_with_mode(16 * 1024, block_size, CacheMode::MlxAffineQ8)?,
        fixture_block_with_mode(16 * 1024, block_size, CacheMode::MlxAffineQ4)?,
    ];
    let namespace_hashes = blocks
        .iter()
        .map(|block| block.key.namespace_hash.0.clone())
        .collect::<BTreeSet<_>>();
    let block_ids = blocks
        .iter()
        .map(|block| block.key.block_id.0.clone())
        .collect::<BTreeSet<_>>();
    Ok((
        namespace_hashes.len() == blocks.len(),
        block_ids.len() == blocks.len(),
    ))
}

fn summarize(cases: &[CaseReport]) -> Summary {
    Summary {
        q8_min_logit_cosine: min_cosine(cases, CacheMode::MlxAffineQ8),
        q4_min_logit_cosine: min_cosine(cases, CacheMode::MlxAffineQ4),
        q8_all_greedy_agree: all_greedy_agree(cases, CacheMode::MlxAffineQ8),
        q4_all_greedy_agree: all_greedy_agree(cases, CacheMode::MlxAffineQ4),
        q8_average_memory_reduction: average_memory_reduction(cases, CacheMode::MlxAffineQ8),
        q4_average_memory_reduction: average_memory_reduction(cases, CacheMode::MlxAffineQ4),
        max_eval_us: cases.iter().map(|case| case.eval_us).max().unwrap_or(0),
    }
}

fn min_cosine(cases: &[CaseReport], mode: CacheMode) -> f64 {
    cases
        .iter()
        .filter(|case| case.mode == mode)
        .map(|case| case.logit_cosine)
        .fold(f64::INFINITY, f64::min)
}

fn all_greedy_agree(cases: &[CaseReport], mode: CacheMode) -> bool {
    cases
        .iter()
        .filter(|case| case.mode == mode)
        .all(|case| case.greedy_agreement)
}

fn average_memory_reduction(cases: &[CaseReport], mode: CacheMode) -> f64 {
    let matching = cases
        .iter()
        .filter(|case| case.mode == mode)
        .collect::<Vec<_>>();
    if matching.is_empty() {
        return 0.0;
    }
    matching
        .iter()
        .map(|case| case.memory_reduction)
        .sum::<f64>()
        / matching.len() as f64
}

#[cfg(feature = "planar-iso-experiments")]
fn planar_iso_report() -> PlanarIsoReport {
    use gemma4d_kv::ExperimentalCompressionPlan;

    let candidates = ExperimentalCompressionPlan::candidates();
    PlanarIsoReport {
        feature_enabled: true,
        accepted_by_default: candidates.iter().any(|candidate| candidate.accepted),
        status: "feature_enabled_but_experimental",
        candidates: candidates
            .into_iter()
            .map(|candidate| format!("{:?}: {}", candidate.mode, candidate.reason))
            .collect(),
    }
}

#[cfg(not(feature = "planar-iso-experiments"))]
fn planar_iso_report() -> PlanarIsoReport {
    PlanarIsoReport {
        feature_enabled: false,
        accepted_by_default: false,
        status: "feature_disabled_default",
        candidates: vec![
            "planar4".to_owned(),
            "planar3".to_owned(),
            "iso4".to_owned(),
            "iso3".to_owned(),
        ],
    }
}

fn parse_out_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let mut out = None;
    while let Some(arg) = args.next() {
        if arg == "--out" {
            out = args.next().map(PathBuf::from);
        }
    }
    out.ok_or_else(|| "usage: m09_compression_eval --out <path>".into())
}
