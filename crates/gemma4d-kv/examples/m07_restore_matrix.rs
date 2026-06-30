use std::{env, fs, num::NonZeroU64, path::PathBuf, time::Instant};

use gemma4d_kv::{
    CacheAccountingSnapshot, Error, RamPrefixCache, fixture_block, fresh_prefill_fixture,
};
use serde::Serialize;

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    milestone: &'static str,
    status: &'static str,
    cache_mode: &'static str,
    block_size_tokens: u64,
    no_ssd_dependency: bool,
    cases: Vec<CaseReport>,
    accounting: CacheAccountingSnapshot,
}

#[derive(Debug, Serialize)]
struct CaseReport {
    name: String,
    sequence_len: u64,
    exact: bool,
    fresh_greedy_token: u32,
    restored_greedy_token: u32,
    fresh_greedy_logit: f32,
    restored_greedy_logit: f32,
    cold_prefill_ttft_ms: f64,
    warm_ram_restore_ttft_ms: f64,
    rejected_wrong_model: bool,
    rejected_wrong_template: bool,
    rejected_wrong_prompt_hash: bool,
    restored_bytes: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_path = parse_out_path()?;
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let block_size = NonZeroU64::new(16 * 1024).expect("non-zero");
    let mut cache =
        RamPrefixCache::new(NonZeroU64::new(64 * 1024 * 1024 * 1024).expect("non-zero"));
    let mut cases = Vec::new();

    for sequence_len in [1024, 4096, 8192, 16384] {
        let cold_start = Instant::now();
        let block = fixture_block(sequence_len, block_size)?;
        let key = block.key.clone();
        let namespace = block.namespace.clone();
        let fresh = fresh_prefill_fixture(sequence_len);
        let cold_ttft_ms = elapsed_ms(cold_start.elapsed());
        cache.insert(block)?;

        let warm_start = Instant::now();
        let restored = cache.restore(&key, &namespace)?;
        let warm_ttft_ms = elapsed_ms(warm_start.elapsed());
        let mut wrong_model = namespace.clone();
        wrong_model.model_id = "wrong-model".to_owned();
        let mut wrong_template = namespace.clone();
        wrong_template.chat_template_sha256 = "wrong-template".to_owned();
        let mut wrong_prompt = namespace.clone();
        wrong_prompt.prompt_token_hash = "wrong-prompt".to_owned();

        let rejected_wrong_model = matches!(
            cache.restore(&key, &wrong_model),
            Err(Error::NamespaceMismatch { .. })
        );
        let rejected_wrong_template = matches!(
            cache.restore(&key, &wrong_template),
            Err(Error::NamespaceMismatch { .. })
        );
        let rejected_wrong_prompt_hash = matches!(
            cache.restore(&key, &wrong_prompt),
            Err(Error::NamespaceMismatch { .. })
        );

        cases.push(CaseReport {
            name: format!("restore_vs_fresh_{sequence_len}"),
            sequence_len,
            exact: restored.observation == fresh,
            fresh_greedy_token: fresh.greedy_token,
            restored_greedy_token: restored.observation.greedy_token,
            fresh_greedy_logit: fresh.greedy_logit(),
            restored_greedy_logit: restored.observation.greedy_logit(),
            cold_prefill_ttft_ms: cold_ttft_ms,
            warm_ram_restore_ttft_ms: warm_ttft_ms,
            rejected_wrong_model,
            rejected_wrong_template,
            rejected_wrong_prompt_hash,
            restored_bytes: restored.byte_len,
        });
    }

    let passed = cases.iter().all(|case| {
        case.exact
            && case.rejected_wrong_model
            && case.rejected_wrong_template
            && case.rejected_wrong_prompt_hash
    });
    let report = Report {
        schema_version: 1,
        milestone: "M07",
        status: if passed { "passed" } else { "failed" },
        cache_mode: "ram_prefix_bf16",
        block_size_tokens: block_size.get(),
        no_ssd_dependency: true,
        cases,
        accounting: cache.accounting(),
    };

    let json = serde_json::to_string_pretty(&report)?;
    fs::write(&out_path, json)?;
    println!(
        "M07 KV restore matrix: {} cases {}",
        report.cases.len(),
        report.status
    );
    println!("evidence: {}", out_path.display());
    if passed {
        Ok(())
    } else {
        Err("M07 restore matrix failed".into())
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
    out.ok_or_else(|| "usage: m07_restore_matrix --out <path>".into())
}

fn elapsed_ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}
