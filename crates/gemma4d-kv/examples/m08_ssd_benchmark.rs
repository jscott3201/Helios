use std::{
    env, fs,
    num::NonZeroU64,
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use gemma4d_kv::{Error, SsdPrefixCache, SsdRestorePhase, fixture_block, fresh_prefill_fixture};
use serde::Serialize;

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    milestone: &'static str,
    status: &'static str,
    cache_mode: &'static str,
    block_size_tokens: u64,
    no_mid_decode_ssd_fetch: bool,
    cache_dir: String,
    cases: Vec<CaseReport>,
    accounting: gemma4d_kv::SsdCacheAccountingSnapshot,
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
    warm_ssd_restore_ttft_ms: f64,
    write_bytes: u64,
    read_bytes: u64,
    rejected_wrong_namespace: bool,
    rejected_corrupt_block: bool,
    rejected_mid_decode_fetch: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;
    if let Some(parent) = args.out_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir_all(&args.cache_dir)?;

    let block_size = NonZeroU64::new(16 * 1024).expect("non-zero");
    let mut cache = SsdPrefixCache::open(
        &args.cache_dir,
        NonZeroU64::new(64 * 1024 * 1024).expect("non-zero"),
    )?;
    let mut cases = Vec::new();

    for sequence_len in [1024, 4096, 8192, 16384] {
        let cold_start = Instant::now();
        let fresh = fresh_prefill_fixture(sequence_len);
        let block = fixture_block(sequence_len, block_size)?;
        let cold_ttft_ms = elapsed_ms(cold_start.elapsed());

        let key = block.key.clone();
        let namespace = block.namespace.clone();
        let before_write = cache.accounting();
        let entry = cache.write_block(&block)?;
        let after_write = cache.accounting();

        let warm_start = Instant::now();
        let restored = cache.restore_before_prefill(&key, &namespace)?;
        let warm_ttft_ms = elapsed_ms(warm_start.elapsed());
        let after_restore = cache.accounting();

        let mut wrong_namespace = namespace.clone();
        wrong_namespace.model_id = "wrong-model".to_owned();
        let rejected_wrong_namespace = matches!(
            cache.restore_before_prefill(&key, &wrong_namespace),
            Err(Error::NamespaceMismatch { .. })
        );

        corrupt_manifest_checksum(&cache.entry_path(&entry))?;
        let rejected_corrupt_block = matches!(
            cache.restore_before_prefill(&key, &namespace),
            Err(Error::ChecksumMismatch { .. })
        );

        let rejected_mid_decode_fetch = matches!(
            cache.restore_for_phase(&key, &namespace, SsdRestorePhase::MidDecode),
            Err(Error::InvalidBlock(_))
        );

        cases.push(CaseReport {
            name: format!("ssd_restore_vs_fresh_{sequence_len}"),
            sequence_len,
            exact: restored.observation == fresh,
            fresh_greedy_token: fresh.greedy_token,
            restored_greedy_token: restored.observation.greedy_token,
            fresh_greedy_logit: fresh.greedy_logit(),
            restored_greedy_logit: restored.observation.greedy_logit(),
            cold_prefill_ttft_ms: cold_ttft_ms,
            warm_ssd_restore_ttft_ms: warm_ttft_ms,
            write_bytes: after_write
                .bytes_written
                .saturating_sub(before_write.bytes_written),
            read_bytes: after_restore
                .bytes_read
                .saturating_sub(after_write.bytes_read),
            rejected_wrong_namespace,
            rejected_corrupt_block,
            rejected_mid_decode_fetch,
        });
    }

    let accounting = cache.accounting();
    let passed = cases.iter().all(|case| {
        case.exact
            && case.rejected_wrong_namespace
            && case.rejected_corrupt_block
            && case.rejected_mid_decode_fetch
    }) && accounting.mid_decode_fetches == 0;
    let report = Report {
        schema_version: 1,
        milestone: "M08",
        status: if passed { "passed" } else { "failed" },
        cache_mode: "ssd_cold_prefix_bf16",
        block_size_tokens: block_size.get(),
        no_mid_decode_ssd_fetch: accounting.mid_decode_fetches == 0,
        cache_dir: args.cache_dir.display().to_string(),
        cases,
        accounting,
    };

    let json = serde_json::to_string_pretty(&report)?;
    fs::write(&args.out_path, json)?;
    println!(
        "M08 SSD restore benchmark: {} cases {}",
        report.cases.len(),
        report.status
    );
    println!("evidence: {}", args.out_path.display());
    println!("cache dir: {}", args.cache_dir.display());
    if passed {
        Ok(())
    } else {
        Err("M08 SSD restore benchmark failed".into())
    }
}

#[derive(Debug)]
struct Args {
    out_path: PathBuf,
    cache_dir: PathBuf,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut args = env::args().skip(1);
        let mut out_path = None;
        let mut cache_dir = None;
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--out" => out_path = args.next().map(PathBuf::from),
                "--cache-dir" => cache_dir = args.next().map(PathBuf::from),
                _ => {}
            }
        }
        let out_path =
            out_path.ok_or("usage: m08_ssd_benchmark --out <path> [--cache-dir <path>]")?;
        let cache_dir = cache_dir.unwrap_or_else(|| default_cache_dir(&out_path));
        Ok(Self {
            out_path,
            cache_dir,
        })
    }
}

fn default_cache_dir(out_path: &Path) -> PathBuf {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    out_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("ssd-cache-{}-{suffix}", std::process::id()))
}

fn elapsed_ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn corrupt_manifest_checksum(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut value: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
    value["manifest"]["block_checksum"] = serde_json::Value::String("corrupted".to_owned());
    fs::write(path, serde_json::to_vec_pretty(&value)?)?;
    Ok(())
}
