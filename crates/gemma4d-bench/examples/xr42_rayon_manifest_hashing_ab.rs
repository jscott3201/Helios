use std::{
    collections::BTreeMap,
    env, fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use gemma4d_tokenizer::{file_sha256, sha256_hex};
use rayon::{ThreadPoolBuilder, prelude::*};
use serde::Serialize;

const GOAL: &str = "XR42-rayon-manifest-hashing-ab";
const DEFAULT_OUT_DIR: &str = "benchmarks/out/XR42-rayon-manifest-hashing-ab";
const DEFAULT_MODEL_PATH: &str = "artifacts/models/gemma-4-12B-it-4bit";
const DEFAULT_DRAFTER_PATH: &str = "artifacts/models/gemma-4-12B-it-qat-assistant-4bit";
const DETERMINISTIC_SEED: u64 = 20_260_701;

type DynError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Options {
    artifacts: Vec<PathBuf>,
    out_dir: PathBuf,
    trials: usize,
    thread_counts: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Variant {
    name: String,
    rayon_threads: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SafetensorsFile {
    absolute_path: PathBuf,
    relative_path: String,
    bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct InventoryEntry {
    path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Clone, Serialize)]
struct Record {
    schema_version: u32,
    goal: String,
    run_id: String,
    git_sha: String,
    git_status_short: String,
    command: String,
    deterministic_seed: u64,
    seed_usage: String,
    token_lengths: String,
    artifact_label: String,
    artifact_path: String,
    variant: String,
    rayon_threads: Option<usize>,
    trial_index: usize,
    file_count: usize,
    total_bytes: u64,
    inventory_sha256: String,
    baseline_inventory_sha256: String,
    checksum_match: bool,
    elapsed_ms: f64,
    bytes_per_second: f64,
    status: String,
    blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct Summary {
    schema_version: u32,
    goal: String,
    decision: String,
    run_id: String,
    git_sha: String,
    git_status_short: String,
    command: String,
    deterministic_seed: u64,
    seed_usage: String,
    token_lengths: String,
    artifact_count: usize,
    variants: Vec<String>,
    generated_files: Vec<String>,
    blockers: Vec<String>,
    comparisons: Vec<Comparison>,
}

#[derive(Debug, Clone, Serialize)]
struct Comparison {
    artifact_label: String,
    artifact_path: String,
    variant: String,
    rayon_threads: Option<usize>,
    trials: usize,
    file_count: usize,
    total_bytes: u64,
    inventory_sha256: String,
    checksum_matches_all: bool,
    baseline_p50_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    best_ms: f64,
    worst_ms: f64,
    p50_improvement_pct: f64,
    bytes_per_second_p50: f64,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), DynError> {
    let options = parse_options(env::args().skip(1))?;
    fs::create_dir_all(&options.out_dir)?;

    let run_id = format!("xr42-{}", unix_now_millis());
    let git_sha = command_stdout("git", &["rev-parse", "HEAD"]).unwrap_or("unknown".to_owned());
    let git_status_short =
        command_stdout("git", &["status", "--short"]).unwrap_or("unknown".to_owned());
    let command = command_display();
    let variants = variants(&options);

    let mut records = Vec::new();
    let mut blockers = Vec::new();

    for artifact_path in &options.artifacts {
        let artifact_label = artifact_label(artifact_path);
        let files = collect_safetensors(artifact_path)?;
        if files.is_empty() {
            blockers.push(format!(
                "{} has no safetensors files",
                artifact_path.display()
            ));
            continue;
        }

        let baseline_variant = Variant {
            name: "sequential".to_owned(),
            rayon_threads: None,
        };
        let mut baseline_records = run_trials(
            &baseline_variant,
            artifact_path,
            &artifact_label,
            &files,
            "",
            &options,
            &run_id,
            &git_sha,
            &git_status_short,
            &command,
        )?;
        let Some(baseline_hash) = baseline_records
            .first()
            .map(|record| record.inventory_sha256.clone())
        else {
            blockers.push(format!(
                "{} did not produce a sequential baseline",
                artifact_path.display()
            ));
            continue;
        };
        for record in &mut baseline_records {
            record.baseline_inventory_sha256 = baseline_hash.clone();
            record.checksum_match = record.inventory_sha256 == baseline_hash;
        }
        records.extend(baseline_records);

        for variant in variants
            .iter()
            .filter(|variant| variant.name != "sequential")
        {
            records.extend(run_trials(
                variant,
                artifact_path,
                &artifact_label,
                &files,
                &baseline_hash,
                &options,
                &run_id,
                &git_sha,
                &git_status_short,
                &command,
            )?);
        }
    }

    blockers.extend(
        records
            .iter()
            .filter(|record| !record.checksum_match)
            .map(|record| {
                format!(
                    "{} {} trial {} hash mismatch: baseline={} candidate={}",
                    record.artifact_label,
                    record.variant,
                    record.trial_index,
                    record.baseline_inventory_sha256,
                    record.inventory_sha256
                )
            }),
    );

    let generated_files = vec![
        options.out_dir.join("records.jsonl").display().to_string(),
        options.out_dir.join("summary.json").display().to_string(),
        options.out_dir.join("report.md").display().to_string(),
        options.out_dir.join("blockers.md").display().to_string(),
        options.out_dir.join("decision.md").display().to_string(),
    ];
    let comparisons = build_comparisons(&records);
    let decision = decide(&blockers, &comparisons);
    let summary = Summary {
        schema_version: 1,
        goal: GOAL.to_owned(),
        decision: decision.clone(),
        run_id,
        git_sha,
        git_status_short,
        command,
        deterministic_seed: DETERMINISTIC_SEED,
        seed_usage: "no randomness; seed recorded for benchmark-ledger consistency".to_owned(),
        token_lengths: "not_applicable:file hashing only; no tokenizer/model execution".to_owned(),
        artifact_count: options.artifacts.len(),
        variants: variants
            .iter()
            .map(|variant| variant.name.clone())
            .collect(),
        generated_files: generated_files.clone(),
        blockers,
        comparisons,
    };

    write_records_jsonl(&options.out_dir.join("records.jsonl"), &records)?;
    fs::write(
        options.out_dir.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    fs::write(options.out_dir.join("report.md"), render_report(&summary))?;
    fs::write(
        options.out_dir.join("blockers.md"),
        render_blockers(&summary),
    )?;
    fs::write(
        options.out_dir.join("decision.md"),
        render_decision(&summary),
    )?;

    println!(
        "wrote XR42 artifacts to {} with decision {}",
        options.out_dir.display(),
        summary.decision
    );
    Ok(())
}

fn parse_options<I>(args: I) -> Result<Options, DynError>
where
    I: IntoIterator<Item = String>,
{
    let mut artifacts = Vec::new();
    let mut out_dir = PathBuf::from(DEFAULT_OUT_DIR);
    let mut trials = 3usize;
    let mut thread_counts = vec![1usize, 2, 4];
    let mut args = args.into_iter();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--artifact-path" => {
                artifacts.push(PathBuf::from(required_value(&mut args, "--artifact-path")?));
            }
            "--out-dir" => {
                out_dir = PathBuf::from(required_value(&mut args, "--out-dir")?);
            }
            "--trials" => {
                trials = required_value(&mut args, "--trials")?.parse()?;
            }
            "--thread-counts" => {
                thread_counts = required_value(&mut args, "--thread-counts")?
                    .split(',')
                    .map(|value| value.trim().parse::<usize>())
                    .collect::<Result<Vec<_>, _>>()?;
                thread_counts.sort_unstable();
                thread_counts.dedup();
            }
            "-h" | "--help" => return Err(error(usage())),
            other => return Err(error(format!("unknown option '{other}'\n{}", usage()))),
        }
    }

    if artifacts.is_empty() {
        artifacts.push(PathBuf::from(DEFAULT_MODEL_PATH));
        artifacts.push(PathBuf::from(DEFAULT_DRAFTER_PATH));
    }
    if trials == 0 {
        return Err(error("--trials must be > 0"));
    }
    if thread_counts.is_empty() || thread_counts.contains(&0) {
        return Err(error("--thread-counts must contain positive integers"));
    }

    Ok(Options {
        artifacts,
        out_dir,
        trials,
        thread_counts,
    })
}

fn variants(options: &Options) -> Vec<Variant> {
    let mut variants = vec![Variant {
        name: "sequential".to_owned(),
        rayon_threads: None,
    }];
    variants.extend(options.thread_counts.iter().map(|threads| Variant {
        name: format!("rayon_threads_{threads}"),
        rayon_threads: Some(*threads),
    }));
    variants
}

fn run_trials(
    variant: &Variant,
    artifact_path: &Path,
    artifact_label: &str,
    files: &[SafetensorsFile],
    baseline_inventory_sha256: &str,
    options: &Options,
    run_id: &str,
    git_sha: &str,
    git_status_short: &str,
    command: &str,
) -> Result<Vec<Record>, DynError> {
    let mut records = Vec::with_capacity(options.trials);
    for trial_index in 0..options.trials {
        let started = Instant::now();
        let entries = hash_entries(files, variant.rayon_threads)?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        let total_bytes = entries.iter().map(|entry| entry.bytes).sum();
        let inventory_sha256 = inventory_sha256(&entries);
        let baseline_hash = if baseline_inventory_sha256.is_empty() {
            inventory_sha256.clone()
        } else {
            baseline_inventory_sha256.to_owned()
        };
        let checksum_match = inventory_sha256 == baseline_hash;
        records.push(Record {
            schema_version: 1,
            goal: GOAL.to_owned(),
            run_id: run_id.to_owned(),
            git_sha: git_sha.to_owned(),
            git_status_short: git_status_short.to_owned(),
            command: command.to_owned(),
            deterministic_seed: DETERMINISTIC_SEED,
            seed_usage: "no randomness; seed recorded for benchmark-ledger consistency".to_owned(),
            token_lengths: "not_applicable:file hashing only; no tokenizer/model execution"
                .to_owned(),
            artifact_label: artifact_label.to_owned(),
            artifact_path: artifact_path.display().to_string(),
            variant: variant.name.clone(),
            rayon_threads: variant.rayon_threads,
            trial_index,
            file_count: entries.len(),
            total_bytes,
            inventory_sha256,
            baseline_inventory_sha256: baseline_hash,
            checksum_match,
            elapsed_ms,
            bytes_per_second: bytes_per_second(total_bytes, elapsed_ms),
            status: if checksum_match {
                "passed".to_owned()
            } else {
                "failed".to_owned()
            },
            blockers: if checksum_match {
                Vec::new()
            } else {
                vec!["inventory hash mismatch".to_owned()]
            },
        });
    }
    Ok(records)
}

fn hash_entries(
    files: &[SafetensorsFile],
    rayon_threads: Option<usize>,
) -> Result<Vec<InventoryEntry>, DynError> {
    let mut entries = match rayon_threads {
        Some(threads) => {
            let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
            pool.install(|| {
                files
                    .par_iter()
                    .map(hash_entry)
                    .collect::<Result<Vec<_>, _>>()
            })?
        }
        None => files
            .iter()
            .map(hash_entry)
            .collect::<Result<Vec<_>, _>>()?,
    };
    entries.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(entries)
}

fn hash_entry(file: &SafetensorsFile) -> Result<InventoryEntry, DynError> {
    Ok(InventoryEntry {
        path: file.relative_path.clone(),
        bytes: file.bytes,
        sha256: file_sha256(&file.absolute_path)?,
    })
}

fn collect_safetensors(root: &Path) -> Result<Vec<SafetensorsFile>, DynError> {
    let mut files = Vec::new();
    collect_safetensors_inner(root, root, &mut files)?;
    files.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    Ok(files)
}

fn collect_safetensors_inner(
    root: &Path,
    current: &Path,
    files: &mut Vec<SafetensorsFile>,
) -> Result<(), DynError> {
    if !current.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(current)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_safetensors_inner(root, &path, files)?;
            continue;
        }
        if path.extension().and_then(|extension| extension.to_str()) != Some("safetensors") {
            continue;
        }
        let relative = path.strip_prefix(root).unwrap_or(&path);
        let relative_path = relative.display().to_string();
        files.push(SafetensorsFile {
            absolute_path: path,
            relative_path,
            bytes: entry.metadata()?.len(),
        });
    }
    Ok(())
}

fn inventory_sha256(entries: &[InventoryEntry]) -> String {
    if entries.is_empty() {
        return "unavailable:no safetensors files found".to_owned();
    }
    let inventory_body = entries
        .iter()
        .map(|entry| format!("{}\t{}\t{}", entry.path, entry.bytes, entry.sha256))
        .collect::<Vec<_>>()
        .join("\n");
    sha256_hex(inventory_body.as_bytes())
}

fn build_comparisons(records: &[Record]) -> Vec<Comparison> {
    let mut groups: BTreeMap<(String, String), Vec<&Record>> = BTreeMap::new();
    for record in records {
        groups
            .entry((record.artifact_label.clone(), record.variant.clone()))
            .or_default()
            .push(record);
    }

    let mut baseline_p50_by_artifact = BTreeMap::new();
    for ((artifact_label, variant), records) in &groups {
        if variant == "sequential" {
            baseline_p50_by_artifact.insert(artifact_label.clone(), percentile_ms(records, 0.50));
        }
    }

    groups
        .into_iter()
        .map(|((artifact_label, variant), records)| {
            let first = records[0];
            let p50_ms = percentile_ms(&records, 0.50);
            let baseline_p50_ms = baseline_p50_by_artifact
                .get(&artifact_label)
                .copied()
                .unwrap_or(p50_ms);
            Comparison {
                artifact_label,
                artifact_path: first.artifact_path.clone(),
                variant,
                rayon_threads: first.rayon_threads,
                trials: records.len(),
                file_count: first.file_count,
                total_bytes: first.total_bytes,
                inventory_sha256: first.inventory_sha256.clone(),
                checksum_matches_all: records.iter().all(|record| record.checksum_match),
                baseline_p50_ms,
                p50_ms,
                p95_ms: percentile_ms(&records, 0.95),
                best_ms: records
                    .iter()
                    .map(|record| record.elapsed_ms)
                    .fold(f64::INFINITY, f64::min),
                worst_ms: records
                    .iter()
                    .map(|record| record.elapsed_ms)
                    .fold(0.0, f64::max),
                p50_improvement_pct: improvement_pct(baseline_p50_ms, p50_ms),
                bytes_per_second_p50: bytes_per_second(first.total_bytes, p50_ms),
            }
        })
        .collect()
}

fn decide(blockers: &[String], comparisons: &[Comparison]) -> String {
    if !blockers.is_empty() {
        return "blocked_with_evidence".to_owned();
    }
    let best_candidate = comparisons
        .iter()
        .filter(|comparison| comparison.variant != "sequential")
        .filter(|comparison| comparison.checksum_matches_all)
        .map(|comparison| comparison.p50_improvement_pct)
        .fold(f64::NEG_INFINITY, f64::max);
    if best_candidate >= 5.0 {
        "accept_candidate_for_followup".to_owned()
    } else {
        "reject_candidate".to_owned()
    }
}

fn render_report(summary: &Summary) -> String {
    let mut out = String::new();
    out.push_str("# XR42 Rayon Manifest Hashing A/B\n\n");
    out.push_str("## Summary\n\n");
    out.push_str("| Field | Value |\n|---|---|\n");
    out.push_str(&format!("| Decision | `{}` |\n", summary.decision));
    out.push_str(&format!("| Run ID | `{}` |\n", summary.run_id));
    out.push_str(&format!("| Git SHA | `{}` |\n", summary.git_sha));
    out.push_str(&format!(
        "| Deterministic seed | `{}` |\n",
        summary.deterministic_seed
    ));
    out.push_str(&format!("| Seed usage | `{}` |\n", summary.seed_usage));
    out.push_str(&format!(
        "| Token lengths | `{}` |\n",
        summary.token_lengths
    ));
    out.push_str(&format!("| Command | `{}` |\n\n", summary.command));

    out.push_str("## Comparisons\n\n");
    out.push_str("| Artifact | Variant | Threads | Trials | Files | GB | p50 ms | p95 ms | p50 delta | Hash match |\n");
    out.push_str("|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n");
    for comparison in &summary.comparisons {
        out.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{:.3}` | `{:.3}` | `{:.3}` | `{:+.3}%` | `{}` |\n",
            comparison.artifact_label,
            comparison.variant,
            comparison
                .rayon_threads
                .map(|threads| threads.to_string())
                .unwrap_or_else(|| "n/a".to_owned()),
            comparison.trials,
            comparison.file_count,
            comparison.total_bytes as f64 / 1_000_000_000.0,
            comparison.p50_ms,
            comparison.p95_ms,
            comparison.p50_improvement_pct,
            comparison.checksum_matches_all
        ));
    }

    out.push_str("\n## Generated Files\n\n");
    for path in &summary.generated_files {
        out.push_str(&format!("- `{path}`\n"));
    }

    if !summary.blockers.is_empty() {
        out.push_str("\n## Blockers\n\n");
        for blocker in &summary.blockers {
            out.push_str(&format!("- {blocker}\n"));
        }
    }
    out
}

fn render_blockers(summary: &Summary) -> String {
    if summary.blockers.is_empty() {
        return format!(
            "# XR42 Blockers\n\nNo blockers recorded for run `{}`.\n",
            summary.run_id
        );
    }
    let mut out = format!("# XR42 Blockers\n\nRun `{}` blockers:\n", summary.run_id);
    for blocker in &summary.blockers {
        out.push_str(&format!("- {blocker}\n"));
    }
    out
}

fn render_decision(summary: &Summary) -> String {
    let mut out = String::new();
    out.push_str("# XR42 Decision\n\n");
    out.push_str(&format!("Decision: `{}`\n\n", summary.decision));
    out.push_str(
        "This is benchmark-only evidence. Do not integrate Rayon into the default manifest path \
         or any runtime inference path from this run alone.\n",
    );
    out
}

fn write_records_jsonl(path: &Path, records: &[Record]) -> Result<(), DynError> {
    let mut file = fs::File::create(path)?;
    for record in records {
        serde_json::to_writer(&mut file, record)?;
        file.write_all(b"\n")?;
    }
    Ok(())
}

fn percentile_ms(records: &[&Record], percentile: f64) -> f64 {
    let mut values = records
        .iter()
        .map(|record| record.elapsed_ms)
        .collect::<Vec<_>>();
    percentile_value(&mut values, percentile)
}

fn percentile_value(values: &mut [f64], percentile: f64) -> f64 {
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    if values.is_empty() {
        return 0.0;
    }
    let index = ((values.len() as f64 * percentile).ceil() as usize)
        .saturating_sub(1)
        .min(values.len() - 1);
    values[index]
}

fn improvement_pct(baseline_ms: f64, candidate_ms: f64) -> f64 {
    if baseline_ms <= 0.0 {
        0.0
    } else {
        ((baseline_ms - candidate_ms) / baseline_ms) * 100.0
    }
}

fn bytes_per_second(total_bytes: u64, elapsed_ms: f64) -> f64 {
    if elapsed_ms <= 0.0 {
        0.0
    } else {
        total_bytes as f64 / (elapsed_ms / 1000.0)
    }
}

fn artifact_label(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("artifact")
        .to_owned()
}

fn command_stdout(command: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(command).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn command_display() -> String {
    let args = env::args().skip(1).collect::<Vec<_>>().join(" ");
    if args.is_empty() {
        "cargo run -p gemma4d-bench --example xr42_rayon_manifest_hashing_ab --".to_owned()
    } else {
        format!("cargo run -p gemma4d-bench --example xr42_rayon_manifest_hashing_ab -- {args}")
    }
}

fn unix_now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn required_value<I>(args: &mut I, option: &str) -> Result<String, DynError>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| error(format!("{option} requires a value")))
}

fn error(message: impl Into<String>) -> DynError {
    Box::new(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        message.into(),
    ))
}

fn usage() -> String {
    format!(
        "usage: cargo run -p gemma4d-bench --example xr42_rayon_manifest_hashing_ab -- \
         [--artifact-path PATH ...] [--out-dir PATH] [--trials N] [--thread-counts 1,2,4]\n\
         defaults: --artifact-path {DEFAULT_MODEL_PATH} --artifact-path {DEFAULT_DRAFTER_PATH} \
         --out-dir {DEFAULT_OUT_DIR} --trials 3 --thread-counts 1,2,4"
    )
}
