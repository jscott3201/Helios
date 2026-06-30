#![doc = "Reference-parity benchmark harness and report generation."]

use std::{
    fs::{self, File},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

pub const CRATE_NAME: &str = "gemma4d-bench";

pub fn bootstrap_status() -> &'static str {
    "reference-parity-harness"
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptCase {
    pub id: String,
    pub prompt: String,
    pub token_ids: Vec<i32>,
    pub max_new_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunOptions {
    pub model_path: PathBuf,
    pub corpus_path: PathBuf,
    pub out_dir: PathBuf,
    pub gemma4d_bin: PathBuf,
    pub candidate_native: bool,
    pub reference_mlx_helper: bool,
    pub mlx_python: PathBuf,
    pub mlx_helper_script: PathBuf,
    pub llama_cmd: Option<String>,
    pub max_prompts: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReportOptions {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliError {
    Usage(String),
    Runtime(String),
}

impl CliError {
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Usage(_) => 2,
            Self::Runtime(_) => 1,
        }
    }
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Usage(message) | Self::Runtime(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for CliError {}

#[derive(Debug, Clone, PartialEq)]
struct BenchEnvironment {
    os: String,
    arch: String,
    rustc: String,
    git_commit: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelRevision {
    path: String,
    config_hash: String,
    tokenizer_hash: String,
}

#[derive(Debug, Clone, PartialEq)]
struct RunResult {
    name: String,
    status: String,
    command: String,
    generated_tokens: Vec<i32>,
    ttft_ms: Option<f64>,
    decode_ms: Option<f64>,
    decode_tps: Option<f64>,
    peak_memory_gb: Option<f64>,
    peak_rss_mb: Option<f64>,
    exit_code: Option<i32>,
    stdout: String,
    stderr: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenDiff {
    pub status: String,
    pub summary: String,
    pub detail: String,
}

pub fn run_cli<I, S, W, E>(args: I, stdout: &mut W, stderr: &mut E) -> i32
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
    W: Write,
    E: Write,
{
    match dispatch(args) {
        Ok(message) => {
            let _ = writeln!(stdout, "{message}");
            0
        }
        Err(error) => {
            let _ = writeln!(stderr, "{error}");
            error.exit_code()
        }
    }
}

pub fn dispatch<I, S>(args: I) -> Result<String, CliError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut args = args.into_iter().map(Into::into);
    let Some(command) = args.next() else {
        return Err(CliError::Usage(usage()));
    };

    match command.as_str() {
        "run" => {
            let options = parse_run_options(args)?;
            run_benchmarks(&options)
        }
        "report" => {
            let options = parse_report_options(args)?;
            generate_report_file(&options)
        }
        "-h" | "--help" | "help" => Ok(usage()),
        other => Err(CliError::Usage(format!(
            "unknown command '{other}'\n{}",
            usage()
        ))),
    }
}

pub fn parse_run_options<I, S>(args: I) -> Result<RunOptions, CliError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut args = args.into_iter().map(Into::into).peekable();
    let mut model_path = None;
    let mut corpus_path = PathBuf::from("benchmarks/prompts/M04-corpus.tsv");
    let mut out_dir = PathBuf::from("benchmarks/out/M04");
    let mut gemma4d_bin = PathBuf::from("target/debug/gemma4d");
    let mut candidate_native = false;
    let mut reference_mlx_helper = false;
    let mut mlx_python = PathBuf::from(
        std::env::var("GEMMA4D_MLX_LM_PYTHON")
            .unwrap_or_else(|_| "/opt/homebrew/opt/mlx-lm/libexec/bin/python".to_owned()),
    );
    let mut mlx_helper_script = PathBuf::from("native/gemma4_mlx/scripts/gemma4d_mlx_lm_helper.py");
    let mut llama_cmd = None;
    let mut max_prompts = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model-path" => {
                model_path = Some(PathBuf::from(required_value(&mut args, "--model-path")?));
            }
            "--corpus" => {
                corpus_path = PathBuf::from(required_value(&mut args, "--corpus")?);
            }
            "--out-dir" => {
                out_dir = PathBuf::from(required_value(&mut args, "--out-dir")?);
            }
            "--gemma4d-bin" => {
                gemma4d_bin = PathBuf::from(required_value(&mut args, "--gemma4d-bin")?);
            }
            "--candidate-native" => {
                candidate_native = true;
            }
            "--reference" => {
                let value = required_value(&mut args, "--reference")?;
                match value.as_str() {
                    "mlx-helper" | "mlx-python" => reference_mlx_helper = true,
                    other => {
                        return Err(CliError::Usage(format!(
                            "unsupported reference '{other}', expected mlx-helper"
                        )));
                    }
                }
            }
            "--mlx-python" => {
                mlx_python = PathBuf::from(required_value(&mut args, "--mlx-python")?);
            }
            "--mlx-helper-script" => {
                mlx_helper_script =
                    PathBuf::from(required_value(&mut args, "--mlx-helper-script")?);
            }
            "--llama-cmd" => {
                llama_cmd = Some(required_value(&mut args, "--llama-cmd")?);
            }
            "--max-prompts" => {
                let value = required_value(&mut args, "--max-prompts")?;
                max_prompts = Some(parse_positive_usize(&value, "--max-prompts")?);
            }
            "-h" | "--help" => return Err(CliError::Usage(run_usage())),
            other => {
                return Err(CliError::Usage(format!(
                    "unknown run option '{other}'\n{}",
                    run_usage()
                )));
            }
        }
    }

    let model_path = model_path
        .ok_or_else(|| CliError::Usage(format!("run requires --model-path\n{}", run_usage())))?;

    Ok(RunOptions {
        model_path,
        corpus_path,
        out_dir,
        gemma4d_bin,
        candidate_native,
        reference_mlx_helper,
        mlx_python,
        mlx_helper_script,
        llama_cmd,
        max_prompts,
    })
}

pub fn parse_report_options<I, S>(args: I) -> Result<ReportOptions, CliError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut args = args.into_iter().map(Into::into).peekable();
    let mut input_path = None;
    let mut output_path = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => {
                input_path = Some(PathBuf::from(required_value(&mut args, "--input")?));
            }
            "--output" => {
                output_path = Some(PathBuf::from(required_value(&mut args, "--output")?));
            }
            "-h" | "--help" => return Err(CliError::Usage(report_usage())),
            other => {
                return Err(CliError::Usage(format!(
                    "unknown report option '{other}'\n{}",
                    report_usage()
                )));
            }
        }
    }

    let input_path = input_path
        .ok_or_else(|| CliError::Usage(format!("report requires --input\n{}", report_usage())))?;
    let output_path = output_path
        .ok_or_else(|| CliError::Usage(format!("report requires --output\n{}", report_usage())))?;

    Ok(ReportOptions {
        input_path,
        output_path,
    })
}

pub fn load_prompt_corpus(path: &Path) -> Result<Vec<PromptCase>, CliError> {
    let text = fs::read_to_string(path)
        .map_err(|error| CliError::Runtime(format!("failed to read corpus: {error}")))?;
    let mut prompts = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        let line = line.trim_end();
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 4 {
            return Err(CliError::Runtime(format!(
                "invalid corpus line {}: expected 4 tab-separated fields",
                line_index + 1
            )));
        }
        let token_ids = parse_token_ids(fields[2]).map_err(|message| {
            CliError::Runtime(format!(
                "invalid token_ids on corpus line {}: {message}",
                line_index + 1
            ))
        })?;
        if token_ids.is_empty() {
            return Err(CliError::Runtime(format!(
                "invalid corpus line {}: token_ids must not be empty",
                line_index + 1
            )));
        }
        prompts.push(PromptCase {
            id: fields[0].to_owned(),
            prompt: fields[1].to_owned(),
            token_ids,
            max_new_tokens: parse_positive_usize(fields[3], "max_new_tokens")
                .map_err(|error| CliError::Runtime(error.to_string()))?,
        });
    }
    if prompts.is_empty() {
        return Err(CliError::Runtime("prompt corpus is empty".to_owned()));
    }
    Ok(prompts)
}

pub fn compare_tokens(candidate: &[i32], reference: &[i32]) -> TokenDiff {
    if candidate == reference {
        return TokenDiff {
            status: "passed".to_owned(),
            summary: format!("tokens match ({} tokens)", candidate.len()),
            detail: "match".to_owned(),
        };
    }

    let shared_len = candidate.len().min(reference.len());
    for index in 0..shared_len {
        if candidate[index] != reference[index] {
            return TokenDiff {
                status: "failed".to_owned(),
                summary: format!(
                    "first mismatch at token {index}: candidate={} reference={}",
                    candidate[index], reference[index]
                ),
                detail: format!(
                    "candidate={} reference={}",
                    format_tokens(candidate),
                    format_tokens(reference)
                ),
            };
        }
    }

    let summary = if candidate.len() < reference.len() {
        format!(
            "candidate ended early at {} tokens; reference has {} tokens",
            candidate.len(),
            reference.len()
        )
    } else {
        format!(
            "candidate has extra tokens: candidate {} tokens, reference {} tokens",
            candidate.len(),
            reference.len()
        )
    };

    TokenDiff {
        status: "failed".to_owned(),
        summary,
        detail: format!(
            "candidate={} reference={}",
            format_tokens(candidate),
            format_tokens(reference)
        ),
    }
}

pub fn run_benchmarks(options: &RunOptions) -> Result<String, CliError> {
    fs::create_dir_all(&options.out_dir)
        .map_err(|error| CliError::Runtime(format!("failed to create out dir: {error}")))?;
    let records_path = options.out_dir.join("records.jsonl");
    let report_path = options.out_dir.join("report.md");

    let mut prompts = load_prompt_corpus(&options.corpus_path)?;
    if let Some(max_prompts) = options.max_prompts {
        prompts.truncate(max_prompts);
    }

    let run_id = run_id();
    let environment = capture_environment();
    let model_revision = capture_model_revision(&options.model_path);
    let mut writer = File::create(&records_path)
        .map_err(|error| CliError::Runtime(format!("failed to create JSONL: {error}")))?;

    for prompt in &prompts {
        let candidate = run_candidate(options, prompt);
        let mut wrote_reference = false;

        if options.reference_mlx_helper {
            let reference = run_mlx_helper_reference(options, prompt);
            let diff = comparison_for(&candidate, &reference);
            let line = jsonl_record(
                &run_id,
                prompt,
                &environment,
                &model_revision,
                &candidate,
                &reference,
                &diff,
            );
            writeln!(writer, "{line}")
                .map_err(|error| CliError::Runtime(format!("failed to write JSONL: {error}")))?;
            wrote_reference = true;
        }

        if let Some(template) = &options.llama_cmd {
            let reference =
                run_template_reference("llama_cpp", template, &options.model_path, prompt);
            let diff = comparison_for(&candidate, &reference);
            let line = jsonl_record(
                &run_id,
                prompt,
                &environment,
                &model_revision,
                &candidate,
                &reference,
                &diff,
            );
            writeln!(writer, "{line}")
                .map_err(|error| CliError::Runtime(format!("failed to write JSONL: {error}")))?;
            wrote_reference = true;
        }

        if !wrote_reference {
            let reference = inconclusive_reference(
                "none",
                "no reference configured; pass --reference mlx-helper or --llama-cmd",
            );
            let diff = comparison_for(&candidate, &reference);
            let line = jsonl_record(
                &run_id,
                prompt,
                &environment,
                &model_revision,
                &candidate,
                &reference,
                &diff,
            );
            writeln!(writer, "{line}")
                .map_err(|error| CliError::Runtime(format!("failed to write JSONL: {error}")))?;
        }
    }
    drop(writer);

    let report = generate_report(&records_path)?;
    fs::write(&report_path, report)
        .map_err(|error| CliError::Runtime(format!("failed to write report: {error}")))?;

    Ok(format!(
        "wrote {} and {}",
        records_path.display(),
        report_path.display()
    ))
}

pub fn generate_report_file(options: &ReportOptions) -> Result<String, CliError> {
    let report = generate_report(&options.input_path)?;
    if let Some(parent) = options.output_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| CliError::Runtime(format!("failed to create report dir: {error}")))?;
    }
    fs::write(&options.output_path, report)
        .map_err(|error| CliError::Runtime(format!("failed to write report: {error}")))?;
    Ok(format!("wrote {}", options.output_path.display()))
}

pub fn generate_report(input_path: &Path) -> Result<String, CliError> {
    let text = fs::read_to_string(input_path)
        .map_err(|error| CliError::Runtime(format!("failed to read JSONL: {error}")))?;
    let mut rows = Vec::new();
    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut inconclusive = 0usize;
    let mut commands = Vec::new();
    let mut env_summary = None;
    let mut model_summary = None;

    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let prompt_id = extract_json_string(line, "prompt_id").unwrap_or_else(|| "?".to_owned());
        let reference_name =
            extract_json_string(line, "reference_name").unwrap_or_else(|| "?".to_owned());
        let status = extract_json_string(line, "comparison_status")
            .unwrap_or_else(|| "inconclusive".to_owned());
        let summary = extract_json_string(line, "comparison_summary")
            .unwrap_or_else(|| "missing summary".to_owned());
        let candidate_command = extract_json_string(line, "candidate_command").unwrap_or_default();
        let reference_command = extract_json_string(line, "reference_command").unwrap_or_default();
        if !candidate_command.is_empty() {
            commands.push(candidate_command);
        }
        if !reference_command.is_empty() {
            commands.push(reference_command);
        }
        if env_summary.is_none() {
            let rustc = extract_json_string(line, "rustc").unwrap_or_else(|| "unknown".to_owned());
            let os = extract_json_string(line, "os").unwrap_or_else(|| "unknown".to_owned());
            let git_commit =
                extract_json_string(line, "git_commit").unwrap_or_else(|| "unknown".to_owned());
            env_summary = Some((os, rustc, git_commit));
        }
        if model_summary.is_none() {
            let model_path =
                extract_json_string(line, "model_path").unwrap_or_else(|| "unknown".to_owned());
            let config_hash =
                extract_json_string(line, "config_hash").unwrap_or_else(|| "unknown".to_owned());
            let tokenizer_hash =
                extract_json_string(line, "tokenizer_hash").unwrap_or_else(|| "unknown".to_owned());
            model_summary = Some((model_path, config_hash, tokenizer_hash));
        }

        match status.as_str() {
            "passed" => passed += 1,
            "failed" => failed += 1,
            _ => inconclusive += 1,
        }
        rows.push((prompt_id, reference_name, status, summary));
    }

    commands.sort();
    commands.dedup();

    let mut report = String::new();
    report.push_str("# M04 Reference Parity Report\n\n");
    report.push_str("## Summary\n\n");
    report.push_str(&format!(
        "- Passed: {passed}\n- Failed: {failed}\n- Inconclusive: {inconclusive}\n\n"
    ));
    if let Some((os, rustc, git_commit)) = env_summary {
        report.push_str("## Environment\n\n");
        report.push_str("| Item | Value |\n|---|---|\n");
        report.push_str(&format!("| OS | {} |\n", markdown_escape(&os)));
        report.push_str(&format!("| Rust | {} |\n", markdown_escape(&rustc)));
        report.push_str(&format!(
            "| Git commit | `{}` |\n\n",
            markdown_escape(&git_commit)
        ));
    }
    if let Some((model_path, config_hash, tokenizer_hash)) = model_summary {
        report.push_str("## Model\n\n");
        report.push_str("| Item | Value |\n|---|---|\n");
        report.push_str(&format!("| Path | `{}` |\n", markdown_escape(&model_path)));
        report.push_str(&format!(
            "| config.json FNV64 | `{}` |\n",
            markdown_escape(&config_hash)
        ));
        report.push_str(&format!(
            "| tokenizer.json FNV64 | `{}` |\n\n",
            markdown_escape(&tokenizer_hash)
        ));
    }
    report.push_str("## Results\n\n");
    report.push_str("| Prompt | Reference | Status | Summary |\n|---|---|---|---|\n");
    for (prompt_id, reference_name, status, summary) in rows {
        report.push_str(&format!(
            "| `{}` | `{}` | {} | {} |\n",
            markdown_escape(&prompt_id),
            markdown_escape(&reference_name),
            markdown_escape(&status),
            markdown_escape(&summary)
        ));
    }
    report.push_str("\n## Commands\n\n```text\n");
    for command in commands {
        report.push_str(&command);
        report.push('\n');
    }
    report.push_str("```\n");
    Ok(report)
}

fn run_candidate(options: &RunOptions, prompt: &PromptCase) -> RunResult {
    let token_ids = format_tokens_csv(&prompt.token_ids);
    let mut command = Command::new(&options.gemma4d_bin);
    command
        .arg("generate")
        .arg("--model-path")
        .arg(&options.model_path)
        .arg("--token-ids")
        .arg(&token_ids)
        .arg("--max-new-tokens")
        .arg(prompt.max_new_tokens.to_string())
        .arg("--json");
    if options.candidate_native {
        command.env("GEMMA4D_REQUIRE_MLX", "1");
        command.env("GEMMA4D_USE_NATIVE_GRAPH", "1");
    }

    let mut display_parts = Vec::new();
    if options.candidate_native {
        display_parts.push("GEMMA4D_REQUIRE_MLX=1".to_owned());
        display_parts.push("GEMMA4D_USE_NATIVE_GRAPH=1".to_owned());
    }
    display_parts.push(shell_quote(&options.gemma4d_bin.display().to_string()));
    display_parts.extend(
        [
            "generate",
            "--model-path",
            &options.model_path.display().to_string(),
            "--token-ids",
            &token_ids,
            "--max-new-tokens",
            &prompt.max_new_tokens.to_string(),
            "--json",
        ]
        .into_iter()
        .map(shell_quote),
    );
    let command_display = display_parts.join(" ");

    match command.output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            if output.status.success() {
                RunResult {
                    name: "gemma4d".to_owned(),
                    status: "ok".to_owned(),
                    command: command_display,
                    generated_tokens: extract_generated_tokens(&stdout).unwrap_or_default(),
                    ttft_ms: extract_json_f64(&stdout, "ttft_ms"),
                    decode_ms: extract_json_f64(&stdout, "decode_ms"),
                    decode_tps: extract_json_f64(&stdout, "decode_tps"),
                    peak_memory_gb: extract_json_f64(&stdout, "peak_memory_gb"),
                    peak_rss_mb: extract_json_f64(&stdout, "peak_rss_mb"),
                    exit_code: output.status.code(),
                    stdout,
                    stderr,
                }
            } else {
                errored_result(
                    "gemma4d",
                    command_display,
                    output.status.code(),
                    stdout,
                    stderr,
                )
            }
        }
        Err(error) => errored_result(
            "gemma4d",
            command_display,
            None,
            String::new(),
            format!("failed to spawn gemma4d: {error}"),
        ),
    }
}

fn run_mlx_helper_reference(options: &RunOptions, prompt: &PromptCase) -> RunResult {
    let command_display = format!(
        "{} {} {}",
        shell_quote(&options.mlx_python.display().to_string()),
        shell_quote(&options.mlx_helper_script.display().to_string()),
        shell_quote(&options.model_path.display().to_string())
    );
    let mut child = match Command::new(&options.mlx_python)
        .arg(&options.mlx_helper_script)
        .arg(&options.model_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            return errored_result(
                "mlx_python",
                command_display,
                None,
                String::new(),
                format!("failed to spawn MLX helper: {error}"),
            );
        }
    };

    let Some(stdout) = child.stdout.take() else {
        return errored_result(
            "mlx_python",
            command_display,
            None,
            String::new(),
            "failed to capture MLX helper stdout".to_owned(),
        );
    };
    let mut stdin = child.stdin.take();
    let mut reader = BufReader::new(stdout);
    let mut helper_stdout = Vec::new();

    let mut line = String::new();
    if let Err(error) = reader.read_line(&mut line) {
        return errored_result(
            "mlx_python",
            command_display,
            None,
            String::new(),
            format!("failed to read MLX helper startup: {error}"),
        );
    }
    helper_stdout.push(line.trim().to_owned());
    if !line.contains("\"ok\":true") {
        let stderr = wait_stderr(child);
        return errored_result(
            "mlx_python",
            command_display,
            None,
            helper_stdout.join("\n"),
            stderr,
        );
    }

    let mut generated_tokens = Vec::with_capacity(prompt.max_new_tokens);
    let mut peak_memory_gb = 0.0f64;
    let mut peak_rss_mb = 0.0f64;
    let started = Instant::now();
    let prefill_request = format!(
        "{{\"cmd\":\"prefill\",\"tokens\":[{}]}}\n",
        format_tokens_csv(&prompt.token_ids)
    );
    let prefill_line = match send_helper_request(&mut stdin, &mut reader, &prefill_request) {
        Ok(line) => line,
        Err(error) => {
            let stderr = wait_stderr(child);
            return errored_result(
                "mlx_python",
                command_display,
                None,
                helper_stdout.join("\n"),
                format!("{error}; {stderr}"),
            );
        }
    };
    let ttft = started.elapsed();
    helper_stdout.push(prefill_line.clone());
    if !prefill_line.contains("\"ok\":true") {
        let stderr = wait_stderr(child);
        return errored_result(
            "mlx_python",
            command_display,
            None,
            helper_stdout.join("\n"),
            stderr,
        );
    }
    let Some(mut token) = extract_json_i32(&prefill_line, "greedy_token") else {
        let stderr = wait_stderr(child);
        return errored_result(
            "mlx_python",
            command_display,
            None,
            helper_stdout.join("\n"),
            format!("could not parse greedy token from MLX helper; {stderr}"),
        );
    };
    generated_tokens.push(token);
    peak_memory_gb =
        peak_memory_gb.max(extract_json_f64(&prefill_line, "peak_memory_gb").unwrap_or(0.0));
    peak_rss_mb = peak_rss_mb.max(extract_json_f64(&prefill_line, "peak_rss_mb").unwrap_or(0.0));

    let decode_started = Instant::now();
    while generated_tokens.len() < prompt.max_new_tokens {
        let decode_request = format!("{{\"cmd\":\"decode_one\",\"token\":{token}}}\n");
        let decode_line = match send_helper_request(&mut stdin, &mut reader, &decode_request) {
            Ok(line) => line,
            Err(error) => {
                let stderr = wait_stderr(child);
                return errored_result(
                    "mlx_python",
                    command_display,
                    None,
                    helper_stdout.join("\n"),
                    format!("{error}; {stderr}"),
                );
            }
        };
        helper_stdout.push(decode_line.clone());
        if !decode_line.contains("\"ok\":true") {
            let stderr = wait_stderr(child);
            return errored_result(
                "mlx_python",
                command_display,
                None,
                helper_stdout.join("\n"),
                stderr,
            );
        }
        let Some(next_token) = extract_json_i32(&decode_line, "greedy_token") else {
            let stderr = wait_stderr(child);
            return errored_result(
                "mlx_python",
                command_display,
                None,
                helper_stdout.join("\n"),
                format!("could not parse decode token from MLX helper; {stderr}"),
            );
        };
        token = next_token;
        generated_tokens.push(token);
        peak_memory_gb =
            peak_memory_gb.max(extract_json_f64(&decode_line, "peak_memory_gb").unwrap_or(0.0));
        peak_rss_mb = peak_rss_mb.max(extract_json_f64(&decode_line, "peak_rss_mb").unwrap_or(0.0));
    }
    let decode = decode_started.elapsed();

    if let Some(stdin) = stdin.as_mut() {
        let _ = stdin.write_all(b"{\"cmd\":\"shutdown\"}\n");
    }
    let stderr = wait_stderr(child);

    RunResult {
        name: "mlx_python".to_owned(),
        status: "ok".to_owned(),
        command: command_display,
        generated_tokens,
        ttft_ms: Some(duration_ms(ttft)),
        decode_ms: Some(duration_ms(decode)),
        decode_tps: decode_tps(prompt.max_new_tokens, decode),
        peak_memory_gb: Some(peak_memory_gb),
        peak_rss_mb: Some(peak_rss_mb),
        exit_code: Some(0),
        stdout: helper_stdout.join("\n"),
        stderr,
    }
}

fn run_template_reference(
    name: &str,
    template: &str,
    model_path: &Path,
    prompt: &PromptCase,
) -> RunResult {
    let command = template
        .replace("{model_path}", &model_path.display().to_string())
        .replace("{prompt}", &prompt.prompt)
        .replace("{token_ids}", &format_tokens_csv(&prompt.token_ids))
        .replace("{max_new_tokens}", &prompt.max_new_tokens.to_string());
    match Command::new("/bin/sh").arg("-lc").arg(&command).output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            if !output.status.success() {
                return errored_result(name, command, output.status.code(), stdout, stderr);
            }
            let Some(generated_tokens) = extract_generated_tokens(&stdout) else {
                return RunResult {
                    name: name.to_owned(),
                    status: "inconclusive".to_owned(),
                    command,
                    generated_tokens: Vec::new(),
                    ttft_ms: extract_json_f64(&stdout, "ttft_ms"),
                    decode_ms: extract_json_f64(&stdout, "decode_ms"),
                    decode_tps: extract_json_f64(&stdout, "decode_tps"),
                    peak_memory_gb: extract_json_f64(&stdout, "peak_memory_gb"),
                    peak_rss_mb: extract_json_f64(&stdout, "peak_rss_mb"),
                    exit_code: output.status.code(),
                    stdout,
                    stderr: if stderr.is_empty() {
                        "configured command did not emit generated_tokens JSON".to_owned()
                    } else {
                        stderr
                    },
                };
            };
            RunResult {
                name: name.to_owned(),
                status: "ok".to_owned(),
                command,
                generated_tokens,
                ttft_ms: extract_json_f64(&stdout, "ttft_ms"),
                decode_ms: extract_json_f64(&stdout, "decode_ms"),
                decode_tps: extract_json_f64(&stdout, "decode_tps"),
                peak_memory_gb: extract_json_f64(&stdout, "peak_memory_gb"),
                peak_rss_mb: extract_json_f64(&stdout, "peak_rss_mb"),
                exit_code: output.status.code(),
                stdout,
                stderr,
            }
        }
        Err(error) => errored_result(
            name,
            command,
            None,
            String::new(),
            format!("failed to run configured command: {error}"),
        ),
    }
}

fn send_helper_request(
    stdin: &mut Option<std::process::ChildStdin>,
    reader: &mut BufReader<std::process::ChildStdout>,
    request: &str,
) -> Result<String, String> {
    let Some(stdin) = stdin.as_mut() else {
        return Err("MLX helper stdin is unavailable".to_owned());
    };
    stdin
        .write_all(request.as_bytes())
        .map_err(|error| format!("failed to write MLX helper request: {error}"))?;
    stdin
        .flush()
        .map_err(|error| format!("failed to flush MLX helper request: {error}"))?;
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|error| format!("failed to read MLX helper response: {error}"))?;
    Ok(line.trim().to_owned())
}

fn wait_stderr(child: std::process::Child) -> String {
    let output = child.wait_with_output();
    match output {
        Ok(output) => String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        Err(error) => format!("failed to wait for child process: {error}"),
    }
}

fn comparison_for(candidate: &RunResult, reference: &RunResult) -> TokenDiff {
    if candidate.status != "ok" {
        return TokenDiff {
            status: "inconclusive".to_owned(),
            summary: format!("candidate unavailable: {}", candidate.status),
            detail: candidate.stderr.clone(),
        };
    }
    if reference.status != "ok" {
        return TokenDiff {
            status: "inconclusive".to_owned(),
            summary: format!(
                "reference {} unavailable: {}",
                reference.name, reference.status
            ),
            detail: reference.stderr.clone(),
        };
    }
    compare_tokens(&candidate.generated_tokens, &reference.generated_tokens)
}

fn jsonl_record(
    run_id: &str,
    prompt: &PromptCase,
    environment: &BenchEnvironment,
    model_revision: &ModelRevision,
    candidate: &RunResult,
    reference: &RunResult,
    diff: &TokenDiff,
) -> String {
    format!(
        "{{\"schema_version\":1,\"run_id\":\"{}\",\"prompt_id\":\"{}\",\"prompt\":\"{}\",\"token_ids\":{},\"max_new_tokens\":{},\"environment\":{{\"os\":\"{}\",\"arch\":\"{}\",\"rustc\":\"{}\",\"git_commit\":\"{}\"}},\"model_revision\":{{\"model_path\":\"{}\",\"config_hash\":\"{}\",\"tokenizer_hash\":\"{}\"}},\"candidate_name\":\"{}\",\"candidate_status\":\"{}\",\"candidate_command\":\"{}\",\"candidate_generated_tokens\":{},\"candidate_metrics\":{},\"candidate_stdout\":\"{}\",\"candidate_stderr\":\"{}\",\"reference_name\":\"{}\",\"reference_status\":\"{}\",\"reference_command\":\"{}\",\"reference_generated_tokens\":{},\"reference_metrics\":{},\"reference_stdout\":\"{}\",\"reference_stderr\":\"{}\",\"comparison_status\":\"{}\",\"comparison_summary\":\"{}\",\"comparison_detail\":\"{}\"}}",
        json_escape(run_id),
        json_escape(&prompt.id),
        json_escape(&prompt.prompt),
        format_tokens(&prompt.token_ids),
        prompt.max_new_tokens,
        json_escape(&environment.os),
        json_escape(&environment.arch),
        json_escape(&environment.rustc),
        json_escape(&environment.git_commit),
        json_escape(&model_revision.path),
        json_escape(&model_revision.config_hash),
        json_escape(&model_revision.tokenizer_hash),
        json_escape(&candidate.name),
        json_escape(&candidate.status),
        json_escape(&candidate.command),
        format_tokens(&candidate.generated_tokens),
        metrics_json(candidate),
        json_escape(&candidate.stdout),
        json_escape(&candidate.stderr),
        json_escape(&reference.name),
        json_escape(&reference.status),
        json_escape(&reference.command),
        format_tokens(&reference.generated_tokens),
        metrics_json(reference),
        json_escape(&reference.stdout),
        json_escape(&reference.stderr),
        json_escape(&diff.status),
        json_escape(&diff.summary),
        json_escape(&diff.detail),
    )
}

fn metrics_json(result: &RunResult) -> String {
    format!(
        "{{\"ttft_ms\":{},\"decode_ms\":{},\"decode_tps\":{},\"peak_memory_gb\":{},\"peak_rss_mb\":{},\"exit_code\":{}}}",
        json_number(result.ttft_ms),
        json_number(result.decode_ms),
        json_number(result.decode_tps),
        json_number(result.peak_memory_gb),
        json_number(result.peak_rss_mb),
        result
            .exit_code
            .map(|value| value.to_string())
            .unwrap_or_else(|| "null".to_owned())
    )
}

fn json_number(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.3}"))
        .unwrap_or_else(|| "null".to_owned())
}

fn errored_result(
    name: &str,
    command: String,
    exit_code: Option<i32>,
    stdout: String,
    stderr: String,
) -> RunResult {
    RunResult {
        name: name.to_owned(),
        status: "error".to_owned(),
        command,
        generated_tokens: Vec::new(),
        ttft_ms: None,
        decode_ms: None,
        decode_tps: None,
        peak_memory_gb: None,
        peak_rss_mb: None,
        exit_code,
        stdout,
        stderr,
    }
}

fn inconclusive_reference(name: &str, reason: &str) -> RunResult {
    RunResult {
        name: name.to_owned(),
        status: "inconclusive".to_owned(),
        command: String::new(),
        generated_tokens: Vec::new(),
        ttft_ms: None,
        decode_ms: None,
        decode_tps: None,
        peak_memory_gb: None,
        peak_rss_mb: None,
        exit_code: None,
        stdout: String::new(),
        stderr: reason.to_owned(),
    }
}

fn capture_environment() -> BenchEnvironment {
    BenchEnvironment {
        os: command_stdout("uname", &["-a"]),
        arch: std::env::consts::ARCH.to_owned(),
        rustc: command_stdout("rustc", &["-V"]),
        git_commit: command_stdout("git", &["rev-parse", "HEAD"]),
    }
}

fn capture_model_revision(model_path: &Path) -> ModelRevision {
    ModelRevision {
        path: model_path.display().to_string(),
        config_hash: file_hash(&model_path.join("config.json")),
        tokenizer_hash: file_hash(&model_path.join("tokenizer.json")),
    }
}

fn command_stdout(program: &str, args: &[&str]) -> String {
    match Command::new(program).args(args).output() {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).trim().to_owned()
        }
        Ok(output) => format!(
            "unavailable: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ),
        Err(error) => format!("unavailable: {error}"),
    }
}

fn file_hash(path: &Path) -> String {
    match fs::read(path) {
        Ok(bytes) => format!("{:016x}", fnv1a64(&bytes)),
        Err(error) => format!("unavailable:{error}"),
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn run_id() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0));
    format!("m04-{}-{}", now.as_secs(), now.subsec_nanos())
}

fn parse_token_ids(value: &str) -> Result<Vec<i32>, String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(|token| {
            let parsed = token
                .parse::<i32>()
                .map_err(|error| format!("token '{token}' is not an integer: {error}"))?;
            if parsed < 0 {
                return Err(format!("token '{token}' must be non-negative"));
            }
            Ok(parsed)
        })
        .collect()
}

fn parse_positive_usize(value: &str, flag: &str) -> Result<usize, CliError> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| CliError::Usage(format!("{flag} must be a positive integer: {error}")))?;
    if parsed == 0 {
        return Err(CliError::Usage(format!("{flag} must be greater than zero")));
    }
    Ok(parsed)
}

fn required_value<I>(args: &mut std::iter::Peekable<I>, flag: &str) -> Result<String, CliError>
where
    I: Iterator<Item = String>,
{
    let Some(value) = args.next() else {
        return Err(CliError::Usage(format!("{flag} requires a value")));
    };
    if value.starts_with("--") {
        return Err(CliError::Usage(format!("{flag} requires a value")));
    }
    Ok(value)
}

fn extract_generated_tokens(json: &str) -> Option<Vec<i32>> {
    let key = "\"generated_tokens\"";
    let key_pos = json.find(key)?;
    let after_key = &json[key_pos + key.len()..];
    let array_start = after_key.find('[')? + key_pos + key.len();
    let rest = &json[array_start + 1..];
    let array_end = rest.find(']')? + array_start + 1;
    let inner = &json[array_start + 1..array_end];
    if inner.trim().is_empty() {
        return Some(Vec::new());
    }
    parse_token_ids(inner).ok()
}

fn extract_json_i32(json: &str, key: &str) -> Option<i32> {
    let value = extract_json_number_slice(json, key)?;
    value.parse::<i32>().ok()
}

fn extract_json_f64(json: &str, key: &str) -> Option<f64> {
    let value = extract_json_number_slice(json, key)?;
    value.parse::<f64>().ok()
}

fn extract_json_number_slice<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let quoted_key = format!("\"{key}\"");
    let key_pos = json.find(&quoted_key)?;
    let after_key = &json[key_pos + quoted_key.len()..];
    let colon = after_key.find(':')? + key_pos + quoted_key.len();
    let mut start = colon + 1;
    while start < json.len() && json.as_bytes()[start].is_ascii_whitespace() {
        start += 1;
    }
    let mut end = start;
    while end < json.len() {
        let ch = json.as_bytes()[end] as char;
        if ch.is_ascii_digit() || matches!(ch, '-' | '+' | '.' | 'e' | 'E') {
            end += 1;
        } else {
            break;
        }
    }
    if end == start {
        return None;
    }
    Some(&json[start..end])
}

fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let quoted_key = format!("\"{key}\"");
    let key_pos = json.find(&quoted_key)?;
    let after_key = &json[key_pos + quoted_key.len()..];
    let colon = after_key.find(':')? + key_pos + quoted_key.len();
    let mut start = colon + 1;
    while start < json.len() && json.as_bytes()[start].is_ascii_whitespace() {
        start += 1;
    }
    if json.as_bytes().get(start) != Some(&b'"') {
        return None;
    }
    start += 1;
    let mut out = String::new();
    let mut escaped = false;
    for ch in json[start..].chars() {
        if escaped {
            match ch {
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                other => out.push(other),
            }
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Some(out);
        } else {
            out.push(ch);
        }
    }
    None
}

fn format_tokens(tokens: &[i32]) -> String {
    let values = tokens
        .iter()
        .map(i32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!("[{values}]")
}

fn format_tokens_csv(tokens: &[i32]) -> String {
    tokens
        .iter()
        .map(i32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            other if other.is_control() => escaped.push(' '),
            other => escaped.push(other),
        }
    }
    escaped
}

fn markdown_escape(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}

fn shell_quote(value: &str) -> String {
    if !value.is_empty()
        && value.chars().all(|ch| {
            ch.is_ascii_alphanumeric() || matches!(ch, '/' | '.' | '_' | '-' | '=' | ':' | ',')
        })
    {
        return value.to_owned();
    }
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn decode_tps(max_new_tokens: usize, decode: Duration) -> Option<f64> {
    let decode_tokens = max_new_tokens.saturating_sub(1);
    if decode_tokens == 0 || decode.is_zero() {
        Some(0.0)
    } else {
        Some(decode_tokens as f64 / decode.as_secs_f64())
    }
}

fn usage() -> String {
    format!(
        "usage: gemma4d-bench <command>\n\n{}\n\n{}",
        run_usage(),
        report_usage()
    )
}

fn run_usage() -> String {
    "usage: gemma4d-bench run --model-path PATH [--corpus PATH] [--out-dir DIR] [--gemma4d-bin PATH] [--candidate-native] [--reference mlx-helper] [--llama-cmd TEMPLATE] [--max-prompts N]".to_owned()
}

fn report_usage() -> String {
    "usage: gemma4d-bench report --input records.jsonl --output report.md".to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_harness_status() {
        assert_eq!(CRATE_NAME, "gemma4d-bench");
        assert_eq!(bootstrap_status(), "reference-parity-harness");
    }

    #[test]
    fn token_diff_reports_match_and_first_mismatch() {
        let matched = compare_tokens(&[1, 2, 3], &[1, 2, 3]);
        assert_eq!(matched.status, "passed");
        assert!(matched.summary.contains("3 tokens"));

        let mismatched = compare_tokens(&[1, 4, 3], &[1, 2, 3]);
        assert_eq!(mismatched.status, "failed");
        assert!(mismatched.summary.contains("first mismatch at token 1"));
        assert!(mismatched.detail.contains("candidate=[1,4,3]"));
    }

    #[test]
    fn token_diff_reports_length_mismatch_readably() {
        let diff = compare_tokens(&[1, 2], &[1, 2, 3]);
        assert_eq!(diff.status, "failed");
        assert!(diff.summary.contains("ended early"));
    }

    #[test]
    fn parses_generated_tokens_from_server_json() {
        let tokens = extract_generated_tokens(
            r#"{"input_tokens":1,"generated_tokens":[236772, 236761],"ttft_ms":1.0}"#,
        )
        .expect("tokens");
        assert_eq!(tokens, vec![236772, 236761]);
    }

    #[test]
    fn parses_tsv_prompt_corpus() {
        let dir = std::env::temp_dir().join(format!("gemma4d-bench-test-{}", run_id()));
        fs::create_dir_all(&dir).expect("dir");
        let path = dir.join("corpus.tsv");
        fs::write(
            &path,
            "# id\tprompt\ttoken_ids\tmax_new_tokens\nhello\tHello\t9259,236772\t8\n",
        )
        .expect("write");
        let prompts = load_prompt_corpus(&path).expect("corpus");
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0].id, "hello");
        assert_eq!(prompts[0].token_ids, vec![9259, 236772]);
        assert_eq!(prompts[0].max_new_tokens, 8);
    }

    #[test]
    fn report_labels_inconclusive_records() {
        let dir = std::env::temp_dir().join(format!("gemma4d-report-test-{}", run_id()));
        fs::create_dir_all(&dir).expect("dir");
        let records = dir.join("records.jsonl");
        fs::write(
            &records,
            "{\"prompt_id\":\"hello\",\"reference_name\":\"none\",\"comparison_status\":\"inconclusive\",\"comparison_summary\":\"no reference configured\",\"candidate_command\":\"gemma4d generate\",\"reference_command\":\"\",\"os\":\"test-os\",\"rustc\":\"rustc test\",\"git_commit\":\"abc\",\"model_path\":\"model\",\"config_hash\":\"hash\",\"tokenizer_hash\":\"tokhash\"}\n",
        )
        .expect("write");
        let report = generate_report(&records).expect("report");
        assert!(report.contains("Inconclusive: 1"));
        assert!(report.contains("tokhash"));
        assert!(report.contains("no reference configured"));
    }

    #[test]
    fn configured_command_reference_parses_generated_tokens() {
        let prompt = PromptCase {
            id: "cmd".to_owned(),
            prompt: "Hello".to_owned(),
            token_ids: vec![9259],
            max_new_tokens: 2,
        };
        let result = run_template_reference(
            "llama_cpp",
            "printf '%s' '{\"generated_tokens\":[7,8],\"ttft_ms\":1.5}'",
            Path::new("/tmp/model"),
            &prompt,
        );
        assert_eq!(result.status, "ok");
        assert_eq!(result.generated_tokens, vec![7, 8]);
        assert!(result.command.contains("generated_tokens"));
    }
}
