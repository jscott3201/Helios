#![doc = "Local Gemma4D command and future OpenAI-compatible server entrypoints."]

use std::{
    io::Write,
    num::NonZeroU32,
    path::PathBuf,
    process::Command,
    time::{Duration, Instant},
};

use gemma4d_ffi::{self as ffi, KvCache, KvPolicy, LoadConfig, Target};

pub const CRATE_NAME: &str = "gemma4d-server";

pub fn bootstrap_status() -> &'static str {
    "generate-cli"
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerateOptions {
    pub model_path: PathBuf,
    pub prompt: Option<String>,
    pub token_ids: Vec<i32>,
    pub max_new_tokens: usize,
    pub max_context_tokens: NonZeroU32,
    pub output_json: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerateSummary {
    pub input_tokens: usize,
    pub generated_tokens: Vec<i32>,
    pub ttft: Duration,
    pub decode: Duration,
    pub peak_memory_gb: f32,
    pub peak_rss_mb: f32,
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

pub fn run_cli<I, S, W, E>(args: I, stdout: &mut W, stderr: &mut E) -> i32
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
    W: Write,
    E: Write,
{
    match dispatch(args) {
        Ok(output) => {
            let _ = writeln!(stdout, "{output}");
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
        "generate" => {
            let options = parse_generate_options(args)?;
            let output_json = options.output_json;
            let summary = generate(options)?;
            Ok(if output_json {
                summary.to_json()
            } else {
                summary.to_text()
            })
        }
        "-h" | "--help" | "help" => Ok(usage()),
        other => Err(CliError::Usage(format!(
            "unknown command '{other}'\n{}",
            usage()
        ))),
    }
}

pub fn parse_generate_options<I, S>(args: I) -> Result<GenerateOptions, CliError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut args = args.into_iter().map(Into::into).peekable();
    let mut model_path = None;
    let mut prompt = None;
    let mut token_ids = Vec::new();
    let mut max_new_tokens = 16usize;
    let mut max_context_tokens = NonZeroU32::new(8192).expect("non-zero default");
    let mut output_json = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model-path" => {
                model_path = Some(PathBuf::from(required_value(&mut args, "--model-path")?));
            }
            "--prompt" => {
                prompt = Some(required_value(&mut args, "--prompt")?);
            }
            "--token-ids" => {
                token_ids = parse_token_ids(&required_value(&mut args, "--token-ids")?)?;
            }
            "--repeat-token" => {
                let token = parse_token_id(&required_value(&mut args, "--repeat-token")?)?;
                let context_tokens = usize::try_from(max_context_tokens.get())
                    .expect("u32 context length fits usize");
                token_ids = vec![token; context_tokens];
            }
            "--context-tokens" => {
                let value = required_value(&mut args, "--context-tokens")?;
                max_context_tokens = parse_nonzero_u32(&value, "--context-tokens")?;
                if token_ids.len() > 1 && token_ids.iter().all(|token| *token == token_ids[0]) {
                    let token = token_ids[0];
                    let context_tokens =
                        usize::try_from(max_context_tokens.get()).expect("u32 fits usize");
                    token_ids = vec![token; context_tokens];
                }
            }
            "--max-new-tokens" => {
                let value = required_value(&mut args, "--max-new-tokens")?;
                max_new_tokens = parse_positive_usize(&value, "--max-new-tokens")?;
            }
            "--max-context-tokens" => {
                let value = required_value(&mut args, "--max-context-tokens")?;
                max_context_tokens = parse_nonzero_u32(&value, "--max-context-tokens")?;
            }
            "--json" => {
                output_json = true;
            }
            "-h" | "--help" => return Err(CliError::Usage(generate_usage())),
            other => {
                return Err(CliError::Usage(format!(
                    "unknown generate option '{other}'\n{}",
                    generate_usage()
                )));
            }
        }
    }

    let model_path = model_path.ok_or_else(|| {
        CliError::Usage(format!(
            "generate requires --model-path\n{}",
            generate_usage()
        ))
    })?;

    if prompt.is_none() && token_ids.is_empty() {
        return Err(CliError::Usage(format!(
            "generate requires --prompt or --token-ids\n{}",
            generate_usage()
        )));
    }

    Ok(GenerateOptions {
        model_path,
        prompt,
        token_ids,
        max_new_tokens,
        max_context_tokens,
        output_json,
    })
}

pub fn generate(options: GenerateOptions) -> Result<GenerateSummary, CliError> {
    let token_ids = if options.token_ids.is_empty() {
        let prompt = options
            .prompt
            .as_deref()
            .ok_or_else(|| CliError::Usage("generate requires --prompt or token ids".to_owned()))?;
        tokenize_prompt(&options.model_path, prompt)?
    } else {
        options.token_ids.clone()
    };

    let load_config = LoadConfig {
        model_path: options.model_path.display().to_string(),
        model_id: Some("mlx-community/gemma-4-12B-it-4bit".to_owned()),
        model_revision: None,
        expected_architecture: Some("gemma4".to_owned()),
        max_context_tokens: options.max_context_tokens,
        allow_unsupported_config: false,
    };

    let target = Target::load(&load_config)
        .map_err(|error| CliError::Runtime(format!("failed to load target model: {error}")))?;

    let mut cache = KvCache::create(&KvPolicy::default())
        .map_err(|error| CliError::Runtime(format!("failed to create KV cache: {error}")))?;
    let started = Instant::now();
    let mut step = ffi::prefill(&target, &mut cache, &token_ids)
        .map_err(|error| CliError::Runtime(format!("prefill failed: {error}")))?;
    let ttft = started.elapsed();
    let mut peak_memory_gb = step.peak_memory_gb;
    let mut peak_rss_mb = step.peak_rss_mb;

    let mut generated_tokens = Vec::with_capacity(options.max_new_tokens);
    let decode_started = Instant::now();
    for index in 0..options.max_new_tokens {
        generated_tokens.push(step.greedy_token);
        if index + 1 < options.max_new_tokens {
            step = ffi::decode_one(&target, &mut cache, step.greedy_token)
                .map_err(|error| CliError::Runtime(format!("decode failed: {error}")))?;
            peak_memory_gb = peak_memory_gb.max(step.peak_memory_gb);
            peak_rss_mb = peak_rss_mb.max(step.peak_rss_mb);
        }
    }
    let decode = decode_started.elapsed();

    Ok(GenerateSummary {
        input_tokens: token_ids.len(),
        generated_tokens,
        ttft,
        decode,
        peak_memory_gb,
        peak_rss_mb,
    })
}

fn tokenize_prompt(model_path: &PathBuf, prompt: &str) -> Result<Vec<i32>, CliError> {
    let python = std::env::var("GEMMA4D_MLX_LM_PYTHON")
        .unwrap_or_else(|_| "/opt/homebrew/opt/mlx-lm/libexec/bin/python".to_owned());
    let script = r#"
import json
import sys
from pathlib import Path
from mlx_lm.utils import load_tokenizer
tokenizer = load_tokenizer(Path(sys.argv[1]))
print(json.dumps(tokenizer.encode(sys.argv[2]), separators=(",", ":")))
"#;

    let output = Command::new(python)
        .arg("-c")
        .arg(script)
        .arg(model_path)
        .arg(prompt)
        .output()
        .map_err(|error| CliError::Runtime(format!("failed to run tokenizer helper: {error}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(CliError::Runtime(format!(
            "tokenizer helper failed: {}",
            stderr.trim()
        )));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_token_json(stdout.trim())
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

fn parse_positive_usize(value: &str, flag: &str) -> Result<usize, CliError> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| CliError::Usage(format!("{flag} must be a positive integer: {error}")))?;
    if parsed == 0 {
        return Err(CliError::Usage(format!("{flag} must be greater than zero")));
    }
    Ok(parsed)
}

fn parse_nonzero_u32(value: &str, flag: &str) -> Result<NonZeroU32, CliError> {
    let parsed = value
        .parse::<u32>()
        .map_err(|error| CliError::Usage(format!("{flag} must be a positive integer: {error}")))?;
    NonZeroU32::new(parsed)
        .ok_or_else(|| CliError::Usage(format!("{flag} must be greater than zero")))
}

fn parse_token_id(value: &str) -> Result<i32, CliError> {
    let parsed = value
        .parse::<i32>()
        .map_err(|error| CliError::Usage(format!("token id must be an integer: {error}")))?;
    if parsed < 0 {
        return Err(CliError::Usage(format!(
            "token id must be non-negative: {parsed}"
        )));
    }
    Ok(parsed)
}

fn parse_token_ids(value: &str) -> Result<Vec<i32>, CliError> {
    value
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(|token| {
            parse_token_id(token).map_err(|error| {
                CliError::Usage(format!(
                    "--token-ids contains an invalid token '{token}': {error}"
                ))
            })
        })
        .collect()
}

fn parse_token_json(value: &str) -> Result<Vec<i32>, CliError> {
    let trimmed = value.trim();
    let Some(inner) = trimmed
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
    else {
        return Err(CliError::Runtime(format!(
            "tokenizer helper returned invalid token JSON: {trimmed}"
        )));
    };
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }
    parse_token_ids(inner).map_err(|error| {
        CliError::Runtime(format!(
            "tokenizer helper returned invalid token ids: {error}"
        ))
    })
}

fn usage() -> String {
    format!("usage: gemma4d <command>\n\n{}", generate_usage())
}

fn generate_usage() -> String {
    "usage: gemma4d generate --model-path PATH (--prompt TEXT | --token-ids IDS | --repeat-token ID --context-tokens N) [--max-new-tokens N] [--json]".to_owned()
}

impl GenerateSummary {
    fn decode_tps(&self) -> f64 {
        let decode_tokens = self.generated_tokens.len().saturating_sub(1);
        if decode_tokens == 0 || self.decode.is_zero() {
            0.0
        } else {
            decode_tokens as f64 / self.decode.as_secs_f64()
        }
    }

    fn to_text(&self) -> String {
        format!(
            "generated_tokens={:?} input_tokens={} ttft_ms={:.3} decode_ms={:.3} decode_tps={:.3} peak_memory_gb={:.3} peak_rss_mb={:.1}",
            self.generated_tokens,
            self.input_tokens,
            self.ttft.as_secs_f64() * 1000.0,
            self.decode.as_secs_f64() * 1000.0,
            self.decode_tps(),
            self.peak_memory_gb,
            self.peak_rss_mb,
        )
    }

    fn to_json(&self) -> String {
        format!(
            "{{\"input_tokens\":{},\"generated_tokens\":{:?},\"ttft_ms\":{:.3},\"decode_ms\":{:.3},\"decode_tps\":{:.3},\"peak_memory_gb\":{:.3},\"peak_rss_mb\":{:.1}}}",
            self.input_tokens,
            self.generated_tokens,
            self.ttft.as_secs_f64() * 1000.0,
            self.decode.as_secs_f64() * 1000.0,
            self.decode_tps(),
            self.peak_memory_gb,
            self.peak_rss_mb,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_placeholder_status() {
        assert_eq!(CRATE_NAME, "gemma4d-server");
        assert_eq!(bootstrap_status(), "generate-cli");
    }

    #[test]
    fn generate_parse_requires_model_path() {
        let err = parse_generate_options(["--prompt", "hello"]).expect_err("model path required");
        assert_eq!(err.exit_code(), 2);
        assert!(err.to_string().contains("--model-path"));
    }

    #[test]
    fn generate_parse_accepts_token_ids() {
        let options = parse_generate_options([
            "--model-path",
            "/tmp/gemma4d-model",
            "--token-ids",
            "1, 2,3",
            "--max-new-tokens",
            "4",
        ])
        .expect("options");

        assert_eq!(options.model_path, PathBuf::from("/tmp/gemma4d-model"));
        assert_eq!(options.token_ids, vec![1, 2, 3]);
        assert_eq!(options.max_new_tokens, 4);
    }

    #[test]
    fn parses_token_json_from_tokenizer_helper() {
        assert_eq!(
            parse_token_json("[9259,236772]").expect("tokens"),
            vec![9259, 236772]
        );
    }

    #[test]
    fn generate_reports_missing_model_path_gracefully() {
        let options = GenerateOptions {
            model_path: PathBuf::from("/tmp/gemma4d-missing-generate-model-path-for-test"),
            prompt: None,
            token_ids: vec![1, 2, 3],
            max_new_tokens: 1,
            max_context_tokens: NonZeroU32::new(1024).expect("non-zero"),
            output_json: false,
        };

        let err = generate(options).expect_err("missing model path should fail");
        assert_eq!(err.exit_code(), 1);
        assert!(err.to_string().contains("model_path does not exist"));
    }
}
