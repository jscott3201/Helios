#![doc = "Local Gemma4D command and OpenAI-compatible server entrypoints."]

pub mod http;

use std::{
    env,
    io::Write,
    num::NonZeroU32,
    path::PathBuf,
    process::Command,
    time::{Duration, Instant},
};

use gemma4d_adapters::{
    AdapterCompatibility, AdapterRegistry, AdapterSummary, ImportedAdapter, TrustedPathPolicy,
};
use gemma4d_ffi::{self as ffi, KvCache, KvPolicy, LoadConfig, Target};
use http::{
    PERSISTENT_NATIVE_GATE_ENV, ServerBackend, ServerConfig, parse_bind_addr, serve_blocking,
};

pub const CRATE_NAME: &str = "gemma4d-server";

pub fn bootstrap_status() -> &'static str {
    "m11-openai-server"
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
    pub generated_logits: Vec<f32>,
    pub model_load: Duration,
    pub ttft: Duration,
    pub prefill: Duration,
    pub decode: Duration,
    pub total: Duration,
    pub decode_token_latencies: Vec<Duration>,
    pub peak_memory_gb: f32,
    pub peak_rss_mb: f32,
    pub active_kv_bytes: u64,
}

#[derive(Debug)]
pub struct ResidentTarget {
    model_path: PathBuf,
    target: Target,
    model_load: Duration,
}

impl ResidentTarget {
    pub fn load(model_path: PathBuf, max_context_tokens: NonZeroU32) -> Result<Self, CliError> {
        let load_config = LoadConfig {
            model_path: model_path.display().to_string(),
            model_id: Some("mlx-community/gemma-4-12B-it-4bit".to_owned()),
            model_revision: None,
            expected_architecture: Some("gemma4".to_owned()),
            max_context_tokens,
            allow_unsupported_config: false,
        };
        let model_load_started = Instant::now();
        let target = Target::load(&load_config)
            .map_err(|error| CliError::Runtime(format!("failed to load target model: {error}")))?;
        Ok(Self {
            model_path,
            target,
            model_load: model_load_started.elapsed(),
        })
    }

    pub fn model_load(&self) -> Duration {
        self.model_load
    }

    pub fn generate_prompt(
        &self,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<GenerateSummary, CliError> {
        let total_started = Instant::now();
        let token_ids = tokenize_prompt(&self.model_path, prompt)?;
        generate_with_target(
            total_started,
            &self.target,
            token_ids,
            max_new_tokens,
            Duration::ZERO,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterCommand {
    Import(AdapterImportOptions),
    Load(AdapterIdOptions),
    Unload(AdapterIdOptions),
    Pin(AdapterIdOptions),
    List(AdapterListOptions),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterImportOptions {
    pub registry_dir: PathBuf,
    pub trusted_root: PathBuf,
    pub source: PathBuf,
    pub compatibility: AdapterCompatibility,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterIdOptions {
    pub registry_dir: PathBuf,
    pub adapter_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterListOptions {
    pub registry_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServeOptions {
    pub bind_addr: std::net::SocketAddr,
    pub backend: ServerBackend,
    pub model_path: Option<PathBuf>,
    pub max_context_tokens: usize,
    pub memory_budget_mb: u64,
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
        "adapter" => {
            let command = parse_adapter_command(args)?;
            run_adapter_command(command)
        }
        "serve" => {
            let options = parse_serve_options(args)?;
            serve(options)?;
            Ok("server stopped".to_owned())
        }
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

pub fn parse_serve_options<I, S>(args: I) -> Result<ServeOptions, CliError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut args = args.into_iter().map(Into::into).peekable();
    let mut bind_addr = ServerConfig::default().bind_addr;
    let mut backend = ServerConfig::default().backend;
    let mut model_path = None;
    let mut max_context_tokens = ServerConfig::default().max_context_tokens;
    let mut memory_budget_mb = ServerConfig::default().memory_budget_bytes / (1024 * 1024);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bind" => {
                let value = required_value(&mut args, "--bind")?;
                bind_addr = parse_bind_addr(&value).map_err(CliError::Usage)?;
            }
            "--backend" => {
                let value = required_value(&mut args, "--backend")?;
                backend = parse_server_backend(&value)?;
            }
            "--real-helper" => {
                backend = ServerBackend::RealHelper;
            }
            "--persistent-native" => {
                backend = ServerBackend::PersistentNative;
            }
            "--model-path" => {
                model_path = Some(PathBuf::from(required_value(&mut args, "--model-path")?));
            }
            "--max-context-tokens" => {
                max_context_tokens = parse_positive_usize(
                    &required_value(&mut args, "--max-context-tokens")?,
                    "--max-context-tokens",
                )?;
            }
            "--memory-budget-mb" => {
                memory_budget_mb = required_value(&mut args, "--memory-budget-mb")?
                    .parse::<u64>()
                    .map_err(|error| {
                        CliError::Usage(format!("--memory-budget-mb must be an integer: {error}"))
                    })?;
                if memory_budget_mb == 0 {
                    return Err(CliError::Usage(
                        "--memory-budget-mb must be greater than zero".to_owned(),
                    ));
                }
            }
            "-h" | "--help" => return Err(CliError::Usage(serve_usage())),
            other => {
                return Err(CliError::Usage(format!(
                    "unknown serve option '{other}'\n{}",
                    serve_usage()
                )));
            }
        }
    }

    match backend {
        ServerBackend::Stub if model_path.is_some() => {
            return Err(CliError::Usage(
                "--model-path requires --backend real-helper or persistent-native".to_owned(),
            ));
        }
        ServerBackend::RealHelper | ServerBackend::PersistentNative if model_path.is_none() => {
            return Err(CliError::Usage(format!(
                "serve --backend {} requires --model-path",
                backend.cli_name()
            )));
        }
        _ => {}
    }

    if backend == ServerBackend::PersistentNative
        && env::var(PERSISTENT_NATIVE_GATE_ENV).ok().as_deref() != Some("1")
    {
        return Err(CliError::Usage(format!(
            "serve --backend persistent-native requires {PERSISTENT_NATIVE_GATE_ENV}=1"
        )));
    }

    Ok(ServeOptions {
        bind_addr,
        backend,
        model_path,
        max_context_tokens,
        memory_budget_mb,
    })
}

pub fn serve(options: ServeOptions) -> Result<(), CliError> {
    let mut config = ServerConfig::localhost_default().with_bind_addr(options.bind_addr);
    config.backend = options.backend;
    config.model_path = options.model_path;
    config.max_context_tokens = options.max_context_tokens;
    config.memory_budget_bytes = options.memory_budget_mb.saturating_mul(1024 * 1024);
    serve_blocking(config)
        .map_err(|error| CliError::Runtime(format!("OpenAI-compatible server failed: {error}")))
}

pub fn parse_adapter_command<I, S>(args: I) -> Result<AdapterCommand, CliError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut args = args.into_iter().map(Into::into);
    let Some(command) = args.next() else {
        return Err(CliError::Usage(adapter_usage()));
    };
    match command.as_str() {
        "import" => parse_adapter_import_options(args).map(AdapterCommand::Import),
        "load" => parse_adapter_id_options(args).map(AdapterCommand::Load),
        "unload" => parse_adapter_id_options(args).map(AdapterCommand::Unload),
        "pin" => parse_adapter_id_options(args).map(AdapterCommand::Pin),
        "list" => parse_adapter_list_options(args).map(AdapterCommand::List),
        "-h" | "--help" | "help" => Err(CliError::Usage(adapter_usage())),
        other => Err(CliError::Usage(format!(
            "unknown adapter command '{other}'\n{}",
            adapter_usage()
        ))),
    }
}

fn parse_adapter_import_options<I>(args: I) -> Result<AdapterImportOptions, CliError>
where
    I: Iterator<Item = String>,
{
    let mut args = args.peekable();
    let mut registry_dir = None;
    let mut trusted_root = None;
    let mut source = None;
    let mut base_model_id = None;
    let mut base_weight_hash = None;
    let mut tokenizer_hash = None;
    let mut chat_template_hash = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--registry-dir" => {
                registry_dir = Some(PathBuf::from(required_value(&mut args, "--registry-dir")?));
            }
            "--trusted-root" => {
                trusted_root = Some(PathBuf::from(required_value(&mut args, "--trusted-root")?));
            }
            "--source" => {
                source = Some(PathBuf::from(required_value(&mut args, "--source")?));
            }
            "--base-model-id" => {
                base_model_id = Some(required_value(&mut args, "--base-model-id")?);
            }
            "--base-weight-hash" => {
                base_weight_hash = Some(required_value(&mut args, "--base-weight-hash")?);
            }
            "--tokenizer-hash" => {
                tokenizer_hash = Some(required_value(&mut args, "--tokenizer-hash")?);
            }
            "--chat-template-hash" => {
                chat_template_hash = Some(required_value(&mut args, "--chat-template-hash")?);
            }
            "-h" | "--help" => return Err(CliError::Usage(adapter_import_usage())),
            other => {
                return Err(CliError::Usage(format!(
                    "unknown adapter import option '{other}'\n{}",
                    adapter_import_usage()
                )));
            }
        }
    }

    Ok(AdapterImportOptions {
        registry_dir: required_path(registry_dir, "--registry-dir", adapter_import_usage)?,
        trusted_root: required_path(trusted_root, "--trusted-root", adapter_import_usage)?,
        source: required_path(source, "--source", adapter_import_usage)?,
        compatibility: AdapterCompatibility {
            base_model_id: required_string(base_model_id, "--base-model-id", adapter_import_usage)?,
            base_weight_hash: required_string(
                base_weight_hash,
                "--base-weight-hash",
                adapter_import_usage,
            )?,
            tokenizer_hash: required_string(
                tokenizer_hash,
                "--tokenizer-hash",
                adapter_import_usage,
            )?,
            chat_template_hash: required_string(
                chat_template_hash,
                "--chat-template-hash",
                adapter_import_usage,
            )?,
        },
    })
}

fn parse_adapter_id_options<I>(args: I) -> Result<AdapterIdOptions, CliError>
where
    I: Iterator<Item = String>,
{
    let mut args = args.peekable();
    let mut registry_dir = None;
    let mut adapter_id = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--registry-dir" => {
                registry_dir = Some(PathBuf::from(required_value(&mut args, "--registry-dir")?));
            }
            "--adapter" => {
                adapter_id = Some(required_value(&mut args, "--adapter")?);
            }
            "-h" | "--help" => return Err(CliError::Usage(adapter_id_usage())),
            other => {
                return Err(CliError::Usage(format!(
                    "unknown adapter option '{other}'\n{}",
                    adapter_id_usage()
                )));
            }
        }
    }

    Ok(AdapterIdOptions {
        registry_dir: required_path(registry_dir, "--registry-dir", adapter_id_usage)?,
        adapter_id: required_string(adapter_id, "--adapter", adapter_id_usage)?,
    })
}

fn parse_adapter_list_options<I>(args: I) -> Result<AdapterListOptions, CliError>
where
    I: Iterator<Item = String>,
{
    let mut args = args.peekable();
    let mut registry_dir = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--registry-dir" => {
                registry_dir = Some(PathBuf::from(required_value(&mut args, "--registry-dir")?));
            }
            "--trusted-root" => {
                let _ = required_value(&mut args, "--trusted-root")?;
            }
            "-h" | "--help" => return Err(CliError::Usage(adapter_list_usage())),
            other => {
                return Err(CliError::Usage(format!(
                    "unknown adapter list option '{other}'\n{}",
                    adapter_list_usage()
                )));
            }
        }
    }

    Ok(AdapterListOptions {
        registry_dir: required_path(registry_dir, "--registry-dir", adapter_list_usage)?,
    })
}

pub fn run_adapter_command(command: AdapterCommand) -> Result<String, CliError> {
    match command {
        AdapterCommand::Import(options) => {
            let trusted = TrustedPathPolicy::new(&options.trusted_root)
                .map_err(|error| CliError::Runtime(format!("trusted root rejected: {error}")))?;
            let mut registry = AdapterRegistry::open(&options.registry_dir)
                .map_err(|error| CliError::Runtime(format!("registry open failed: {error}")))?;
            let imported = registry
                .import_peft(&options.source, &trusted, &options.compatibility)
                .map_err(|error| CliError::Runtime(format!("adapter import failed: {error}")))?;
            Ok(imported_adapter_to_text(&imported))
        }
        AdapterCommand::Load(options) => {
            let mut registry = AdapterRegistry::open(&options.registry_dir)
                .map_err(|error| CliError::Runtime(format!("registry open failed: {error}")))?;
            let summary = registry
                .load(&options.adapter_id)
                .map_err(|error| CliError::Runtime(format!("adapter load failed: {error}")))?;
            Ok(summary_to_text("adapter_loaded", &summary))
        }
        AdapterCommand::Unload(options) => {
            let mut registry = AdapterRegistry::open(&options.registry_dir)
                .map_err(|error| CliError::Runtime(format!("registry open failed: {error}")))?;
            let summary = registry
                .unload(&options.adapter_id)
                .map_err(|error| CliError::Runtime(format!("adapter unload failed: {error}")))?;
            Ok(summary_to_text("adapter_unloaded", &summary))
        }
        AdapterCommand::Pin(options) => {
            let mut registry = AdapterRegistry::open(&options.registry_dir)
                .map_err(|error| CliError::Runtime(format!("registry open failed: {error}")))?;
            let summary = registry
                .pin(&options.adapter_id)
                .map_err(|error| CliError::Runtime(format!("adapter pin failed: {error}")))?;
            Ok(summary_to_text("adapter_pinned", &summary))
        }
        AdapterCommand::List(options) => {
            let registry = AdapterRegistry::open(&options.registry_dir)
                .map_err(|error| CliError::Runtime(format!("registry open failed: {error}")))?;
            let summaries = registry.summaries();
            if summaries.is_empty() {
                Ok("adapters=[] total_resident_bytes=0".to_owned())
            } else {
                Ok(format!(
                    "adapters=[{}] total_resident_bytes={}",
                    summaries
                        .iter()
                        .map(|summary| summary_to_text("adapter", summary))
                        .collect::<Vec<_>>()
                        .join("; "),
                    registry.total_resident_bytes()
                ))
            }
        }
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
    let total_started = Instant::now();
    let token_ids = if options.token_ids.is_empty() {
        let prompt = options
            .prompt
            .as_deref()
            .ok_or_else(|| CliError::Usage("generate requires --prompt or token ids".to_owned()))?;
        tokenize_prompt(&options.model_path, prompt)?
    } else {
        options.token_ids.clone()
    };

    let resident = ResidentTarget::load(options.model_path, options.max_context_tokens)?;
    let model_load = resident.model_load();

    generate_with_target(
        total_started,
        &resident.target,
        token_ids,
        options.max_new_tokens,
        model_load,
    )
}

fn generate_with_target(
    total_started: Instant,
    target: &Target,
    token_ids: Vec<i32>,
    max_new_tokens: usize,
    model_load: Duration,
) -> Result<GenerateSummary, CliError> {
    let mut cache = KvCache::create(&KvPolicy::default())
        .map_err(|error| CliError::Runtime(format!("failed to create KV cache: {error}")))?;
    let prefill_started = Instant::now();
    let mut step = ffi::prefill(target, &mut cache, &token_ids)
        .map_err(|error| CliError::Runtime(format!("prefill failed: {error}")))?;
    let prefill = prefill_started.elapsed();
    let ttft = prefill;
    let mut peak_memory_gb = step.peak_memory_gb;
    let mut peak_rss_mb = step.peak_rss_mb;
    let mut active_kv_bytes = step.active_kv_bytes;

    let mut generated_tokens = Vec::with_capacity(max_new_tokens);
    let mut generated_logits = Vec::with_capacity(max_new_tokens);
    let mut decode_token_latencies = Vec::with_capacity(max_new_tokens.saturating_sub(1));
    let decode_started = Instant::now();
    for index in 0..max_new_tokens {
        generated_tokens.push(step.greedy_token);
        generated_logits.push(step.greedy_logit);
        if index + 1 < max_new_tokens {
            let token_started = Instant::now();
            step = ffi::decode_one(target, &mut cache, step.greedy_token)
                .map_err(|error| CliError::Runtime(format!("decode failed: {error}")))?;
            decode_token_latencies.push(token_started.elapsed());
            peak_memory_gb = peak_memory_gb.max(step.peak_memory_gb);
            peak_rss_mb = peak_rss_mb.max(step.peak_rss_mb);
            active_kv_bytes = active_kv_bytes.max(step.active_kv_bytes);
        }
    }
    let decode = decode_started.elapsed();
    let total = total_started.elapsed();

    Ok(GenerateSummary {
        input_tokens: token_ids.len(),
        generated_tokens,
        generated_logits,
        model_load,
        ttft,
        prefill,
        decode,
        total,
        decode_token_latencies,
        peak_memory_gb,
        peak_rss_mb,
        active_kv_bytes,
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

pub fn detokenize_tokens(model_path: &PathBuf, tokens: &[i32]) -> Result<String, CliError> {
    let python = std::env::var("GEMMA4D_MLX_LM_PYTHON")
        .unwrap_or_else(|_| "/opt/homebrew/opt/mlx-lm/libexec/bin/python".to_owned());
    let token_json = format!(
        "[{}]",
        tokens
            .iter()
            .map(i32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    let script = r#"
import json
import sys
from pathlib import Path
from mlx_lm.utils import load_tokenizer
tokenizer = load_tokenizer(Path(sys.argv[1]))
print(tokenizer.decode(json.loads(sys.argv[2])), end="")
"#;

    let output = Command::new(python)
        .arg("-c")
        .arg(script)
        .arg(model_path)
        .arg(token_json)
        .output()
        .map_err(|error| CliError::Runtime(format!("failed to run detokenizer helper: {error}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(CliError::Runtime(format!(
            "detokenizer helper failed: {}",
            stderr.trim()
        )));
    }

    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
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

fn required_path(
    value: Option<PathBuf>,
    flag: &str,
    usage: fn() -> String,
) -> Result<PathBuf, CliError> {
    value.ok_or_else(|| CliError::Usage(format!("adapter command requires {flag}\n{}", usage())))
}

fn required_string(
    value: Option<String>,
    flag: &str,
    usage: fn() -> String,
) -> Result<String, CliError> {
    value.ok_or_else(|| CliError::Usage(format!("adapter command requires {flag}\n{}", usage())))
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

fn parse_server_backend(value: &str) -> Result<ServerBackend, CliError> {
    match value {
        "stub" => Ok(ServerBackend::Stub),
        "real-helper" | "real_helper" => Ok(ServerBackend::RealHelper),
        "persistent-native" | "persistent_native" => Ok(ServerBackend::PersistentNative),
        other => Err(CliError::Usage(format!(
            "--backend must be stub, real-helper, or persistent-native, got '{other}'"
        ))),
    }
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
    format!(
        "usage: gemma4d <command>\n\n{}\n\n{}\n\n{}",
        serve_usage(),
        generate_usage(),
        adapter_usage()
    )
}

fn serve_usage() -> String {
    "usage: gemma4d serve [--bind 127.0.0.1:8080] [--backend stub|real-helper|persistent-native --model-path PATH] [--max-context-tokens N] [--memory-budget-mb N]"
        .to_owned()
}

fn generate_usage() -> String {
    "usage: gemma4d generate --model-path PATH (--prompt TEXT | --token-ids IDS | --repeat-token ID --context-tokens N) [--max-new-tokens N] [--json]".to_owned()
}

fn adapter_usage() -> String {
    format!(
        "usage: gemma4d adapter <import|load|unload|pin|list>\n{}\n{}\n{}",
        adapter_import_usage(),
        adapter_id_usage(),
        adapter_list_usage()
    )
}

fn adapter_import_usage() -> String {
    "usage: gemma4d adapter import --registry-dir PATH --trusted-root PATH --source PATH --base-model-id ID --base-weight-hash HASH --tokenizer-hash HASH --chat-template-hash HASH".to_owned()
}

fn adapter_id_usage() -> String {
    "usage: gemma4d adapter (load|unload|pin) --registry-dir PATH --adapter ID".to_owned()
}

fn adapter_list_usage() -> String {
    "usage: gemma4d adapter list --registry-dir PATH".to_owned()
}

fn imported_adapter_to_text(imported: &ImportedAdapter) -> String {
    format!(
        "adapter_imported id={} loaded=true tensors={} resident_bytes={} load_latency_us={} shape_validation={}",
        imported.manifest.adapter_id,
        imported.validation.tensor_count,
        imported.validation.resident_bytes,
        imported.load_latency_us,
        imported.validation.shape_validation_result,
    )
}

fn summary_to_text(label: &str, summary: &AdapterSummary) -> String {
    format!(
        "{label} id={} loaded={} pinned={} active={} resident_bytes={} load_latency_us={} supports_mtp={:?}",
        summary.adapter_id,
        summary.loaded,
        summary.pinned,
        summary.active,
        summary.resident_bytes,
        summary.load_latency_us,
        summary.supports_mtp,
    )
}

impl GenerateSummary {
    fn model_load_ms(&self) -> f64 {
        self.model_load.as_secs_f64() * 1000.0
    }

    fn prefill_ms(&self) -> f64 {
        self.prefill.as_secs_f64() * 1000.0
    }

    fn ttft_ms(&self) -> f64 {
        self.ttft.as_secs_f64() * 1000.0
    }

    fn decode_ms(&self) -> f64 {
        self.decode.as_secs_f64() * 1000.0
    }

    fn total_ms(&self) -> f64 {
        self.total.as_secs_f64() * 1000.0
    }

    fn decode_tps(&self) -> f64 {
        let decode_tokens = self.generated_tokens.len().saturating_sub(1);
        if decode_tokens == 0 || self.decode.is_zero() {
            0.0
        } else {
            decode_tokens as f64 / self.decode.as_secs_f64()
        }
    }

    fn decode_latency_ms_json(&self) -> String {
        let values = self
            .decode_token_latencies
            .iter()
            .map(|latency| format!("{:.3}", latency.as_secs_f64() * 1000.0))
            .collect::<Vec<_>>();
        format!("[{}]", values.join(","))
    }

    fn to_text(&self) -> String {
        format!(
            "generated_tokens={:?} input_tokens={} model_load_ms={:.3} prefill_ms={:.3} ttft_ms={:.3} decode_ms={:.3} total_ms={:.3} decode_tps={:.3} peak_memory_gb={:.3} peak_rss_mb={:.1} active_kv_bytes={}",
            self.generated_tokens,
            self.input_tokens,
            self.model_load_ms(),
            self.prefill_ms(),
            self.ttft_ms(),
            self.decode_ms(),
            self.total_ms(),
            self.decode_tps(),
            self.peak_memory_gb,
            self.peak_rss_mb,
            self.active_kv_bytes,
        )
    }

    fn to_json(&self) -> String {
        format!(
            "{{\"input_tokens\":{},\"generated_tokens\":{:?},\"generated_logits\":{:?},\"model_load_ms\":{:.3},\"prefill_ms\":{:.3},\"ttft_ms\":{:.3},\"decode_ms\":{:.3},\"total_ms\":{:.3},\"decode_tps\":{:.3},\"decode_token_latencies_ms\":{},\"mlx_active_memory_gb\":null,\"mlx_cache_memory_gb\":null,\"peak_memory_gb\":{:.3},\"peak_rss_mb\":{:.1},\"active_kv_bytes\":{}}}",
            self.input_tokens,
            self.generated_tokens,
            self.generated_logits,
            self.model_load_ms(),
            self.prefill_ms(),
            self.ttft_ms(),
            self.decode_ms(),
            self.total_ms(),
            self.decode_tps(),
            self.decode_latency_ms_json(),
            self.peak_memory_gb,
            self.peak_rss_mb,
            self.active_kv_bytes,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_placeholder_status() {
        assert_eq!(CRATE_NAME, "gemma4d-server");
        assert_eq!(bootstrap_status(), "m11-openai-server");
    }

    #[test]
    fn serve_parse_defaults_to_localhost() {
        let options = parse_serve_options(std::iter::empty::<&str>()).expect("serve options");
        assert_eq!(options.bind_addr.to_string(), "127.0.0.1:8080");
        assert_eq!(options.backend, ServerBackend::Stub);
        assert_eq!(options.model_path, None);
        assert_eq!(options.max_context_tokens, 32_768);
        assert!(options.memory_budget_mb > 0);
    }

    #[test]
    fn serve_parse_accepts_real_helper_backend() {
        let options = parse_serve_options([
            "--backend",
            "real-helper",
            "--model-path",
            "/tmp/gemma4d-model",
        ])
        .expect("serve options");

        assert_eq!(options.backend, ServerBackend::RealHelper);
        assert_eq!(
            options.model_path,
            Some(PathBuf::from("/tmp/gemma4d-model"))
        );
    }

    #[test]
    fn serve_parse_requires_model_path_for_real_helper() {
        let err =
            parse_serve_options(["--backend", "real-helper"]).expect_err("model path required");
        assert_eq!(err.exit_code(), 2);
        assert!(err.to_string().contains("--model-path"));
    }

    #[test]
    fn serve_parse_rejects_non_localhost_bind() {
        let err = parse_serve_options(["--bind", "0.0.0.0:8080"]).expect_err("non-local bind");
        assert_eq!(err.exit_code(), 2);
        assert!(err.to_string().contains("localhost"));
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

    #[test]
    fn adapter_import_load_unload_pin_and_list_commands_work() {
        let fixture = AdapterCliFixture::new("adapter-cli");
        fixture.write_adapter("rust-coding-r16-v1");

        let import = dispatch([
            "adapter",
            "import",
            "--registry-dir",
            fixture.registry_dir.to_str().expect("utf8"),
            "--trusted-root",
            fixture.trusted_root.to_str().expect("utf8"),
            "--source",
            fixture.adapter_dir.to_str().expect("utf8"),
            "--base-model-id",
            "mlx-community/gemma-4-12B-it-4bit",
            "--base-weight-hash",
            "base-weight-hash",
            "--tokenizer-hash",
            "tokenizer-hash",
            "--chat-template-hash",
            "chat-template-hash",
        ])
        .expect("import");
        assert!(import.contains("adapter_imported id=rust-coding-r16-v1"));
        assert!(import.contains("loaded=true"));

        let pin = dispatch([
            "adapter",
            "pin",
            "--registry-dir",
            fixture.registry_dir.to_str().expect("utf8"),
            "--adapter",
            "rust-coding-r16-v1",
        ])
        .expect("pin");
        assert!(pin.contains("pinned=true"));

        let unload = dispatch([
            "adapter",
            "unload",
            "--registry-dir",
            fixture.registry_dir.to_str().expect("utf8"),
            "--adapter",
            "rust-coding-r16-v1",
        ])
        .expect("unload");
        assert!(unload.contains("loaded=false"));

        let load = dispatch([
            "adapter",
            "load",
            "--registry-dir",
            fixture.registry_dir.to_str().expect("utf8"),
            "--adapter",
            "rust-coding-r16-v1",
        ])
        .expect("load");
        assert!(load.contains("loaded=true"));

        let list = dispatch([
            "adapter",
            "list",
            "--registry-dir",
            fixture.registry_dir.to_str().expect("utf8"),
        ])
        .expect("list");
        assert!(list.contains("rust-coding-r16-v1"));
        assert!(list.contains("pinned=true"));
    }

    #[test]
    fn adapter_import_rejects_untrusted_source() {
        let fixture = AdapterCliFixture::new("adapter-cli-untrusted");
        fixture.write_adapter("rust-coding-r16-v1");
        let outside = fixture.root.join("outside");
        std::fs::create_dir_all(&outside).expect("outside");
        let err = dispatch([
            "adapter",
            "import",
            "--registry-dir",
            fixture.registry_dir.to_str().expect("utf8"),
            "--trusted-root",
            outside.to_str().expect("utf8"),
            "--source",
            fixture.adapter_dir.to_str().expect("utf8"),
            "--base-model-id",
            "mlx-community/gemma-4-12B-it-4bit",
            "--base-weight-hash",
            "base-weight-hash",
            "--tokenizer-hash",
            "tokenizer-hash",
            "--chat-template-hash",
            "chat-template-hash",
        ])
        .expect_err("untrusted path rejected");
        assert_eq!(err.exit_code(), 1);
        assert!(err.to_string().contains("outside trusted root"));
    }

    #[test]
    fn adapter_parse_requires_registry_dir() {
        let err = parse_adapter_command(["list"]).expect_err("registry required");
        assert_eq!(err.exit_code(), 2);
        assert!(err.to_string().contains("--registry-dir"));
    }

    struct AdapterCliFixture {
        root: PathBuf,
        trusted_root: PathBuf,
        registry_dir: PathBuf,
        adapter_dir: PathBuf,
    }

    impl AdapterCliFixture {
        fn new(name: &str) -> Self {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos();
            let root = std::env::temp_dir().join(format!("gemma4d-server-{name}-{nonce}"));
            let trusted_root = root.join("trusted");
            let registry_dir = root.join("registry");
            let adapter_dir = trusted_root.join("rust-coding-r16-v1");
            std::fs::create_dir_all(&adapter_dir).expect("adapter dir");
            std::fs::create_dir_all(&registry_dir).expect("registry dir");
            Self {
                root,
                trusted_root,
                registry_dir,
                adapter_dir,
            }
        }

        fn write_adapter(&self, adapter_id: &str) {
            let raw = format!(
                r#"{{
  "peft_type": "LORA",
  "base_model_name_or_path": "mlx-community/gemma-4-12B-it-4bit",
  "r": 16,
  "lora_alpha": 32.0,
  "lora_dropout": 0.05,
  "target_modules": ["q_proj", "v_proj"],
  "gemma4d": {{
    "adapter_id": "{adapter_id}",
    "base_model_id": "mlx-community/gemma-4-12B-it-4bit",
    "base_weight_hash": "base-weight-hash",
    "tokenizer_hash": "tokenizer-hash",
    "chat_template_hash": "chat-template-hash",
    "adapter_type": "lora",
    "dtype": "bf16",
    "supports_mtp": "unknown"
  }}
}}"#
            );
            std::fs::write(self.adapter_dir.join("adapter_config.json"), raw)
                .expect("adapter config");
            write_safetensors(&self.adapter_dir.join("adapter_model.safetensors"));
        }
    }

    fn write_safetensors(path: &std::path::Path) {
        let header = serde_json::json!({
            "__metadata__": {"format": "pt"},
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": {
                "dtype": "F32",
                "shape": [16, 8],
                "data_offsets": [0, 512]
            },
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": {
                "dtype": "F32",
                "shape": [8, 16],
                "data_offsets": [512, 1024]
            },
            "base_model.model.layers.0.self_attn.v_proj.lora_A.weight": {
                "dtype": "F32",
                "shape": [16, 8],
                "data_offsets": [1024, 1536]
            },
            "base_model.model.layers.0.self_attn.v_proj.lora_B.weight": {
                "dtype": "F32",
                "shape": [8, 16],
                "data_offsets": [1536, 2048]
            }
        });
        let header = serde_json::to_vec(&header).expect("header");
        let mut bytes = Vec::with_capacity(8 + header.len() + 2048);
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend(vec![0u8; 2048]);
        std::fs::write(path, bytes).expect("safetensors");
    }
}
