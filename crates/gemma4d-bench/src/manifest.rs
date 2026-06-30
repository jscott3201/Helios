use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use gemma4d_tokenizer::{file_sha256, sha256_hex};
use serde::{Deserialize, Serialize};

use crate::CliError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestOptions {
    pub model_path: PathBuf,
    pub drafter_path: Option<PathBuf>,
    pub out_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestReport {
    pub schema_version: u32,
    pub goal: String,
    pub timestamp_unix: u64,
    pub git_sha: String,
    pub git_status_short: String,
    pub rust_version: String,
    pub cargo_version: String,
    pub machine: MachineSummary,
    pub mlx: MlxSummary,
    pub model: ArtifactIdentity,
    pub drafter: Option<ArtifactIdentity>,
    pub relevant_environment: BTreeMap<String, Option<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MachineSummary {
    pub uname: String,
    pub macos: String,
    pub arch: String,
    pub hw_memsize_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlxSummary {
    pub python: String,
    pub mlx_version: String,
    pub mlx_lm_version: String,
    pub raw: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactIdentity {
    pub path: String,
    pub exists: bool,
    pub revision: Option<String>,
    pub revision_source: String,
    pub config_sha256: String,
    pub tokenizer_sha256: String,
    pub tokenizer_config_sha256: String,
    pub chat_template_sha256: String,
    pub safetensors_inventory_sha256: String,
    pub safetensors_file_count: usize,
    pub safetensors_total_bytes: u64,
    pub safetensors: Vec<SafetensorsEntry>,
    pub local_artifact_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetensorsEntry {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

pub fn write_manifest_artifacts(options: &ManifestOptions) -> Result<String, CliError> {
    fs::create_dir_all(&options.out_dir).map_err(|error| {
        CliError::Runtime(format!("failed to create manifest out dir: {error}"))
    })?;
    let manifest = capture_manifest(options);
    let manifest_path = options.out_dir.join("manifest.json");
    let report_path = options.out_dir.join("report.md");
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest)
            .map_err(|error| CliError::Runtime(format!("failed to serialize manifest: {error}")))?,
    )
    .map_err(|error| CliError::Runtime(format!("failed to write manifest.json: {error}")))?;
    fs::write(
        &report_path,
        render_manifest_report(&manifest, &manifest_path),
    )
    .map_err(|error| CliError::Runtime(format!("failed to write report.md: {error}")))?;

    Ok(format!(
        "wrote {} and {}",
        manifest_path.display(),
        report_path.display()
    ))
}

pub fn capture_manifest(options: &ManifestOptions) -> ManifestReport {
    ManifestReport {
        schema_version: 1,
        goal: "P11-model-revision-and-manifest-pinning".to_owned(),
        timestamp_unix: unix_now(),
        git_sha: command_stdout("git", &["rev-parse", "HEAD"])
            .unwrap_or_else(|| "unknown".to_owned()),
        git_status_short: command_stdout("git", &["status", "--short"])
            .unwrap_or_else(|| "unknown".to_owned()),
        rust_version: command_stdout("rustc", &["-Vv"]).unwrap_or_else(|| "unknown".to_owned()),
        cargo_version: command_stdout("cargo", &["-V"]).unwrap_or_else(|| "unknown".to_owned()),
        machine: capture_machine_summary(),
        mlx: capture_mlx_summary(),
        model: capture_artifact_identity(&options.model_path, "GEMMA4D_MODEL_REVISION"),
        drafter: options
            .drafter_path
            .as_ref()
            .map(|path| capture_artifact_identity(path, "GEMMA4D_DRAFTER_REVISION")),
        relevant_environment: capture_relevant_environment(),
    }
}

pub fn capture_artifact_identity(path: &Path, revision_env: &str) -> ArtifactIdentity {
    let safetensors = safetensors_inventory(path);
    let revision = env::var(revision_env)
        .ok()
        .filter(|value| !value.trim().is_empty());
    let revision_source = if revision.is_some() {
        format!("env:{revision_env}")
    } else {
        "unavailable:no revision metadata found".to_owned()
    };
    let config_sha256 = file_sha_or_unavailable(&path.join("config.json"));
    let tokenizer_sha256 = file_sha_or_unavailable(&path.join("tokenizer.json"));
    let tokenizer_config_sha256 = file_sha_or_unavailable(&path.join("tokenizer_config.json"));
    let chat_template_sha256 = chat_template_sha(path);
    let local_artifact_sha256 = local_artifact_hash(
        &config_sha256,
        &tokenizer_sha256,
        &tokenizer_config_sha256,
        &chat_template_sha256,
        &safetensors.inventory_sha256,
    );

    ArtifactIdentity {
        path: path.display().to_string(),
        exists: path.exists(),
        revision,
        revision_source,
        config_sha256,
        tokenizer_sha256,
        tokenizer_config_sha256,
        chat_template_sha256,
        safetensors_inventory_sha256: safetensors.inventory_sha256,
        safetensors_file_count: safetensors.entries.len(),
        safetensors_total_bytes: safetensors.total_bytes,
        safetensors: safetensors.entries,
        local_artifact_sha256,
    }
}

pub fn render_manifest_report(manifest: &ManifestReport, manifest_path: &Path) -> String {
    let mut out = String::new();
    out.push_str("# P11 Model Revision and Manifest Pinning\n\n");
    out.push_str("## Summary\n\n");
    out.push_str("| Field | Value |\n|---|---|\n");
    out.push_str(&format!("| Manifest | `{}` |\n", manifest_path.display()));
    out.push_str(&format!(
        "| Git SHA | `{}` |\n",
        escape_md(&manifest.git_sha)
    ));
    out.push_str(&format!(
        "| Git status | `{}` |\n",
        escape_md(&manifest.git_status_short)
    ));
    out.push_str(&format!(
        "| Rust | `{}` |\n",
        escape_md(&single_line(&manifest.rust_version))
    ));
    out.push_str(&format!(
        "| Cargo | `{}` |\n",
        escape_md(&manifest.cargo_version)
    ));
    out.push_str(&format!(
        "| MLX | `{}` / mlx-lm `{}` |\n",
        escape_md(&manifest.mlx.mlx_version),
        escape_md(&manifest.mlx.mlx_lm_version)
    ));
    out.push_str(&format!(
        "| Machine | `{}` |\n\n",
        escape_md(&single_line(&manifest.machine.uname))
    ));

    out.push_str("## Model Artifact Identity\n\n");
    render_artifact_table(&mut out, "Model", &manifest.model);
    if let Some(drafter) = &manifest.drafter {
        render_artifact_table(&mut out, "Drafter", drafter);
    }

    out.push_str("## Relevant Environment\n\n");
    out.push_str("| Variable | Value |\n|---|---|\n");
    for (key, value) in &manifest.relevant_environment {
        out.push_str(&format!(
            "| `{}` | `{}` |\n",
            escape_md(key),
            escape_md(value.as_deref().unwrap_or("unset"))
        ));
    }
    out
}

fn render_artifact_table(out: &mut String, title: &str, artifact: &ArtifactIdentity) {
    out.push_str(&format!("### {title}\n\n"));
    out.push_str("| Field | Value |\n|---|---|\n");
    out.push_str(&format!("| Path | `{}` |\n", escape_md(&artifact.path)));
    out.push_str(&format!("| Exists | `{}` |\n", artifact.exists));
    out.push_str(&format!(
        "| Revision | `{}` |\n",
        escape_md(artifact.revision.as_deref().unwrap_or("unavailable"))
    ));
    out.push_str(&format!(
        "| Revision source | `{}` |\n",
        escape_md(&artifact.revision_source)
    ));
    out.push_str(&format!(
        "| Local artifact SHA-256 | `{}` |\n",
        escape_md(&artifact.local_artifact_sha256)
    ));
    out.push_str(&format!(
        "| Config SHA-256 | `{}` |\n",
        escape_md(&artifact.config_sha256)
    ));
    out.push_str(&format!(
        "| Tokenizer SHA-256 | `{}` |\n",
        escape_md(&artifact.tokenizer_sha256)
    ));
    out.push_str(&format!(
        "| Tokenizer config SHA-256 | `{}` |\n",
        escape_md(&artifact.tokenizer_config_sha256)
    ));
    out.push_str(&format!(
        "| Chat template SHA-256 | `{}` |\n",
        escape_md(&artifact.chat_template_sha256)
    ));
    out.push_str(&format!(
        "| Safetensors inventory SHA-256 | `{}` |\n",
        escape_md(&artifact.safetensors_inventory_sha256)
    ));
    out.push_str(&format!(
        "| Safetensors files | `{}` |\n",
        artifact.safetensors_file_count
    ));
    out.push_str(&format!(
        "| Safetensors bytes | `{}` |\n\n",
        artifact.safetensors_total_bytes
    ));
}

fn capture_machine_summary() -> MachineSummary {
    MachineSummary {
        uname: command_stdout("uname", &["-a"]).unwrap_or_else(|| "unknown".to_owned()),
        macos: command_stdout("sw_vers", &[]).unwrap_or_else(|| "unknown".to_owned()),
        arch: command_stdout("uname", &["-m"]).unwrap_or_else(|| env::consts::ARCH.to_owned()),
        hw_memsize_bytes: command_stdout("sysctl", &["-n", "hw.memsize"])
            .and_then(|value| value.trim().parse::<u64>().ok()),
    }
}

fn capture_mlx_summary() -> MlxSummary {
    let python = env::var("GEMMA4D_MLX_LM_PYTHON")
        .unwrap_or_else(|_| "/opt/homebrew/opt/mlx-lm/libexec/bin/python".to_owned());
    let raw = command_stdout(
        &python,
        &[
            "-c",
            "import json, mlx.core as mx, mlx_lm; print(json.dumps({'mlx': mx.__version__, 'mlx_lm': getattr(mlx_lm, '__version__', 'unknown')}))",
        ],
    )
    .or_else(|| {
        command_stdout(
            "python3",
            &[
                "-c",
                "import json, mlx.core as mx, mlx_lm; print(json.dumps({'mlx': mx.__version__, 'mlx_lm': getattr(mlx_lm, '__version__', 'unknown')}))",
            ],
        )
    })
    .unwrap_or_else(|| "unavailable:mlx import failed".to_owned());
    let value = serde_json::from_str::<serde_json::Value>(&raw).ok();
    MlxSummary {
        python,
        mlx_version: value
            .as_ref()
            .and_then(|value| value.get("mlx"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown")
            .to_owned(),
        mlx_lm_version: value
            .as_ref()
            .and_then(|value| value.get("mlx_lm"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown")
            .to_owned(),
        raw,
    }
}

fn capture_relevant_environment() -> BTreeMap<String, Option<String>> {
    [
        "GEMMA4D_MODEL_REVISION",
        "GEMMA4D_DRAFTER_REVISION",
        "GEMMA4D_MODEL_PATH",
        "GEMMA4D_DRAFTER_MODEL_PATH",
        "GEMMA4D_MLX_LM_PYTHON",
        "GEMMA4D_USE_NATIVE_GRAPH",
        "GEMMA4D_REQUIRE_MLX",
        "GEMMA4D_FULL_MODEL_TESTS",
        "RUSTFLAGS",
    ]
    .into_iter()
    .map(|key| (key.to_owned(), env::var(key).ok()))
    .collect()
}

struct SafetensorsInventory {
    inventory_sha256: String,
    entries: Vec<SafetensorsEntry>,
    total_bytes: u64,
}

fn safetensors_inventory(model_path: &Path) -> SafetensorsInventory {
    let mut entries = Vec::new();
    collect_safetensors(model_path, model_path, &mut entries);
    entries.sort_by(|left, right| left.path.cmp(&right.path));
    let total_bytes = entries.iter().map(|entry| entry.bytes).sum();
    let inventory_body = entries
        .iter()
        .map(|entry| format!("{}\t{}\t{}", entry.path, entry.bytes, entry.sha256))
        .collect::<Vec<_>>()
        .join("\n");
    let inventory_sha256 = if entries.is_empty() {
        "unavailable:no safetensors files found".to_owned()
    } else {
        sha256_hex(inventory_body.as_bytes())
    };
    SafetensorsInventory {
        inventory_sha256,
        entries,
        total_bytes,
    }
}

fn collect_safetensors(root: &Path, current: &Path, entries: &mut Vec<SafetensorsEntry>) {
    let Ok(read_dir) = fs::read_dir(current) else {
        return;
    };
    for entry in read_dir.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_safetensors(root, &path, entries);
            continue;
        }
        if path.extension().and_then(|extension| extension.to_str()) != Some("safetensors") {
            continue;
        }
        let relative = path.strip_prefix(root).unwrap_or(&path);
        let bytes = entry.metadata().map(|metadata| metadata.len()).unwrap_or(0);
        let sha256 = file_sha256(&path).unwrap_or_else(|error| format!("unavailable:{error}"));
        entries.push(SafetensorsEntry {
            path: relative.display().to_string(),
            bytes,
            sha256,
        });
    }
}

fn chat_template_sha(path: &Path) -> String {
    for name in [
        "chat_template.json",
        "chat_template.jinja",
        "tokenizer_config.json",
    ] {
        let candidate = path.join(name);
        if candidate.exists() {
            return file_sha_or_unavailable(&candidate);
        }
    }
    "unavailable:no chat template file found".to_owned()
}

fn local_artifact_hash(
    config_sha256: &str,
    tokenizer_sha256: &str,
    tokenizer_config_sha256: &str,
    chat_template_sha256: &str,
    safetensors_inventory_sha256: &str,
) -> String {
    sha256_hex(
        format!(
            "gemma4d:artifact:v1\nconfig={config_sha256}\ntokenizer={tokenizer_sha256}\ntokenizer_config={tokenizer_config_sha256}\nchat_template={chat_template_sha256}\nsafetensors={safetensors_inventory_sha256}\n"
        )
        .as_bytes(),
    )
}

fn file_sha_or_unavailable(path: &Path) -> String {
    file_sha256(path).unwrap_or_else(|error| format!("unavailable:{}: {error}", path.display()))
}

fn command_stdout(command: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(command).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn single_line(value: &str) -> String {
    value.lines().collect::<Vec<_>>().join(" / ")
}

fn escape_md(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}
