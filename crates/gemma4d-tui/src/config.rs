use std::{fs, path::Path};

use serde::{Deserialize, Serialize};

use crate::TuiError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationStatus {
    Pending,
    Valid,
    Invalid,
}

impl ValidationStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Valid => "valid",
            Self::Invalid => "invalid",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    Info,
    Warning,
    Error,
}

impl DiagnosticSeverity {
    pub fn label(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Error => "error",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfigDiagnostic {
    pub severity: DiagnosticSeverity,
    pub path: String,
    pub message: String,
}

impl ConfigDiagnostic {
    fn error(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity: DiagnosticSeverity::Error,
            path: path.into(),
            message: message.into(),
        }
    }

    fn warning(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity: DiagnosticSeverity::Warning,
            path: path.into(),
            message: message.into(),
        }
    }

    fn info(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity: DiagnosticSeverity::Info,
            path: path.into(),
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfigValidation {
    pub path: std::path::PathBuf,
    pub status: ValidationStatus,
    pub summary: String,
    pub diagnostics: Vec<ConfigDiagnostic>,
}

impl ConfigValidation {
    pub fn pending(path: std::path::PathBuf) -> Self {
        Self {
            path,
            status: ValidationStatus::Pending,
            summary: "config validation pending".to_owned(),
            diagnostics: Vec::new(),
        }
    }

    fn invalid(path: &Path, diagnostics: Vec<ConfigDiagnostic>) -> Self {
        Self {
            path: path.to_path_buf(),
            status: ValidationStatus::Invalid,
            summary: format!(
                "{} invalid: {} diagnostic(s)",
                path.display(),
                diagnostics.len()
            ),
            diagnostics,
        }
    }

    fn valid(path: &Path, diagnostics: Vec<ConfigDiagnostic>) -> Self {
        Self {
            path: path.to_path_buf(),
            status: ValidationStatus::Valid,
            summary: format!(
                "{} valid: {} diagnostic(s)",
                path.display(),
                diagnostics.len()
            ),
            diagnostics,
        }
    }
}

pub fn validate_config_path(path: &Path) -> ConfigValidation {
    let source = match fs::read_to_string(path) {
        Ok(source) => source,
        Err(error) => {
            return ConfigValidation::invalid(
                path,
                vec![ConfigDiagnostic::error(
                    "$",
                    format!("could not read config: {error}"),
                )],
            );
        }
    };

    validate_config_source(path, &source)
}

pub fn validate_config_source(path: &Path, source: &str) -> ConfigValidation {
    let value = match toml::from_str::<toml::Value>(source) {
        Ok(value) => value,
        Err(error) => {
            return ConfigValidation::invalid(
                path,
                vec![ConfigDiagnostic::error(
                    "$",
                    format!("TOML parse error: {error}"),
                )],
            );
        }
    };

    let Some(root) = value.as_table() else {
        return ConfigValidation::invalid(
            path,
            vec![ConfigDiagnostic::error(
                "$",
                "config root must be a TOML table",
            )],
        );
    };

    let mut diagnostics = Vec::new();
    if !root.contains_key("tui") {
        diagnostics.push(ConfigDiagnostic::error(
            "[tui]",
            "missing TUI settings table",
        ));
    }

    if let Some(tui) = table_at(root, "tui") {
        validate_tui_table(tui, &mut diagnostics);
    }

    if let Some(runtime) = table_at(root, "runtime") {
        validate_runtime_table(runtime, &mut diagnostics);
    } else if root.contains_key("model") || root.contains_key("kv") || root.contains_key("adapters")
    {
        diagnostics.push(ConfigDiagnostic::error(
            "[runtime]",
            "tiny16-style configs must include runtime limits",
        ));
    }

    if let Some(model) = table_at(root, "model") {
        validate_model_table(model, &mut diagnostics);
    }

    if !root.contains_key("runtime") {
        diagnostics.push(ConfigDiagnostic::info(
            "[runtime]",
            "runtime table absent; treating config as TUI-only",
        ));
    }

    if diagnostics
        .iter()
        .any(|diagnostic| diagnostic.severity == DiagnosticSeverity::Error)
    {
        ConfigValidation::invalid(path, diagnostics)
    } else {
        ConfigValidation::valid(path, diagnostics)
    }
}

fn validate_tui_table(table: &toml::Table, diagnostics: &mut Vec<ConfigDiagnostic>) {
    if let Some(tick_ms) = table
        .get("tick_ms")
        .or_else(|| table.get("tick_rate_ms"))
        .and_then(toml::Value::as_integer)
    {
        if !(50..=2_000).contains(&tick_ms) {
            diagnostics.push(ConfigDiagnostic::warning(
                "[tui].tick_ms",
                "tick rate outside expected 50..=2000 ms range",
            ));
        }
    } else {
        diagnostics.push(ConfigDiagnostic::warning(
            "[tui].tick_ms",
            "no tick_ms or tick_rate_ms configured",
        ));
    }

    if table
        .get("confirm_destructive_actions")
        .and_then(toml::Value::as_bool)
        .is_some_and(|confirm| !confirm)
    {
        diagnostics.push(ConfigDiagnostic::warning(
            "[tui].confirm_destructive_actions",
            "destructive confirmations are disabled",
        ));
    }
}

fn validate_runtime_table(table: &toml::Table, diagnostics: &mut Vec<ConfigDiagnostic>) {
    let hard_limit = table
        .get("hard_memory_limit_mb")
        .and_then(toml::Value::as_integer);
    let headroom = table
        .get("leave_system_headroom_mb")
        .and_then(toml::Value::as_integer);

    match (hard_limit, headroom) {
        (Some(limit), Some(headroom)) if limit <= 0 || headroom <= 0 => {
            diagnostics.push(ConfigDiagnostic::error(
                "[runtime]",
                "hard_memory_limit_mb and leave_system_headroom_mb must be positive",
            ))
        }
        (Some(limit), Some(headroom)) if headroom >= limit => {
            diagnostics.push(ConfigDiagnostic::error(
                "[runtime].leave_system_headroom_mb",
                "system headroom must be lower than hard_memory_limit_mb",
            ))
        }
        (Some(_), Some(_)) => {}
        _ => diagnostics.push(ConfigDiagnostic::error(
            "[runtime]",
            "missing hard_memory_limit_mb or leave_system_headroom_mb",
        )),
    }
}

fn validate_model_table(table: &toml::Table, diagnostics: &mut Vec<ConfigDiagnostic>) {
    let target = table.get("target").and_then(toml::Value::as_str);
    if target.is_none_or(str::is_empty) {
        diagnostics.push(ConfigDiagnostic::error(
            "[model].target",
            "model target must be set",
        ));
    }

    validate_revision_pin(
        table.get("target_revision").and_then(toml::Value::as_str),
        "[model].target_revision",
        diagnostics,
    );
    if table
        .get("drafter")
        .and_then(toml::Value::as_str)
        .is_some_and(|drafter| !drafter.is_empty())
    {
        validate_revision_pin(
            table.get("drafter_revision").and_then(toml::Value::as_str),
            "[model].drafter_revision",
            diagnostics,
        );
    }
}

fn validate_revision_pin(
    revision: Option<&str>,
    path: &str,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    let Some(revision) = revision
        .map(str::trim)
        .filter(|revision| !revision.is_empty())
    else {
        diagnostics.push(ConfigDiagnostic::warning(
            path,
            "revision or local artifact hash should be pinned before release evidence",
        ));
        return;
    };
    if revision == "PIN_ME" || revision.starts_with("unavailable:") {
        diagnostics.push(ConfigDiagnostic::warning(
            path,
            "revision is not pinned; use a real revision or local-artifact-sha256 from the P11 manifest",
        ));
    }
}

fn table_at<'a>(root: &'a toml::Table, key: &str) -> Option<&'a toml::Table> {
    root.get(key)?.as_table()
}

pub fn write_validation_report(path: &Path, validation: &ConfigValidation) -> Result<(), TuiError> {
    let body = serde_json::to_string_pretty(validation)
        .map_err(|error| TuiError::Config(error.to_string()))?;
    fs::write(path, body)?;
    Ok(())
}
