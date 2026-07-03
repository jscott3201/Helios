use std::{
    collections::BTreeMap,
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use gemma4d_tokenizer::sha256_hex;
use serde::{Deserialize, Serialize};

use crate::CliError;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildProvenance {
    pub git_sha: String,
    pub git_status_short: String,
    pub dirty_diff_sha256: String,
    pub dirty_diff_bytes: usize,
    pub runner_binary_path: String,
    pub runner_binary_link_mtime_unix_seconds: u64,
    #[serde(default)]
    pub gemma4d_env: BTreeMap<String, String>,
}

pub fn capture_build_provenance() -> Result<BuildProvenance, CliError> {
    let repo_root = git_toplevel()?;
    let git_sha = git_stdout(&repo_root, &["rev-parse", "HEAD"], "git SHA")?;
    let git_status_short = git_stdout(&repo_root, &["status", "--short"], "git status")?;
    let dirty_diff = git_stdout_bytes(&repo_root, &["diff", "--binary", "HEAD"], "dirty diff")?;
    assert_git_dirty_views_agree(&git_status_short, &dirty_diff)?;

    let runner_binary = env::current_exe().map_err(|error| {
        CliError::Runtime(format!(
            "failed to capture build provenance: current executable path unavailable: {error}"
        ))
    })?;
    let runner_metadata = fs::metadata(&runner_binary).map_err(|error| {
        CliError::Runtime(format!(
            "failed to capture build provenance: runner binary metadata unavailable for {}: {error}",
            runner_binary.display()
        ))
    })?;
    let runner_mtime = runner_metadata.modified().map_err(|error| {
        CliError::Runtime(format!(
            "failed to capture build provenance: runner binary mtime unavailable for {}: {error}",
            runner_binary.display()
        ))
    })?;

    Ok(BuildProvenance {
        git_sha,
        git_status_short,
        dirty_diff_sha256: sha256_hex(&dirty_diff),
        dirty_diff_bytes: dirty_diff.len(),
        runner_binary_path: runner_binary.display().to_string(),
        runner_binary_link_mtime_unix_seconds: system_time_unix_seconds(
            runner_mtime,
            "runner binary link mtime",
        )?,
        gemma4d_env: capture_gemma4d_env(env::vars()),
    })
}

fn capture_gemma4d_env<I, K, V>(vars: I) -> BTreeMap<String, String>
where
    I: IntoIterator<Item = (K, V)>,
    K: Into<String>,
    V: Into<String>,
{
    vars.into_iter()
        .map(|(key, value)| (key.into(), value.into()))
        .filter(|(key, _)| key.starts_with("GEMMA4D_"))
        .collect()
}

fn git_toplevel() -> Result<PathBuf, CliError> {
    let output = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .map_err(|error| {
            CliError::Runtime(format!(
                "failed to capture build provenance git repository root: `git rev-parse --show-toplevel` could not start: {error}"
            ))
        })?;
    if !output.status.success() {
        return Err(CliError::Runtime(format!(
            "failed to capture build provenance git repository root: `git rev-parse --show-toplevel` exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(PathBuf::from(
        String::from_utf8_lossy(&output.stdout).trim().to_owned(),
    ))
}

fn git_stdout(repo_root: &Path, args: &[&str], label: &str) -> Result<String, CliError> {
    let bytes = git_stdout_bytes(repo_root, args, label)?;
    Ok(String::from_utf8_lossy(&bytes).trim().to_owned())
}

fn git_stdout_bytes(repo_root: &Path, args: &[&str], label: &str) -> Result<Vec<u8>, CliError> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(args)
        .output()
        .map_err(|error| {
            CliError::Runtime(format!(
                "failed to capture build provenance {label}: `{}` could not start: {error}",
                git_invocation(repo_root, args)
            ))
        })?;
    if !output.status.success() {
        return Err(CliError::Runtime(format!(
            "failed to capture build provenance {label}: `{}` exited with {}: {}",
            git_invocation(repo_root, args),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(output.stdout)
}

fn assert_git_dirty_views_agree(git_status_short: &str, dirty_diff: &[u8]) -> Result<(), CliError> {
    let status_dirty = !git_status_short.trim().is_empty();
    let diff_dirty = !dirty_diff.is_empty();
    if status_dirty == diff_dirty {
        return Ok(());
    }

    let status_preview = git_status_short
        .lines()
        .take(12)
        .collect::<Vec<_>>()
        .join("\\n");
    Err(CliError::Runtime(format!(
        "failed to capture build provenance: git status and dirty diff disagree \
         (status_dirty={status_dirty}, diff_dirty={diff_dirty}); status preview: {status_preview}"
    )))
}

fn git_invocation(repo_root: &Path, args: &[&str]) -> String {
    let mut parts = vec![
        "git".to_owned(),
        "-C".to_owned(),
        repo_root.display().to_string(),
    ];
    parts.extend(args.iter().map(|arg| (*arg).to_owned()));
    parts.join(" ")
}

fn system_time_unix_seconds(time: SystemTime, label: &str) -> Result<u64, CliError> {
    time.duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|error| {
            CliError::Runtime(format!(
                "failed to capture build provenance {label}: before UNIX_EPOCH: {error}"
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::{assert_git_dirty_views_agree, capture_gemma4d_env};

    #[test]
    fn dirty_views_accept_clean_tree() {
        assert!(assert_git_dirty_views_agree("", b"").is_ok());
    }

    #[test]
    fn dirty_views_accept_tracked_dirty_tree() {
        assert!(assert_git_dirty_views_agree(" M src/lib.rs", b"diff --git a/src/lib.rs").is_ok());
    }

    #[test]
    fn dirty_views_reject_status_only_dirty_tree() {
        assert!(assert_git_dirty_views_agree("?? scratch.txt", b"").is_err());
    }

    #[test]
    fn captures_only_gemma4d_env_vars() {
        let env = capture_gemma4d_env([
            ("PATH", "/bin"),
            ("GEMMA4D_REQUIRE_MLX", "1"),
            ("GEMMA4D_EXPERIMENTAL_MTP_BLOCK_PREFIX_ROLLBACK", "1"),
        ]);

        assert_eq!(env.len(), 2);
        assert_eq!(env.get("GEMMA4D_REQUIRE_MLX"), Some(&"1".to_owned()));
        assert!(!env.contains_key("PATH"));
    }
}
