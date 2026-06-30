#![doc = "Engine coordination primitives for greedy and MTP speculative decoding."]

use std::time::Instant;

pub const CRATE_NAME: &str = "gemma4d-engine";

pub fn bootstrap_status() -> &'static str {
    "mtp-speculative-decoding"
}

#[derive(Debug, Clone, PartialEq)]
pub struct MtpConfig {
    pub enabled: bool,
    pub draft_block_size: usize,
    pub adapter_id: Option<String>,
    pub active_kv_compressed: bool,
    pub auto_disable_min_acceptance_rate: f64,
}

impl MtpConfig {
    pub fn block_size(block_size: usize) -> Self {
        Self {
            draft_block_size: block_size,
            ..Self::default()
        }
    }

    fn admission_disable_reason(&self) -> Result<Option<String>, MtpError> {
        if !self.enabled {
            return Ok(Some("MTP disabled by config".to_owned()));
        }
        if self.draft_block_size == 0 {
            return Err(MtpError::InvalidConfig(
                "draft_block_size must be greater than zero".to_owned(),
            ));
        }
        if let Some(adapter_id) = &self.adapter_id {
            if adapter_id != "none" {
                return Ok(Some(format!(
                    "MTP disabled because adapter '{adapter_id}' is active"
                )));
            }
        }
        if self.active_kv_compressed {
            return Ok(Some(
                "MTP disabled because compressed active KV is not verified in M06".to_owned(),
            ));
        }
        Ok(None)
    }
}

impl Default for MtpConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            draft_block_size: 1,
            adapter_id: None,
            active_kv_compressed: false,
            auto_disable_min_acceptance_rate: 0.35,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TargetStep {
    pub token_id: i32,
    pub logit: f32,
    pub peak_memory_gb: f32,
}

impl TargetStep {
    pub fn new(token_id: i32) -> Self {
        Self {
            token_id,
            logit: 0.0,
            peak_memory_gb: 0.0,
        }
    }
}

pub trait GreedyTarget {
    fn next_greedy(
        &mut self,
        prompt_tokens: &[i32],
        accepted_tokens: &[i32],
    ) -> Result<TargetStep, MtpError>;
}

pub trait Drafter {
    fn draft(
        &mut self,
        prompt_tokens: &[i32],
        accepted_tokens: &[i32],
        block_size: usize,
    ) -> Result<Vec<i32>, MtpError>;
}

#[derive(Debug, Clone, PartialEq)]
pub enum MtpError {
    InvalidConfig(String),
    Target(String),
    Drafter(String),
}

impl std::fmt::Display for MtpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(message) | Self::Target(message) | Self::Drafter(message) => {
                f.write_str(message)
            }
        }
    }
}

impl std::error::Error for MtpError {}

#[derive(Debug, Clone, PartialEq)]
pub struct MtpMetrics {
    pub draft_block_size: usize,
    pub attempted_draft_tokens: u64,
    pub accepted_draft_tokens: u64,
    pub acceptance_rate: f64,
    pub accepted_tokens_per_verify: f64,
    pub target_verify_passes: u64,
    pub decode_tokens_per_second: f64,
    pub peak_memory_gb: f32,
    pub rollback_count: u64,
    pub auto_disabled: bool,
    pub auto_disable_reason: Option<String>,
}

impl MtpMetrics {
    pub fn new(draft_block_size: usize) -> Self {
        Self {
            draft_block_size,
            attempted_draft_tokens: 0,
            accepted_draft_tokens: 0,
            acceptance_rate: 0.0,
            accepted_tokens_per_verify: 0.0,
            target_verify_passes: 0,
            decode_tokens_per_second: 0.0,
            peak_memory_gb: 0.0,
            rollback_count: 0,
            auto_disabled: false,
            auto_disable_reason: None,
        }
    }

    fn refresh_rates(&mut self) {
        self.acceptance_rate = if self.attempted_draft_tokens == 0 {
            0.0
        } else {
            self.accepted_draft_tokens as f64 / self.attempted_draft_tokens as f64
        };
        self.accepted_tokens_per_verify = if self.target_verify_passes == 0 {
            0.0
        } else {
            self.accepted_draft_tokens as f64 / self.target_verify_passes as f64
        };
    }

    fn auto_disable(&mut self, reason: impl Into<String>) {
        self.auto_disabled = true;
        self.auto_disable_reason = Some(reason.into());
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MtpRun {
    pub generated_tokens: Vec<i32>,
    pub metrics: MtpMetrics,
}

pub fn non_mtp_greedy<T: GreedyTarget>(
    target: &mut T,
    prompt_tokens: &[i32],
    max_new_tokens: usize,
) -> Result<Vec<i32>, MtpError> {
    let mut generated = Vec::with_capacity(max_new_tokens);
    for _ in 0..max_new_tokens {
        let step = target.next_greedy(prompt_tokens, &generated)?;
        generated.push(step.token_id);
    }
    Ok(generated)
}

pub fn speculative_greedy<T, D>(
    target: &mut T,
    drafter: &mut D,
    prompt_tokens: &[i32],
    max_new_tokens: usize,
    config: &MtpConfig,
) -> Result<MtpRun, MtpError>
where
    T: GreedyTarget,
    D: Drafter,
{
    let mut metrics = MtpMetrics::new(config.draft_block_size);
    let started = Instant::now();

    if let Some(reason) = config.admission_disable_reason()? {
        metrics.auto_disable(reason);
        let generated = non_mtp_greedy(target, prompt_tokens, max_new_tokens)?;
        finish_timing(&mut metrics, started, generated.len());
        return Ok(MtpRun {
            generated_tokens: generated,
            metrics,
        });
    }

    let mut generated = Vec::with_capacity(max_new_tokens);
    while generated.len() < max_new_tokens {
        let remaining = max_new_tokens - generated.len();
        let block_size = config.draft_block_size.min(remaining);
        let mut draft = drafter.draft(prompt_tokens, &generated, block_size)?;
        if draft.is_empty() {
            metrics.auto_disable("MTP drafter returned no tokens");
            fill_greedy_tail(
                target,
                prompt_tokens,
                max_new_tokens,
                &mut generated,
                &mut metrics,
            )?;
            break;
        }
        draft.truncate(block_size);
        metrics.attempted_draft_tokens += draft.len() as u64;
        metrics.target_verify_passes += 1;

        let mut rejected = false;
        for drafted_token in draft {
            if generated.len() == max_new_tokens {
                break;
            }
            let step = target.next_greedy(prompt_tokens, &generated)?;
            metrics.peak_memory_gb = metrics.peak_memory_gb.max(step.peak_memory_gb);
            if step.token_id == drafted_token {
                generated.push(drafted_token);
                metrics.accepted_draft_tokens += 1;
            } else {
                generated.push(step.token_id);
                metrics.rollback_count += 1;
                rejected = true;
                break;
            }
        }

        metrics.refresh_rates();
        if rejected && metrics.acceptance_rate < config.auto_disable_min_acceptance_rate {
            metrics.auto_disable(format!(
                "MTP acceptance rate {:.3} fell below threshold {:.3}",
                metrics.acceptance_rate, config.auto_disable_min_acceptance_rate
            ));
            fill_greedy_tail(
                target,
                prompt_tokens,
                max_new_tokens,
                &mut generated,
                &mut metrics,
            )?;
            break;
        }
    }

    finish_timing(&mut metrics, started, generated.len());
    Ok(MtpRun {
        generated_tokens: generated,
        metrics,
    })
}

fn fill_greedy_tail<T: GreedyTarget>(
    target: &mut T,
    prompt_tokens: &[i32],
    max_new_tokens: usize,
    generated: &mut Vec<i32>,
    metrics: &mut MtpMetrics,
) -> Result<(), MtpError> {
    while generated.len() < max_new_tokens {
        let step = target.next_greedy(prompt_tokens, generated)?;
        metrics.peak_memory_gb = metrics.peak_memory_gb.max(step.peak_memory_gb);
        generated.push(step.token_id);
    }
    Ok(())
}

fn finish_timing(metrics: &mut MtpMetrics, started: Instant, generated_tokens: usize) {
    metrics.refresh_rates();
    let elapsed = started.elapsed().as_secs_f64();
    metrics.decode_tokens_per_second = if elapsed > 0.0 {
        generated_tokens as f64 / elapsed
    } else {
        0.0
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_mtp_status() {
        assert_eq!(CRATE_NAME, "gemma4d-engine");
        assert_eq!(bootstrap_status(), "mtp-speculative-decoding");
    }

    #[test]
    fn mtp_block_size_1_matches_non_mtp_greedy() {
        let prompt = [9259];
        let expected = vec![236772, 236772, 236772, 236772];
        let mut baseline = ScriptedTarget::new(expected.clone());
        let baseline_tokens = non_mtp_greedy(&mut baseline, &prompt, expected.len()).unwrap();

        let mut target = ScriptedTarget::new(expected.clone());
        let mut drafter = PerfectDrafter::new(expected.clone());
        let run = speculative_greedy(
            &mut target,
            &mut drafter,
            &prompt,
            expected.len(),
            &MtpConfig::block_size(1),
        )
        .unwrap();

        assert_eq!(run.generated_tokens, baseline_tokens);
        assert_eq!(run.metrics.draft_block_size, 1);
        assert_eq!(run.metrics.attempted_draft_tokens, 4);
        assert_eq!(run.metrics.accepted_draft_tokens, 4);
        assert_eq!(run.metrics.rollback_count, 0);
        assert_eq!(run.metrics.acceptance_rate, 1.0);
        assert!(!run.metrics.auto_disabled);
    }

    #[test]
    fn mtp_block_size_2_matches_non_mtp_with_rollback() {
        let prompt = [9259];
        let expected = vec![10, 11, 12, 13];
        let mut baseline = ScriptedTarget::new(expected.clone());
        let baseline_tokens = non_mtp_greedy(&mut baseline, &prompt, expected.len()).unwrap();

        let mut target = ScriptedTarget::new(expected);
        let mut drafter = BlockDrafter::new(vec![vec![10, 99], vec![12, 13]]);
        let run = speculative_greedy(
            &mut target,
            &mut drafter,
            &prompt,
            baseline_tokens.len(),
            &MtpConfig::block_size(2),
        )
        .unwrap();

        assert_eq!(run.generated_tokens, baseline_tokens);
        assert_eq!(run.metrics.draft_block_size, 2);
        assert_eq!(run.metrics.attempted_draft_tokens, 4);
        assert_eq!(run.metrics.accepted_draft_tokens, 3);
        assert_eq!(run.metrics.rollback_count, 1);
        assert_eq!(run.metrics.target_verify_passes, 2);
        assert_eq!(run.metrics.accepted_tokens_per_verify, 1.5);
        assert!(!run.metrics.auto_disabled);
    }

    #[test]
    fn mtp_auto_disables_when_acceptance_falls_below_threshold() {
        let prompt = [9259];
        let expected = vec![10, 11, 12, 13];
        let mut target = ScriptedTarget::new(expected.clone());
        let mut drafter = BlockDrafter::new(vec![vec![99, 98]]);
        let run = speculative_greedy(
            &mut target,
            &mut drafter,
            &prompt,
            expected.len(),
            &MtpConfig::block_size(2),
        )
        .unwrap();

        assert_eq!(run.generated_tokens, expected);
        assert_eq!(run.metrics.attempted_draft_tokens, 2);
        assert_eq!(run.metrics.accepted_draft_tokens, 0);
        assert_eq!(run.metrics.rollback_count, 1);
        assert!(run.metrics.auto_disabled);
        assert!(
            run.metrics
                .auto_disable_reason
                .as_deref()
                .unwrap_or_default()
                .contains("acceptance rate")
        );
    }

    #[test]
    fn mtp_disables_when_adapter_is_active() {
        let prompt = [1];
        let expected = vec![2, 3];
        let mut target = ScriptedTarget::new(expected.clone());
        let mut drafter = PerfectDrafter::new(expected.clone());
        let config = MtpConfig {
            adapter_id: Some("rust-expert".to_owned()),
            ..MtpConfig::block_size(1)
        };
        let run = speculative_greedy(&mut target, &mut drafter, &prompt, expected.len(), &config)
            .unwrap();

        assert_eq!(run.generated_tokens, expected);
        assert_eq!(run.metrics.attempted_draft_tokens, 0);
        assert!(run.metrics.auto_disabled);
        assert!(
            run.metrics
                .auto_disable_reason
                .as_deref()
                .unwrap_or_default()
                .contains("adapter")
        );
    }

    #[test]
    fn mtp_disables_for_compressed_active_kv_in_m06() {
        let prompt = [1];
        let expected = vec![2, 3];
        let mut target = ScriptedTarget::new(expected.clone());
        let mut drafter = PerfectDrafter::new(expected.clone());
        let config = MtpConfig {
            active_kv_compressed: true,
            ..MtpConfig::block_size(1)
        };
        let run = speculative_greedy(&mut target, &mut drafter, &prompt, expected.len(), &config)
            .unwrap();

        assert_eq!(run.generated_tokens, expected);
        assert_eq!(run.metrics.attempted_draft_tokens, 0);
        assert!(run.metrics.auto_disabled);
        assert!(
            run.metrics
                .auto_disable_reason
                .as_deref()
                .unwrap_or_default()
                .contains("compressed active KV")
        );
    }

    #[derive(Debug, Clone)]
    struct ScriptedTarget {
        expected: Vec<i32>,
    }

    impl ScriptedTarget {
        fn new(expected: Vec<i32>) -> Self {
            Self { expected }
        }
    }

    impl GreedyTarget for ScriptedTarget {
        fn next_greedy(
            &mut self,
            _prompt_tokens: &[i32],
            accepted_tokens: &[i32],
        ) -> Result<TargetStep, MtpError> {
            self.expected
                .get(accepted_tokens.len())
                .copied()
                .map(TargetStep::new)
                .ok_or_else(|| {
                    MtpError::Target(format!(
                        "fixture target exhausted at generated length {}",
                        accepted_tokens.len()
                    ))
                })
        }
    }

    #[derive(Debug, Clone)]
    struct PerfectDrafter {
        expected: Vec<i32>,
    }

    impl PerfectDrafter {
        fn new(expected: Vec<i32>) -> Self {
            Self { expected }
        }
    }

    impl Drafter for PerfectDrafter {
        fn draft(
            &mut self,
            _prompt_tokens: &[i32],
            accepted_tokens: &[i32],
            block_size: usize,
        ) -> Result<Vec<i32>, MtpError> {
            Ok(self
                .expected
                .iter()
                .skip(accepted_tokens.len())
                .take(block_size)
                .copied()
                .collect())
        }
    }

    #[derive(Debug, Clone)]
    struct BlockDrafter {
        blocks: Vec<Vec<i32>>,
        next: usize,
    }

    impl BlockDrafter {
        fn new(blocks: Vec<Vec<i32>>) -> Self {
            Self { blocks, next: 0 }
        }
    }

    impl Drafter for BlockDrafter {
        fn draft(
            &mut self,
            _prompt_tokens: &[i32],
            _accepted_tokens: &[i32],
            _block_size: usize,
        ) -> Result<Vec<i32>, MtpError> {
            let block = self.blocks.get(self.next).cloned().ok_or_else(|| {
                MtpError::Drafter(format!("fixture drafter exhausted at block {}", self.next))
            })?;
            self.next += 1;
            Ok(block)
        }
    }
}
