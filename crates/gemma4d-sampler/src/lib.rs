#![doc = "Sampling policy and deterministic greedy token selection."]

pub const CRATE_NAME: &str = "gemma4d-sampler";

pub fn bootstrap_status() -> &'static str {
    "greedy"
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GreedyChoice {
    pub token_id: i32,
    pub logit: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    EmptyLogits,
    NonFiniteLogit { index: usize, value: f32 },
    TokenIdOverflow { index: usize, vocab_offset: i32 },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyLogits => write!(f, "greedy sampling requires at least one logit"),
            Self::NonFiniteLogit { index, value } => {
                write!(f, "logit at index {index} is not finite: {value}")
            }
            Self::TokenIdOverflow {
                index,
                vocab_offset,
            } => write!(
                f,
                "token id overflow for index {index} with vocab offset {vocab_offset}"
            ),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub fn greedy_argmax(logits: &[f32]) -> Result<GreedyChoice> {
    greedy_argmax_with_vocab_offset(logits, 0)
}

pub fn greedy_argmax_with_vocab_offset(logits: &[f32], vocab_offset: i32) -> Result<GreedyChoice> {
    let Some((&first, rest)) = logits.split_first() else {
        return Err(Error::EmptyLogits);
    };
    if !first.is_finite() {
        return Err(Error::NonFiniteLogit {
            index: 0,
            value: first,
        });
    }

    let mut best_index = 0usize;
    let mut best_logit = first;

    for (index, &logit) in rest.iter().enumerate() {
        let index = index + 1;
        if !logit.is_finite() {
            return Err(Error::NonFiniteLogit {
                index,
                value: logit,
            });
        }
        if logit > best_logit {
            best_index = index;
            best_logit = logit;
        }
    }

    let index_as_i32 = i32::try_from(best_index).map_err(|_| Error::TokenIdOverflow {
        index: best_index,
        vocab_offset,
    })?;
    let token_id = vocab_offset
        .checked_add(index_as_i32)
        .ok_or(Error::TokenIdOverflow {
            index: best_index,
            vocab_offset,
        })?;

    Ok(GreedyChoice {
        token_id,
        logit: best_logit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_placeholder_status() {
        assert_eq!(CRATE_NAME, "gemma4d-sampler");
        assert_eq!(bootstrap_status(), "greedy");
    }

    #[test]
    fn greedy_argmax_selects_largest_logit() {
        assert_eq!(
            greedy_argmax(&[-1.0, 3.5, 2.0]).expect("choice"),
            GreedyChoice {
                token_id: 1,
                logit: 3.5
            }
        );
    }

    #[test]
    fn greedy_argmax_breaks_ties_toward_lowest_token_id() {
        assert_eq!(
            greedy_argmax_with_vocab_offset(&[7.0, 7.0, 6.0], 10).expect("choice"),
            GreedyChoice {
                token_id: 10,
                logit: 7.0
            }
        );
    }

    #[test]
    fn greedy_argmax_rejects_empty_or_non_finite_logits() {
        assert_eq!(greedy_argmax(&[]), Err(Error::EmptyLogits));
        match greedy_argmax(&[1.0, f32::NAN]).expect_err("NaN should fail") {
            Error::NonFiniteLogit { index, value } => {
                assert_eq!(index, 1);
                assert!(value.is_nan());
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
