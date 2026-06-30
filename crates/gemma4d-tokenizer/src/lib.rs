#![doc = "Gemma4D config validation and fixture tokenizer loading."]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fmt, fs,
    path::{Path, PathBuf},
};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    UnsupportedConfig(String),
    InvalidTokenizer(String),
    UnknownTokenId(u32),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "{}: {}", path.display(), source),
            Self::Json { path, source } => write!(f, "{}: {}", path.display(), source),
            Self::UnsupportedConfig(message) => write!(f, "unsupported Gemma config: {message}"),
            Self::InvalidTokenizer(message) => write!(f, "invalid tokenizer fixture: {message}"),
            Self::UnknownTokenId(id) => write!(f, "unknown token id {id}"),
        }
    }
}

impl std::error::Error for Error {}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GemmaConfig {
    #[serde(default)]
    pub architecture: Option<String>,
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub model_type: Option<String>,
    pub num_hidden_layers: u32,
    pub sliding_window: u32,
    pub max_position_embeddings: u32,
    pub vocab_size: u32,
    #[serde(default)]
    pub attention_pattern: Vec<String>,
    #[serde(default)]
    pub global_layers_have_unified_kv: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedGemmaConfig {
    pub architecture: String,
    pub layers: u32,
    pub sliding_window: u32,
    pub max_context_length: u32,
    pub vocab_size: u32,
}

impl GemmaConfig {
    pub const SUPPORTED_ARCHITECTURE: &'static str = "Gemma4UnifiedForConditionalGeneration";
    pub const SUPPORTED_LAYERS: u32 = 48;
    pub const SUPPORTED_SLIDING_WINDOW: u32 = 1024;
    pub const SUPPORTED_MAX_CONTEXT: u32 = 262_144;
    pub const SUPPORTED_VOCAB_SIZE: u32 = 262_144;

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|source| Error::Io {
            path: path.to_owned(),
            source,
        })?;
        serde_json::from_slice(&bytes).map_err(|source| Error::Json {
            path: path.to_owned(),
            source,
        })
    }

    pub fn validate(&self) -> Result<ValidatedGemmaConfig> {
        let architecture = self.architecture();
        if architecture != Self::SUPPORTED_ARCHITECTURE {
            return Err(Error::UnsupportedConfig(format!(
                "architecture must be {}, got {}",
                Self::SUPPORTED_ARCHITECTURE,
                architecture
            )));
        }
        if self.num_hidden_layers != Self::SUPPORTED_LAYERS {
            return Err(Error::UnsupportedConfig(format!(
                "num_hidden_layers must be {}, got {}",
                Self::SUPPORTED_LAYERS,
                self.num_hidden_layers
            )));
        }
        if self.sliding_window != Self::SUPPORTED_SLIDING_WINDOW {
            return Err(Error::UnsupportedConfig(format!(
                "sliding_window must be {}, got {}",
                Self::SUPPORTED_SLIDING_WINDOW,
                self.sliding_window
            )));
        }
        if self.max_position_embeddings != Self::SUPPORTED_MAX_CONTEXT {
            return Err(Error::UnsupportedConfig(format!(
                "max_position_embeddings must be {}, got {}",
                Self::SUPPORTED_MAX_CONTEXT,
                self.max_position_embeddings
            )));
        }
        if self.vocab_size != Self::SUPPORTED_VOCAB_SIZE {
            return Err(Error::UnsupportedConfig(format!(
                "vocab_size must be {}, got {}",
                Self::SUPPORTED_VOCAB_SIZE,
                self.vocab_size
            )));
        }
        if !self.has_hybrid_attention() {
            return Err(Error::UnsupportedConfig(
                "attention_pattern must include both local and global/full attention".to_owned(),
            ));
        }
        if !self.global_layers_have_unified_kv {
            return Err(Error::UnsupportedConfig(
                "global_layers_have_unified_kv must be true".to_owned(),
            ));
        }

        Ok(ValidatedGemmaConfig {
            architecture,
            layers: self.num_hidden_layers,
            sliding_window: self.sliding_window,
            max_context_length: self.max_position_embeddings,
            vocab_size: self.vocab_size,
        })
    }

    fn architecture(&self) -> String {
        self.architecture
            .clone()
            .or_else(|| self.architectures.first().cloned())
            .unwrap_or_else(|| "<missing>".to_owned())
    }

    fn has_hybrid_attention(&self) -> bool {
        let has_local = self
            .attention_pattern
            .iter()
            .any(|kind| matches!(kind.as_str(), "local" | "sliding" | "sliding_attention"));
        let has_global = self
            .attention_pattern
            .iter()
            .any(|kind| matches!(kind.as_str(), "global" | "full" | "full_attention"));
        has_local && has_global
    }
}

#[derive(Debug, Clone, Deserialize)]
struct TokenizerFile {
    format: String,
    unknown_token: String,
    bos_token: String,
    eos_token: String,
    stop_tokens: Vec<String>,
    vocab: Vec<TokenEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct TokenEntry {
    token: String,
    id: u32,
}

#[derive(Debug, Clone)]
pub struct FixtureTokenizer {
    token_to_id: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    ordered_tokens: Vec<(String, u32)>,
    unknown_id: u32,
    bos_id: u32,
    eos_id: u32,
    stop_token_ids: Vec<u32>,
}

impl FixtureTokenizer {
    pub const FORMAT: &'static str = "gemma4d_fixture_tokenizer_v1";

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|source| Error::Io {
            path: path.to_owned(),
            source,
        })?;
        let file: TokenizerFile = serde_json::from_slice(&bytes).map_err(|source| Error::Json {
            path: path.to_owned(),
            source,
        })?;

        if file.format != Self::FORMAT {
            return Err(Error::InvalidTokenizer(format!(
                "format must be {}, got {}",
                Self::FORMAT,
                file.format
            )));
        }

        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();
        for entry in file.vocab {
            if token_to_id.insert(entry.token.clone(), entry.id).is_some() {
                return Err(Error::InvalidTokenizer(format!(
                    "duplicate token {:?}",
                    entry.token
                )));
            }
            if id_to_token.insert(entry.id, entry.token.clone()).is_some() {
                return Err(Error::InvalidTokenizer(format!(
                    "duplicate token id {}",
                    entry.id
                )));
            }
        }

        let unknown_id = required_id(&token_to_id, &file.unknown_token)?;
        let bos_id = required_id(&token_to_id, &file.bos_token)?;
        let eos_id = required_id(&token_to_id, &file.eos_token)?;
        let mut stop_token_ids = Vec::with_capacity(file.stop_tokens.len());
        for token in &file.stop_tokens {
            stop_token_ids.push(required_id(&token_to_id, token)?);
        }

        let mut ordered_tokens: Vec<(String, u32)> = token_to_id
            .iter()
            .filter(|(token, _)| !token.is_empty())
            .map(|(token, id)| (token.clone(), *id))
            .collect();
        ordered_tokens.sort_by(|(left, _), (right, _)| {
            right.len().cmp(&left.len()).then_with(|| left.cmp(right))
        });

        Ok(Self {
            token_to_id,
            id_to_token,
            ordered_tokens,
            unknown_id,
            bos_id,
            eos_id,
            stop_token_ids,
        })
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    pub fn bos_id(&self) -> u32 {
        self.bos_id
    }

    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }

    pub fn stop_token_ids(&self) -> &[u32] {
        &self.stop_token_ids
    }

    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut rest = text;

        while !rest.is_empty() {
            if let Some((token, id)) = self
                .ordered_tokens
                .iter()
                .find(|(token, _)| rest.starts_with(token.as_str()))
            {
                ids.push(*id);
                rest = &rest[token.len()..];
            } else {
                ids.push(self.unknown_id);
                let next = rest
                    .char_indices()
                    .nth(1)
                    .map(|(idx, _)| idx)
                    .unwrap_or(rest.len());
                rest = &rest[next..];
            }
        }

        ids
    }

    pub fn detokenize(&self, ids: &[u32]) -> Result<String> {
        let mut out = String::new();
        for id in ids {
            let token = self.id_to_token.get(id).ok_or(Error::UnknownTokenId(*id))?;
            out.push_str(token);
        }
        Ok(out)
    }
}

fn required_id(token_to_id: &HashMap<String, u32>, token: &str) -> Result<u32> {
    token_to_id
        .get(token)
        .copied()
        .ok_or_else(|| Error::InvalidTokenizer(format!("required token {token:?} is missing")))
}

pub fn file_sha256(path: impl AsRef<Path>) -> Result<String> {
    let path = path.as_ref();
    let bytes = fs::read(path).map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })?;
    Ok(sha256_hex(&bytes))
}

pub fn canonical_json_sha256<T: Serialize>(value: &T) -> Result<String> {
    let bytes = serde_json::to_vec(value).map_err(|source| Error::Json {
        path: PathBuf::from("<canonical-json>"),
        source,
    })?;
    Ok(sha256_hex(&bytes))
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(path: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures")
            .join(path)
    }

    #[test]
    fn validates_supported_gemma4_config() {
        let config = GemmaConfig::load(fixture("model/gemma4_12b_config.json")).expect("config");
        let validated = config.validate().expect("supported config");
        assert_eq!(validated.architecture, GemmaConfig::SUPPORTED_ARCHITECTURE);
        assert_eq!(validated.layers, 48);
        assert_eq!(validated.vocab_size, 262_144);
    }

    #[test]
    fn unsupported_configs_fail_clearly() {
        let mut config =
            GemmaConfig::load(fixture("model/gemma4_12b_config.json")).expect("config");
        config.num_hidden_layers = 28;
        let err = config.validate().expect_err("layer count must be rejected");
        assert!(err.to_string().contains("num_hidden_layers"));
    }

    #[test]
    fn loads_fixture_tokenizer_and_round_trips_boundaries() {
        let tokenizer =
            FixtureTokenizer::load(fixture("tokenizer/tokenizer.json")).expect("tokenizer");
        let ids = tokenizer.tokenize("<bos><start_of_turn>user\nHello<end_of_turn>\n");
        assert_eq!(ids, [1, 3, 6, 9, 20, 4, 9]);
        assert_eq!(
            tokenizer.detokenize(&ids).expect("detokenized"),
            "<bos><start_of_turn>user\nHello<end_of_turn>\n"
        );
        assert_eq!(tokenizer.stop_token_ids(), &[2, 4]);
    }

    #[test]
    fn file_hashes_are_stable() {
        let path = fixture("tokenizer/tokenizer.json");
        let first = file_sha256(&path).expect("hash");
        let second = file_sha256(&path).expect("hash");
        assert_eq!(first, second);
        assert_eq!(first.len(), 64);
    }
}
