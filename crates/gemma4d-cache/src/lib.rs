#![doc = "Deterministic cache-key hash inputs for Gemma4D."]

use gemma4d_chat::{ChatTemplateConfig, template_hash};
use gemma4d_tokenizer::sha256_hex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fmt, path::PathBuf};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Json(serde_json::Error),
    Chat(gemma4d_chat::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(source) => write!(f, "{source}"),
            Self::Chat(source) => write!(f, "{source}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<serde_json::Error> for Error {
    fn from(source: serde_json::Error) -> Self {
        Self::Json(source)
    }
}

impl From<gemma4d_chat::Error> for Error {
    fn from(source: gemma4d_chat::Error) -> Self {
        Self::Chat(source)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct CacheKeyInputs {
    pub model_repo: String,
    pub model_revision: String,
    pub weight_hash: String,
    pub quantization_hash: String,
    pub tokenizer_hash: String,
    pub chat_template_hash: String,
    pub prompt_token_prefix_hash: String,
    pub raw_prompt_hash: String,
    pub adapter_id: Option<String>,
    pub adapter_hash: Option<String>,
    pub kv_layout_version: String,
    pub kv_dtype: String,
    pub mlx_version: String,
    pub engine_version: String,
}

impl CacheKeyInputs {
    pub fn namespace_hash(&self) -> Result<String> {
        let bytes = serde_json::to_vec(self)?;
        Ok(sha256_hex(&bytes))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptHashes {
    pub raw_prompt_hash: String,
    pub prompt_token_prefix_hash: String,
}

pub fn prompt_hashes(rendered_prompt: &str, token_ids: &[u32]) -> PromptHashes {
    PromptHashes {
        raw_prompt_hash: sha256_hex(rendered_prompt.as_bytes()),
        prompt_token_prefix_hash: prompt_token_prefix_hash(token_ids),
    }
}

pub fn prompt_token_prefix_hash(token_ids: &[u32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"gemma4d:prompt-token-prefix:v1\0");
    for id in token_ids {
        hasher.update(id.to_le_bytes());
    }
    hex::encode(hasher.finalize())
}

pub fn cache_key_inputs_for_prompt(
    rendered_prompt: &str,
    token_ids: &[u32],
    tokenizer_hash: String,
    template: &ChatTemplateConfig,
) -> Result<CacheKeyInputs> {
    let prompt_hashes = prompt_hashes(rendered_prompt, token_ids);
    Ok(CacheKeyInputs {
        model_repo: "mlx-community/gemma-4-12B-it-4bit".to_owned(),
        model_revision: "fixture-m02".to_owned(),
        weight_hash: "fixture-no-weights-loaded".to_owned(),
        quantization_hash: "mlx-4bit-fixture".to_owned(),
        tokenizer_hash,
        chat_template_hash: template_hash(template)?,
        prompt_token_prefix_hash: prompt_hashes.prompt_token_prefix_hash,
        raw_prompt_hash: prompt_hashes.raw_prompt_hash,
        adapter_id: None,
        adapter_hash: None,
        kv_layout_version: "kv-layout-v1".to_owned(),
        kv_dtype: "bf16".to_owned(),
        mlx_version: "not-loaded-m02".to_owned(),
        engine_version: env!("CARGO_PKG_VERSION").to_owned(),
    })
}

#[allow(dead_code)]
fn fixture(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gemma4d_chat::{PromptFixture, compile_prompt};
    use gemma4d_tokenizer::{FixtureTokenizer, file_sha256};

    #[test]
    fn prompt_hash_inputs_are_deterministic() {
        let fixture_path = fixture("prompts/simple_chat.json");
        let tokenizer_path = fixture("tokenizer/tokenizer.json");
        let prompt = PromptFixture::load(fixture_path).expect("fixture");
        let tokenizer = FixtureTokenizer::load(&tokenizer_path).expect("tokenizer");
        let rendered = compile_prompt(&prompt.chat_messages(), &prompt.template).expect("prompt");
        let ids = tokenizer.tokenize(&rendered);
        let tokenizer_hash = file_sha256(tokenizer_path).expect("tokenizer hash");

        let first =
            cache_key_inputs_for_prompt(&rendered, &ids, tokenizer_hash.clone(), &prompt.template)
                .expect("inputs");
        let second = cache_key_inputs_for_prompt(&rendered, &ids, tokenizer_hash, &prompt.template)
            .expect("inputs");

        assert_eq!(first, second);
        assert_eq!(
            first.namespace_hash().expect("hash"),
            second.namespace_hash().expect("hash")
        );
        assert_eq!(first.namespace_hash().expect("hash").len(), 64);
    }

    #[test]
    fn prompt_token_order_changes_hash() {
        let left = prompt_token_prefix_hash(&[1, 2, 3]);
        let right = prompt_token_prefix_hash(&[1, 3, 2]);
        assert_ne!(left, right);
    }
}
