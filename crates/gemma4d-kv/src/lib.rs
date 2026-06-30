#![doc = "KV prefix cache metadata, namespace hashing, RAM residency, and SSD cold-cache policy."]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, VecDeque},
    fmt, fs,
    num::NonZeroU64,
    path::{Path, PathBuf},
};

pub const CRATE_NAME: &str = "gemma4d-kv";
pub const KV_LAYOUT_VERSION: u32 = 1;
pub const SSD_MANIFEST_VERSION: u32 = 1;
pub const SSD_BLOCK_FILE_VERSION: u32 = 1;
pub const M09_MIN_Q8_LOGIT_COSINE: f64 = 0.999;
pub const M09_MIN_Q4_LOGIT_COSINE: f64 = 0.98;

const SSD_INDEX_FILE: &str = "index.json";

pub type Result<T> = std::result::Result<T, Error>;

pub fn bootstrap_status() -> &'static str {
    "m09-kv-compression-research"
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidBlock(String),
    InvalidManifest(String),
    Io(String),
    Json(String),
    NamespaceMismatch { expected: String, actual: String },
    ChecksumMismatch { block_id: String },
    NotFound { block_id: String },
    BudgetExceeded { block_bytes: u64, budget_bytes: u64 },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBlock(message) => f.write_str(message),
            Self::InvalidManifest(message) => f.write_str(message),
            Self::Io(message) => f.write_str(message),
            Self::Json(message) => f.write_str(message),
            Self::NamespaceMismatch { expected, actual } => {
                write!(
                    f,
                    "cache namespace mismatch: expected {expected}, got {actual}"
                )
            }
            Self::ChecksumMismatch { block_id } => {
                write!(f, "cache block checksum mismatch for {block_id}")
            }
            Self::NotFound { block_id } => write!(f, "cache block not found: {block_id}"),
            Self::BudgetExceeded {
                block_bytes,
                budget_bytes,
            } => write!(
                f,
                "cache block of {block_bytes} bytes exceeds RAM budget of {budget_bytes} bytes"
            ),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(source: std::io::Error) -> Self {
        Self::Io(source.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(source: serde_json::Error) -> Self {
        Self::Json(source.to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheMode {
    Bf16,
    MlxAffineQ8,
    MlxAffineQ4,
}

impl CacheMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::MlxAffineQ8 => "mlx_affine_q8",
            Self::MlxAffineQ4 => "mlx_affine_q4",
        }
    }

    pub fn bits_per_value(self) -> u8 {
        match self {
            Self::Bf16 => 16,
            Self::MlxAffineQ8 => 8,
            Self::MlxAffineQ4 => 4,
        }
    }

    pub fn is_compressed(self) -> bool {
        self != Self::Bf16
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionType {
    Sliding,
    Full,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvNamespace {
    pub model_id: String,
    pub model_revision: String,
    pub weights_sha256: String,
    pub quantization_sha256: String,
    pub tokenizer_sha256: String,
    pub chat_template_sha256: String,
    pub prompt_token_hash: String,
    pub raw_prompt_hash: String,
    pub adapter_id: Option<String>,
    pub adapter_weight_hash: Option<String>,
    pub kv_layout_version: u32,
    pub cache_mode: CacheMode,
    pub mlx_version: String,
    pub engine_version: String,
}

impl KvNamespace {
    pub fn namespace_hash(&self) -> Result<NamespaceHash> {
        Ok(NamespaceHash(sha256_json(self)?))
    }

    pub fn fixture(sequence_len: u64) -> Self {
        Self {
            model_id: "mlx-community/gemma-4-12B-it-4bit".to_owned(),
            model_revision: "m07-fixture".to_owned(),
            weights_sha256: "weights-fixture-sha256".to_owned(),
            quantization_sha256: "qat-4bit-fixture-sha256".to_owned(),
            tokenizer_sha256: "tokenizer-fixture-sha256".to_owned(),
            chat_template_sha256: "chat-template-fixture-sha256".to_owned(),
            prompt_token_hash: prompt_token_hash(sequence_len),
            raw_prompt_hash: sha256_bytes(
                b"gemma4d:raw-prompt:v1\0",
                format!("fixture-prompt-{sequence_len}").as_bytes(),
            ),
            adapter_id: None,
            adapter_weight_hash: None,
            kv_layout_version: KV_LAYOUT_VERSION,
            cache_mode: CacheMode::Bf16,
            mlx_version: "m07-fixture-mlx".to_owned(),
            engine_version: env!("CARGO_PKG_VERSION").to_owned(),
        }
    }

    pub fn with_cache_mode(mut self, cache_mode: CacheMode) -> Self {
        self.cache_mode = cache_mode;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NamespaceHash(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvBlockKey {
    pub namespace_hash: NamespaceHash,
    pub block_index: u64,
    pub block_size_tokens: NonZeroU64,
    pub sequence_start: u64,
    pub sequence_end: u64,
    pub block_id: BlockId,
}

impl KvBlockKey {
    pub fn new(
        namespace: &KvNamespace,
        block_index: u64,
        block_size_tokens: NonZeroU64,
        sequence_start: u64,
        sequence_end: u64,
    ) -> Result<Self> {
        if sequence_start >= sequence_end {
            return Err(Error::InvalidBlock(format!(
                "sequence_start {sequence_start} must be before sequence_end {sequence_end}"
            )));
        }
        let token_count = sequence_end - sequence_start;
        if token_count > block_size_tokens.get() {
            return Err(Error::InvalidBlock(format!(
                "block spans {token_count} tokens but block size is {}",
                block_size_tokens.get()
            )));
        }
        let namespace_hash = namespace.namespace_hash()?;
        let block_id = BlockId(sha256_json(&BlockIdInputs {
            namespace_hash: namespace_hash.0.clone(),
            block_index,
            block_size_tokens: block_size_tokens.get(),
            sequence_start,
            sequence_end,
        })?);
        Ok(Self {
            namespace_hash,
            block_index,
            block_size_tokens,
            sequence_start,
            sequence_end,
            block_id,
        })
    }

    pub fn token_count(&self) -> u64 {
        self.sequence_end - self.sequence_start
    }
}

#[derive(Debug, Serialize)]
struct BlockIdInputs {
    namespace_hash: String,
    block_index: u64,
    block_size_tokens: u64,
    sequence_start: u64,
    sequence_end: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerBlockMetadata {
    pub layer: u32,
    pub attention_type: AttentionType,
    pub absolute_start: u64,
    pub absolute_end: u64,
    pub block_local_start: u64,
    pub block_local_end: u64,
    pub sliding_window_local_start: Option<u64>,
    pub sliding_window_local_end: Option<u64>,
    pub full_attention_cumulative_len: Option<u64>,
    pub kv_shared_from_layer: Option<u32>,
    pub physical_stored_tensors: u32,
    pub head_dim: u32,
    pub kv_heads: u32,
    pub compression: CacheMode,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeLogicalHandle {
    pub handle_id: u64,
    pub namespace_hash: NamespaceHash,
    pub sequence_start: u64,
    pub sequence_end: u64,
    pub byte_len: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PrefillObservation {
    pub sequence_len: u64,
    pub greedy_token: u32,
    pub greedy_logit_bits: u32,
}

impl PrefillObservation {
    pub fn greedy_logit(self) -> f32 {
        f32::from_bits(self.greedy_logit_bits)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct StoredPrefixPayload {
    token_count: u64,
    greedy_token: u32,
    greedy_logit_bits: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RamPrefixBlock {
    pub key: KvBlockKey,
    pub namespace: KvNamespace,
    pub byte_len: u64,
    pub layers: Vec<LayerBlockMetadata>,
    pub native_handle: Option<NativeLogicalHandle>,
    payload: StoredPrefixPayload,
    payload_checksum: String,
}

impl RamPrefixBlock {
    pub fn from_observation(
        namespace: KvNamespace,
        block_index: u64,
        block_size_tokens: NonZeroU64,
        sequence_start: u64,
        observation: PrefillObservation,
        byte_len: u64,
    ) -> Result<Self> {
        if observation.sequence_len <= sequence_start {
            return Err(Error::InvalidBlock(format!(
                "observation sequence_len {} must exceed sequence_start {sequence_start}",
                observation.sequence_len
            )));
        }
        let sequence_end = observation.sequence_len;
        let key = KvBlockKey::new(
            &namespace,
            block_index,
            block_size_tokens,
            sequence_start,
            sequence_end,
        )?;
        let cache_mode = namespace.cache_mode;
        let payload = StoredPrefixPayload {
            token_count: key.token_count(),
            greedy_token: observation.greedy_token,
            greedy_logit_bits: observation.greedy_logit_bits,
        };
        let payload_checksum = checksum_payload(&key, &payload)?;
        Ok(Self {
            key,
            namespace,
            byte_len,
            layers: default_gemma4_layers(sequence_start, sequence_end, cache_mode),
            native_handle: None,
            payload,
            payload_checksum,
        })
    }

    pub fn with_native_handle(mut self, handle_id: u64) -> Self {
        self.native_handle = Some(NativeLogicalHandle {
            handle_id,
            namespace_hash: self.key.namespace_hash.clone(),
            sequence_start: self.key.sequence_start,
            sequence_end: self.key.sequence_end,
            byte_len: self.byte_len,
        });
        self
    }

    pub fn observation(&self) -> PrefillObservation {
        PrefillObservation {
            sequence_len: self.key.sequence_end,
            greedy_token: self.payload.greedy_token,
            greedy_logit_bits: self.payload.greedy_logit_bits,
        }
    }

    fn verify_checksum(&self) -> Result<()> {
        let expected = checksum_payload(&self.key, &self.payload)?;
        if expected == self.payload_checksum {
            Ok(())
        } else {
            Err(Error::ChecksumMismatch {
                block_id: self.key.block_id.0.clone(),
            })
        }
    }

    #[cfg(test)]
    fn corrupt_checksum_for_test(&mut self) {
        self.payload_checksum = "corrupted".to_owned();
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RestoredPrefix {
    pub block_id: BlockId,
    pub namespace_hash: NamespaceHash,
    pub sequence_start: u64,
    pub sequence_end: u64,
    pub byte_len: u64,
    pub observation: PrefillObservation,
    pub native_handle: Option<NativeLogicalHandle>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CacheAccountingSnapshot {
    pub budget_bytes: u64,
    pub resident_bytes: u64,
    pub resident_blocks: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub restore_failures: u64,
    pub hit_rate: f64,
    pub ssd_enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SsdCacheAccountingSnapshot {
    pub budget_bytes: u64,
    pub stored_bytes: u64,
    pub stored_blocks: usize,
    pub hits: u64,
    pub misses: u64,
    pub writes: u64,
    pub reads: u64,
    pub evictions: u64,
    pub restore_failures: u64,
    pub namespace_rejections: u64,
    pub corruptions: u64,
    pub bytes_written: u64,
    pub bytes_read: u64,
    pub hit_rate: f64,
    pub mid_decode_fetches: u64,
    pub ssd_enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompressionWorkload {
    SimpleChat,
    JsonToolFixture,
    CodeReview,
}

impl CompressionWorkload {
    pub fn label(self) -> &'static str {
        match self {
            Self::SimpleChat => "simple_chat",
            Self::JsonToolFixture => "json_tool_fixture",
            Self::CodeReview => "code_review",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompressionQualityResult {
    pub mode: CacheMode,
    pub workload: CompressionWorkload,
    pub sequence_len: u64,
    pub logit_cosine: f64,
    pub greedy_agreement: bool,
    pub bf16_bytes: u64,
    pub compressed_bytes: u64,
    pub memory_delta_bytes: i64,
    pub memory_reduction: f64,
    pub accepted: bool,
    pub gate: CompressionQualityGate,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CompressionQualityGate {
    pub min_logit_cosine: f64,
    pub require_greedy_agreement: bool,
    pub experimental: bool,
}

impl CompressionQualityGate {
    pub fn for_mode(mode: CacheMode) -> Self {
        match mode {
            CacheMode::Bf16 => Self {
                min_logit_cosine: 1.0,
                require_greedy_agreement: true,
                experimental: false,
            },
            CacheMode::MlxAffineQ8 => Self {
                min_logit_cosine: M09_MIN_Q8_LOGIT_COSINE,
                require_greedy_agreement: true,
                experimental: false,
            },
            CacheMode::MlxAffineQ4 => Self {
                min_logit_cosine: M09_MIN_Q4_LOGIT_COSINE,
                require_greedy_agreement: true,
                experimental: false,
            },
        }
    }
}

pub fn evaluate_compression_fixture(
    sequence_len: u64,
    workload: CompressionWorkload,
    mode: CacheMode,
) -> CompressionQualityResult {
    let bf16_logits = fixture_logits(sequence_len, workload);
    let reconstructed = affine_reconstruct(&bf16_logits, mode);
    let logit_cosine = cosine_similarity(&bf16_logits, &reconstructed);
    let greedy_agreement = argmax(&bf16_logits) == argmax(&reconstructed);
    let bf16_bytes = estimated_kv_bytes_for_mode(sequence_len, CacheMode::Bf16);
    let compressed_bytes = estimated_kv_bytes_for_mode(sequence_len, mode);
    let gate = CompressionQualityGate::for_mode(mode);
    let accepted = logit_cosine >= gate.min_logit_cosine
        && (!gate.require_greedy_agreement || greedy_agreement)
        && !gate.experimental;
    CompressionQualityResult {
        mode,
        workload,
        sequence_len,
        logit_cosine,
        greedy_agreement,
        bf16_bytes,
        compressed_bytes,
        memory_delta_bytes: compressed_bytes as i64 - bf16_bytes as i64,
        memory_reduction: if bf16_bytes == 0 {
            0.0
        } else {
            1.0 - (compressed_bytes as f64 / bf16_bytes as f64)
        },
        accepted,
        gate,
    }
}

#[cfg(feature = "planar-iso-experiments")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentalCompressionMode {
    Planar4,
    Planar3,
    Iso4,
    Iso3,
}

#[cfg(feature = "planar-iso-experiments")]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExperimentalCompressionPlan {
    pub mode: ExperimentalCompressionMode,
    pub feature_flag: &'static str,
    pub k_only_global_prefix: bool,
    pub accepted: bool,
    pub reason: String,
}

#[cfg(feature = "planar-iso-experiments")]
impl ExperimentalCompressionPlan {
    pub fn candidates() -> Vec<Self> {
        [
            ExperimentalCompressionMode::Planar4,
            ExperimentalCompressionMode::Planar3,
            ExperimentalCompressionMode::Iso4,
            ExperimentalCompressionMode::Iso3,
        ]
        .into_iter()
        .map(|mode| Self {
            mode,
            feature_flag: "planar-iso-experiments",
            k_only_global_prefix: matches!(
                mode,
                ExperimentalCompressionMode::Planar4 | ExperimentalCompressionMode::Planar3
            ),
            accepted: false,
            reason: "M09 keeps Planar/Iso behind the experiment feature until quality gates pass"
                .to_owned(),
        })
        .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SsdRestorePhase {
    BeforePrefill,
    MidDecode,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersistedLayerManifest {
    pub layer: u32,
    pub attention_type: AttentionType,
    pub head_dim: u32,
    pub kv_heads: u32,
    pub attention_k_eq_v: bool,
    pub sequence_start: u64,
    pub sequence_end: u64,
    pub local_start: u64,
    pub local_end: u64,
    pub shape: Vec<u64>,
    pub stored_tensors: Vec<String>,
    pub compression: CacheMode,
    pub checksum: String,
}

impl PersistedLayerManifest {
    fn from_metadata(metadata: &LayerBlockMetadata) -> Result<Self> {
        let stored_tensors = if metadata.physical_stored_tensors == 1 {
            vec![format!("layer_{:02}_kv_shared", metadata.layer)]
        } else {
            vec![
                format!("layer_{:02}_k", metadata.layer),
                format!("layer_{:02}_v", metadata.layer),
            ]
        };
        let shape = vec![
            metadata.absolute_end - metadata.absolute_start,
            metadata.kv_heads as u64,
            metadata.head_dim as u64,
        ];
        let checksum = sha256_json(&LayerChecksumInputs {
            layer: metadata.layer,
            attention_type: metadata.attention_type,
            sequence_start: metadata.absolute_start,
            sequence_end: metadata.absolute_end,
            local_start: metadata.block_local_start,
            local_end: metadata.block_local_end,
            stored_tensors: &stored_tensors,
            compression: metadata.compression,
        })?;
        Ok(Self {
            layer: metadata.layer,
            attention_type: metadata.attention_type,
            head_dim: metadata.head_dim,
            kv_heads: metadata.kv_heads,
            attention_k_eq_v: metadata.physical_stored_tensors == 1,
            sequence_start: metadata.absolute_start,
            sequence_end: metadata.absolute_end,
            local_start: metadata.block_local_start,
            local_end: metadata.block_local_end,
            shape,
            stored_tensors,
            compression: metadata.compression,
            checksum,
        })
    }
}

#[derive(Debug, Serialize)]
struct LayerChecksumInputs<'a> {
    layer: u32,
    attention_type: AttentionType,
    sequence_start: u64,
    sequence_end: u64,
    local_start: u64,
    local_end: u64,
    stored_tensors: &'a [String],
    compression: CacheMode,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompressionManifestMetadata {
    pub mode: CacheMode,
    pub algorithm: String,
    pub bits_per_value: u8,
    pub affine_scale_format: Option<String>,
    pub experimental: bool,
    pub namespace_hash_includes_mode: bool,
}

impl CompressionManifestMetadata {
    pub fn for_mode(mode: CacheMode) -> Self {
        let (algorithm, affine_scale_format, experimental) = match mode {
            CacheMode::Bf16 => ("bf16", None, false),
            CacheMode::MlxAffineQ8 => (
                "mlx_affine_quantized",
                Some("per-block fp32 scale+bias"),
                false,
            ),
            CacheMode::MlxAffineQ4 => (
                "mlx_affine_quantized",
                Some("per-block fp32 scale+bias"),
                false,
            ),
        };
        Self {
            mode,
            algorithm: algorithm.to_owned(),
            bits_per_value: mode.bits_per_value(),
            affine_scale_format: affine_scale_format.map(str::to_owned),
            experimental,
            namespace_hash_includes_mode: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersistedKvManifest {
    pub manifest_version: u32,
    pub block_file_version: u32,
    pub kv_layout_version: u32,
    pub engine_version: String,
    pub mlx_version: String,
    pub model_id: String,
    pub model_revision: String,
    pub weights_sha256: String,
    pub quantization_sha256: String,
    pub tokenizer_sha256: String,
    pub chat_template_sha256: String,
    pub adapter_id: Option<String>,
    pub adapter_weight_hash: Option<String>,
    pub prompt_token_hash: String,
    pub raw_prompt_hash: String,
    pub namespace_hash: NamespaceHash,
    pub block_id: BlockId,
    pub cache_mode: CacheMode,
    pub compression: CompressionManifestMetadata,
    pub block_size_tokens: u64,
    pub sequence_start: u64,
    pub sequence_end: u64,
    pub logical_byte_len: u64,
    pub block_checksum: String,
    pub layers: Vec<PersistedLayerManifest>,
}

impl PersistedKvManifest {
    pub fn from_block(block: &RamPrefixBlock) -> Result<Self> {
        Ok(Self {
            manifest_version: SSD_MANIFEST_VERSION,
            block_file_version: SSD_BLOCK_FILE_VERSION,
            kv_layout_version: block.namespace.kv_layout_version,
            engine_version: block.namespace.engine_version.clone(),
            mlx_version: block.namespace.mlx_version.clone(),
            model_id: block.namespace.model_id.clone(),
            model_revision: block.namespace.model_revision.clone(),
            weights_sha256: block.namespace.weights_sha256.clone(),
            quantization_sha256: block.namespace.quantization_sha256.clone(),
            tokenizer_sha256: block.namespace.tokenizer_sha256.clone(),
            chat_template_sha256: block.namespace.chat_template_sha256.clone(),
            adapter_id: block.namespace.adapter_id.clone(),
            adapter_weight_hash: block.namespace.adapter_weight_hash.clone(),
            prompt_token_hash: block.namespace.prompt_token_hash.clone(),
            raw_prompt_hash: block.namespace.raw_prompt_hash.clone(),
            namespace_hash: block.key.namespace_hash.clone(),
            block_id: block.key.block_id.clone(),
            cache_mode: block.namespace.cache_mode,
            compression: CompressionManifestMetadata::for_mode(block.namespace.cache_mode),
            block_size_tokens: block.key.block_size_tokens.get(),
            sequence_start: block.key.sequence_start,
            sequence_end: block.key.sequence_end,
            logical_byte_len: block.byte_len,
            block_checksum: checksum_block(block)?,
            layers: block
                .layers
                .iter()
                .map(PersistedLayerManifest::from_metadata)
                .collect::<Result<Vec<_>>>()?,
        })
    }

    fn validate_for_block(&self, block: &RamPrefixBlock) -> Result<()> {
        if self.manifest_version != SSD_MANIFEST_VERSION {
            return Err(Error::InvalidManifest(format!(
                "unsupported SSD manifest version {}",
                self.manifest_version
            )));
        }
        if self.block_file_version != SSD_BLOCK_FILE_VERSION {
            return Err(Error::InvalidManifest(format!(
                "unsupported SSD block file version {}",
                self.block_file_version
            )));
        }
        if self.kv_layout_version != block.namespace.kv_layout_version {
            return Err(Error::InvalidManifest(
                "KV layout version mismatch".to_owned(),
            ));
        }
        if self.namespace_hash != block.key.namespace_hash {
            return Err(Error::InvalidManifest("namespace hash mismatch".to_owned()));
        }
        if self.block_id != block.key.block_id {
            return Err(Error::InvalidManifest("block id mismatch".to_owned()));
        }
        if self.cache_mode != block.namespace.cache_mode || self.compression.mode != self.cache_mode
        {
            return Err(Error::InvalidManifest(
                "compression mode mismatch".to_owned(),
            ));
        }
        if !self.compression.namespace_hash_includes_mode {
            return Err(Error::InvalidManifest(
                "compression mode is not declared as namespace-scoped".to_owned(),
            ));
        }
        if self.block_size_tokens != block.key.block_size_tokens.get() {
            return Err(Error::InvalidManifest("block size mismatch".to_owned()));
        }
        if self.sequence_start != block.key.sequence_start
            || self.sequence_end != block.key.sequence_end
        {
            return Err(Error::InvalidManifest("sequence span mismatch".to_owned()));
        }
        if self.logical_byte_len != block.byte_len {
            return Err(Error::InvalidManifest(
                "logical byte length mismatch".to_owned(),
            ));
        }
        let block_checksum = checksum_block(block)?;
        if self.block_checksum != block_checksum {
            return Err(Error::ChecksumMismatch {
                block_id: block.key.block_id.0.clone(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SsdIndexEntry {
    pub block_id: BlockId,
    pub namespace_hash: NamespaceHash,
    pub relative_path: String,
    pub logical_bytes: u64,
    pub stored_bytes: u64,
    pub sequence_start: u64,
    pub sequence_end: u64,
    pub block_size_tokens: u64,
    pub cache_mode: CacheMode,
}

#[derive(Debug, Clone)]
pub struct SsdPrefixCache {
    root: PathBuf,
    budget_bytes: u64,
    stored_bytes: u64,
    index: HashMap<BlockId, SsdIndexEntry>,
    lru: VecDeque<BlockId>,
    hits: u64,
    misses: u64,
    writes: u64,
    reads: u64,
    evictions: u64,
    restore_failures: u64,
    namespace_rejections: u64,
    corruptions: u64,
    bytes_written: u64,
    bytes_read: u64,
    mid_decode_fetches: u64,
}

impl SsdPrefixCache {
    pub fn open(root: impl AsRef<Path>, budget_bytes: NonZeroU64) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(root.join("blocks"))?;
        let mut cache = Self {
            root,
            budget_bytes: budget_bytes.get(),
            stored_bytes: 0,
            index: HashMap::new(),
            lru: VecDeque::new(),
            hits: 0,
            misses: 0,
            writes: 0,
            reads: 0,
            evictions: 0,
            restore_failures: 0,
            namespace_rejections: 0,
            corruptions: 0,
            bytes_written: 0,
            bytes_read: 0,
            mid_decode_fetches: 0,
        };
        cache.load_index()?;
        cache.evict_to_fit(0, None)?;
        cache.persist_index()?;
        Ok(cache)
    }

    pub fn write_block(&mut self, block: &RamPrefixBlock) -> Result<SsdIndexEntry> {
        block.verify_checksum()?;
        let block_bytes = serialized_block_bytes(block)?;
        let stored_bytes = block_bytes.len() as u64;
        if stored_bytes > self.budget_bytes {
            return Err(Error::BudgetExceeded {
                block_bytes: stored_bytes,
                budget_bytes: self.budget_bytes,
            });
        }

        let block_id = block.key.block_id.clone();
        if let Some(previous) = self.index.remove(&block_id) {
            self.stored_bytes = self.stored_bytes.saturating_sub(previous.stored_bytes);
            self.remove_lru(&block_id);
        }

        self.evict_to_fit(stored_bytes, Some(&block_id))?;

        let relative_path = format!("blocks/{}.json", block_id.0);
        let path = self.root.join(&relative_path);
        let tmp_path = self.root.join(format!("{relative_path}.tmp"));
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&tmp_path, &block_bytes)?;
        fs::rename(&tmp_path, &path)?;

        let entry = SsdIndexEntry {
            block_id: block_id.clone(),
            namespace_hash: block.key.namespace_hash.clone(),
            relative_path,
            logical_bytes: block.byte_len,
            stored_bytes,
            sequence_start: block.key.sequence_start,
            sequence_end: block.key.sequence_end,
            block_size_tokens: block.key.block_size_tokens.get(),
            cache_mode: block.namespace.cache_mode,
        };
        self.stored_bytes = self.stored_bytes.saturating_add(stored_bytes);
        self.index.insert(block_id.clone(), entry.clone());
        self.touch(&block_id);
        self.writes = self.writes.saturating_add(1);
        self.bytes_written = self.bytes_written.saturating_add(stored_bytes);
        self.persist_index()?;
        Ok(entry)
    }

    pub fn restore_before_prefill(
        &mut self,
        key: &KvBlockKey,
        expected_namespace: &KvNamespace,
    ) -> Result<RestoredPrefix> {
        self.restore_for_phase(key, expected_namespace, SsdRestorePhase::BeforePrefill)
    }

    pub fn restore_for_phase(
        &mut self,
        key: &KvBlockKey,
        expected_namespace: &KvNamespace,
        phase: SsdRestorePhase,
    ) -> Result<RestoredPrefix> {
        if phase == SsdRestorePhase::MidDecode {
            self.restore_failures = self.restore_failures.saturating_add(1);
            return Err(Error::InvalidBlock(
                "SSD cold-cache restore is only allowed before prefill".to_owned(),
            ));
        }

        let expected_hash = expected_namespace.namespace_hash()?;
        if key.namespace_hash != expected_hash {
            self.restore_failures = self.restore_failures.saturating_add(1);
            self.namespace_rejections = self.namespace_rejections.saturating_add(1);
            return Err(Error::NamespaceMismatch {
                expected: expected_hash.0,
                actual: key.namespace_hash.0.clone(),
            });
        }

        let block_id = key.block_id.clone();
        let Some(entry) = self.index.get(&block_id).cloned() else {
            self.misses = self.misses.saturating_add(1);
            return Err(Error::NotFound {
                block_id: block_id.0,
            });
        };

        let bytes = match fs::read(self.root.join(&entry.relative_path)) {
            Ok(bytes) => bytes,
            Err(source) => {
                self.misses = self.misses.saturating_add(1);
                return Err(Error::Io(source.to_string()));
            }
        };
        self.reads = self.reads.saturating_add(1);
        self.bytes_read = self.bytes_read.saturating_add(bytes.len() as u64);

        let file = match serde_json::from_slice::<PersistedBlockFile>(&bytes) {
            Ok(file) => file,
            Err(source) => {
                self.restore_failures = self.restore_failures.saturating_add(1);
                self.corruptions = self.corruptions.saturating_add(1);
                return Err(Error::Json(source.to_string()));
            }
        };

        let mut block = match validate_persisted_file(file) {
            Ok(block) => block,
            Err(error) => {
                self.restore_failures = self.restore_failures.saturating_add(1);
                if matches!(
                    error,
                    Error::ChecksumMismatch { .. } | Error::InvalidManifest(_)
                ) {
                    self.corruptions = self.corruptions.saturating_add(1);
                }
                return Err(error);
            }
        };

        let actual_hash = block.namespace.namespace_hash()?;
        if actual_hash != expected_hash {
            self.restore_failures = self.restore_failures.saturating_add(1);
            self.namespace_rejections = self.namespace_rejections.saturating_add(1);
            return Err(Error::NamespaceMismatch {
                expected: expected_hash.0,
                actual: actual_hash.0,
            });
        }
        if block.key != *key {
            self.restore_failures = self.restore_failures.saturating_add(1);
            self.corruptions = self.corruptions.saturating_add(1);
            return Err(Error::InvalidManifest(
                "persisted block key does not match requested key".to_owned(),
            ));
        }

        block.native_handle = None;
        let restored = RestoredPrefix {
            block_id: block.key.block_id.clone(),
            namespace_hash: block.key.namespace_hash.clone(),
            sequence_start: block.key.sequence_start,
            sequence_end: block.key.sequence_end,
            byte_len: block.byte_len,
            observation: block.observation(),
            native_handle: None,
        };
        self.hits = self.hits.saturating_add(1);
        self.touch(&block_id);
        self.persist_index()?;
        Ok(restored)
    }

    pub fn contains(&self, block_id: &BlockId) -> bool {
        self.index.contains_key(block_id)
    }

    pub fn entry_path(&self, entry: &SsdIndexEntry) -> PathBuf {
        self.root.join(&entry.relative_path)
    }

    pub fn accounting(&self) -> SsdCacheAccountingSnapshot {
        let total = self.hits + self.misses;
        SsdCacheAccountingSnapshot {
            budget_bytes: self.budget_bytes,
            stored_bytes: self.stored_bytes,
            stored_blocks: self.index.len(),
            hits: self.hits,
            misses: self.misses,
            writes: self.writes,
            reads: self.reads,
            evictions: self.evictions,
            restore_failures: self.restore_failures,
            namespace_rejections: self.namespace_rejections,
            corruptions: self.corruptions,
            bytes_written: self.bytes_written,
            bytes_read: self.bytes_read,
            hit_rate: if total == 0 {
                0.0
            } else {
                self.hits as f64 / total as f64
            },
            mid_decode_fetches: self.mid_decode_fetches,
            ssd_enabled: true,
        }
    }

    fn load_index(&mut self) -> Result<()> {
        let index_path = self.root.join(SSD_INDEX_FILE);
        if !index_path.exists() {
            return Ok(());
        }
        let bytes = fs::read(index_path)?;
        let persisted: PersistedSsdIndex = serde_json::from_slice(&bytes)?;
        if persisted.schema_version != SSD_MANIFEST_VERSION {
            return Err(Error::InvalidManifest(format!(
                "unsupported SSD index version {}",
                persisted.schema_version
            )));
        }
        for entry in persisted.entries {
            if self.root.join(&entry.relative_path).exists() {
                self.stored_bytes = self.stored_bytes.saturating_add(entry.stored_bytes);
                self.index.insert(entry.block_id.clone(), entry);
            }
        }
        for block_id in persisted.lru {
            if self.index.contains_key(&block_id) && !self.lru.contains(&block_id) {
                self.lru.push_back(block_id);
            }
        }
        for block_id in self.index.keys() {
            if !self.lru.contains(block_id) {
                self.lru.push_back(block_id.clone());
            }
        }
        Ok(())
    }

    fn persist_index(&self) -> Result<()> {
        let persisted = PersistedSsdIndex {
            schema_version: SSD_MANIFEST_VERSION,
            budget_bytes: self.budget_bytes,
            entries: self.index.values().cloned().collect(),
            lru: self.lru.iter().cloned().collect(),
        };
        let bytes = serde_json::to_vec_pretty(&persisted)?;
        fs::write(self.root.join(SSD_INDEX_FILE), bytes)?;
        Ok(())
    }

    fn evict_to_fit(&mut self, incoming_bytes: u64, protected: Option<&BlockId>) -> Result<()> {
        while self.stored_bytes + incoming_bytes > self.budget_bytes {
            let Some(victim_id) = self.lru.pop_front() else {
                break;
            };
            if protected == Some(&victim_id) {
                self.lru.push_back(victim_id);
                break;
            }
            if let Some(victim) = self.index.remove(&victim_id) {
                let path = self.root.join(&victim.relative_path);
                match fs::remove_file(path) {
                    Ok(()) => {}
                    Err(source) if source.kind() == std::io::ErrorKind::NotFound => {}
                    Err(source) => return Err(Error::Io(source.to_string())),
                }
                self.stored_bytes = self.stored_bytes.saturating_sub(victim.stored_bytes);
                self.evictions = self.evictions.saturating_add(1);
            }
        }
        Ok(())
    }

    fn touch(&mut self, block_id: &BlockId) {
        self.remove_lru(block_id);
        self.lru.push_back(block_id.clone());
    }

    fn remove_lru(&mut self, block_id: &BlockId) {
        if let Some(index) = self.lru.iter().position(|candidate| candidate == block_id) {
            self.lru.remove(index);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedBlockFile {
    file_version: u32,
    manifest: PersistedKvManifest,
    block: RamPrefixBlock,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedSsdIndex {
    schema_version: u32,
    budget_bytes: u64,
    entries: Vec<SsdIndexEntry>,
    lru: Vec<BlockId>,
}

#[derive(Debug, Clone)]
pub struct RamPrefixCache {
    budget_bytes: u64,
    resident_bytes: u64,
    blocks: HashMap<BlockId, RamPrefixBlock>,
    lru: VecDeque<BlockId>,
    hits: u64,
    misses: u64,
    evictions: u64,
    restore_failures: u64,
}

impl RamPrefixCache {
    pub fn new(budget_bytes: NonZeroU64) -> Self {
        Self {
            budget_bytes: budget_bytes.get(),
            resident_bytes: 0,
            blocks: HashMap::new(),
            lru: VecDeque::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
            restore_failures: 0,
        }
    }

    pub fn insert(&mut self, block: RamPrefixBlock) -> Result<Vec<BlockId>> {
        if block.byte_len > self.budget_bytes {
            return Err(Error::BudgetExceeded {
                block_bytes: block.byte_len,
                budget_bytes: self.budget_bytes,
            });
        }

        let id = block.key.block_id.clone();
        if let Some(previous) = self.blocks.remove(&id) {
            self.resident_bytes = self.resident_bytes.saturating_sub(previous.byte_len);
            self.remove_lru(&id);
        }

        let mut evicted = Vec::new();
        while self.resident_bytes + block.byte_len > self.budget_bytes {
            let Some(victim_id) = self.lru.pop_front() else {
                break;
            };
            if let Some(victim) = self.blocks.remove(&victim_id) {
                self.resident_bytes = self.resident_bytes.saturating_sub(victim.byte_len);
                self.evictions = self.evictions.saturating_add(1);
                evicted.push(victim_id);
            }
        }

        self.resident_bytes += block.byte_len;
        self.blocks.insert(id.clone(), block);
        self.lru.push_back(id);
        Ok(evicted)
    }

    pub fn restore(
        &mut self,
        key: &KvBlockKey,
        expected_namespace: &KvNamespace,
    ) -> Result<RestoredPrefix> {
        let expected_hash = expected_namespace.namespace_hash()?;
        if key.namespace_hash != expected_hash {
            self.restore_failures = self.restore_failures.saturating_add(1);
            return Err(Error::NamespaceMismatch {
                expected: expected_hash.0,
                actual: key.namespace_hash.0.clone(),
            });
        }

        let block_id = key.block_id.clone();
        let Some(block) = self.blocks.get(&block_id) else {
            self.misses = self.misses.saturating_add(1);
            return Err(Error::NotFound {
                block_id: block_id.0,
            });
        };

        let actual_hash = block.namespace.namespace_hash()?;
        if actual_hash != expected_hash {
            self.restore_failures = self.restore_failures.saturating_add(1);
            return Err(Error::NamespaceMismatch {
                expected: expected_hash.0,
                actual: actual_hash.0,
            });
        }
        block.verify_checksum().inspect_err(|_| {
            self.restore_failures = self.restore_failures.saturating_add(1);
        })?;

        let restored = RestoredPrefix {
            block_id: block.key.block_id.clone(),
            namespace_hash: block.key.namespace_hash.clone(),
            sequence_start: block.key.sequence_start,
            sequence_end: block.key.sequence_end,
            byte_len: block.byte_len,
            observation: block.observation(),
            native_handle: block.native_handle.clone(),
        };
        self.hits = self.hits.saturating_add(1);
        self.touch(&block_id);
        Ok(restored)
    }

    pub fn contains(&self, block_id: &BlockId) -> bool {
        self.blocks.contains_key(block_id)
    }

    pub fn accounting(&self) -> CacheAccountingSnapshot {
        let total = self.hits + self.misses;
        CacheAccountingSnapshot {
            budget_bytes: self.budget_bytes,
            resident_bytes: self.resident_bytes,
            resident_blocks: self.blocks.len(),
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            restore_failures: self.restore_failures,
            hit_rate: if total == 0 {
                0.0
            } else {
                self.hits as f64 / total as f64
            },
            ssd_enabled: false,
        }
    }

    fn touch(&mut self, block_id: &BlockId) {
        self.remove_lru(block_id);
        self.lru.push_back(block_id.clone());
    }

    fn remove_lru(&mut self, block_id: &BlockId) {
        if let Some(index) = self.lru.iter().position(|candidate| candidate == block_id) {
            self.lru.remove(index);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConversationFork {
    pub fork_id: String,
    pub parent_fork_id: Option<String>,
    pub shared_prefix_blocks: Vec<BlockId>,
    pub private_suffix_blocks: Vec<BlockId>,
}

impl ConversationFork {
    pub fn root(shared_prefix_blocks: Vec<BlockId>) -> Self {
        let fork_id = fork_hash("root", None, &shared_prefix_blocks, &[]);
        Self {
            fork_id,
            parent_fork_id: None,
            shared_prefix_blocks,
            private_suffix_blocks: Vec::new(),
        }
    }

    pub fn fork(&self, new_private_suffix: Vec<BlockId>) -> Self {
        let fork_id = fork_hash(
            "fork",
            Some(&self.fork_id),
            &self.shared_prefix_blocks,
            &new_private_suffix,
        );
        Self {
            fork_id,
            parent_fork_id: Some(self.fork_id.clone()),
            shared_prefix_blocks: self.shared_prefix_blocks.clone(),
            private_suffix_blocks: new_private_suffix,
        }
    }
}

pub fn prompt_token_hash(sequence_len: u64) -> String {
    sha256_bytes(
        b"gemma4d:prompt-token-prefix:v1\0",
        &sequence_len.to_le_bytes(),
    )
}

pub fn fresh_prefill_fixture(sequence_len: u64) -> PrefillObservation {
    let digest = sha256_bytes(
        b"gemma4d:m07:fresh-prefill:v1\0",
        &sequence_len.to_le_bytes(),
    );
    let token = u32::from_le_bytes(
        hex::decode(&digest[..8])
            .expect("hex")
            .try_into()
            .expect("four bytes"),
    ) % 262_144;
    let logit = 10.0 + (sequence_len as f32 / 1024.0);
    PrefillObservation {
        sequence_len,
        greedy_token: token,
        greedy_logit_bits: logit.to_bits(),
    }
}

pub fn fixture_block(sequence_len: u64, block_size_tokens: NonZeroU64) -> Result<RamPrefixBlock> {
    fixture_block_with_mode(sequence_len, block_size_tokens, CacheMode::Bf16)
}

pub fn fixture_block_with_mode(
    sequence_len: u64,
    block_size_tokens: NonZeroU64,
    cache_mode: CacheMode,
) -> Result<RamPrefixBlock> {
    let namespace = KvNamespace::fixture(sequence_len);
    let namespace = namespace.with_cache_mode(cache_mode);
    let observation = fresh_prefill_fixture(sequence_len);
    let byte_len = estimated_kv_bytes_for_mode(sequence_len, cache_mode);
    RamPrefixBlock::from_observation(namespace, 0, block_size_tokens, 0, observation, byte_len)
        .map(|block| block.with_native_handle(sequence_len))
}

pub fn estimated_bf16_kv_bytes(token_count: u64) -> u64 {
    estimated_kv_bytes_for_mode(token_count, CacheMode::Bf16)
}

pub fn estimated_kv_bytes_for_mode(token_count: u64, cache_mode: CacheMode) -> u64 {
    const LAYERS: u64 = 48;
    const APPROX_KV_WIDTH: u64 = 16 * 256 * 2;
    let values = token_count * LAYERS * APPROX_KV_WIDTH;
    match cache_mode {
        CacheMode::Bf16 => values * 2,
        CacheMode::MlxAffineQ8 => values + compression_metadata_overhead(token_count),
        CacheMode::MlxAffineQ4 => values.div_ceil(2) + compression_metadata_overhead(token_count),
    }
}

fn compression_metadata_overhead(token_count: u64) -> u64 {
    const LAYERS: u64 = 48;
    const SCALE_AND_BIAS_BYTES: u64 = 8;
    LAYERS * SCALE_AND_BIAS_BYTES + (token_count / 1024).max(1) * 64
}

fn fixture_logits(sequence_len: u64, workload: CompressionWorkload) -> Vec<f32> {
    let mut logits = Vec::with_capacity(128);
    for index in 0..128_u32 {
        let input = format!("{}:{sequence_len}:{index}", workload.label());
        let digest = sha256_bytes(b"gemma4d:m09:logits:v1\0", input.as_bytes());
        let raw = u32::from_le_bytes(
            hex::decode(&digest[..8])
                .expect("hex")
                .try_into()
                .expect("four bytes"),
        );
        let centered = (raw as f32 / u32::MAX as f32) * 2.0 - 1.0;
        logits.push(centered * 3.0 + (index as f32 % 7.0) * 0.03);
    }
    let greedy_index = (fresh_prefill_fixture(sequence_len).greedy_token as usize) % logits.len();
    logits[greedy_index] += 18.0;
    logits
}

fn affine_reconstruct(values: &[f32], mode: CacheMode) -> Vec<f32> {
    if mode == CacheMode::Bf16 || values.is_empty() {
        return values.to_vec();
    }
    let levels = match mode {
        CacheMode::Bf16 => unreachable!(),
        CacheMode::MlxAffineQ8 => 255.0,
        CacheMode::MlxAffineQ4 => 15.0,
    };
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let scale = ((max - min) / levels).max(f32::EPSILON);
    values
        .iter()
        .map(|value| {
            let quantized = ((*value - min) / scale).round().clamp(0.0, levels);
            quantized * scale + min
        })
        .collect()
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f64 {
    let (mut dot, mut left_norm, mut right_norm) = (0.0, 0.0, 0.0);
    for (left, right) in left.iter().zip(right.iter()) {
        let left = *left as f64;
        let right = *right as f64;
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        dot / (left_norm.sqrt() * right_norm.sqrt())
    }
}

fn argmax(values: &[f32]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
}

pub fn default_gemma4_layers(
    sequence_start: u64,
    sequence_end: u64,
    compression: CacheMode,
) -> Vec<LayerBlockMetadata> {
    (0..48)
        .map(|layer| {
            let attention_type = if (layer + 1) % 6 == 0 {
                AttentionType::Full
            } else {
                AttentionType::Sliding
            };
            let token_count = sequence_end - sequence_start;
            LayerBlockMetadata {
                layer,
                attention_type,
                absolute_start: sequence_start,
                absolute_end: sequence_end,
                block_local_start: 0,
                block_local_end: token_count,
                sliding_window_local_start: (attention_type == AttentionType::Sliding)
                    .then_some(sequence_end.saturating_sub(1024)),
                sliding_window_local_end: (attention_type == AttentionType::Sliding)
                    .then_some(sequence_end),
                full_attention_cumulative_len: (attention_type == AttentionType::Full)
                    .then_some(sequence_end),
                kv_shared_from_layer: (layer >= 28).then_some(layer % 6),
                physical_stored_tensors: if attention_type == AttentionType::Full {
                    1
                } else {
                    2
                },
                head_dim: if attention_type == AttentionType::Full {
                    512
                } else {
                    256
                },
                kv_heads: if attention_type == AttentionType::Full {
                    1
                } else {
                    8
                },
                compression,
            }
        })
        .collect()
}

fn checksum_payload(key: &KvBlockKey, payload: &StoredPrefixPayload) -> Result<String> {
    sha256_json(&(key, payload))
}

fn checksum_block(block: &RamPrefixBlock) -> Result<String> {
    sha256_json(block)
}

fn serialized_block_bytes(block: &RamPrefixBlock) -> Result<Vec<u8>> {
    let manifest = PersistedKvManifest::from_block(block)?;
    let file = PersistedBlockFile {
        file_version: SSD_BLOCK_FILE_VERSION,
        manifest,
        block: block.clone(),
    };
    Ok(serde_json::to_vec_pretty(&file)?)
}

fn validate_persisted_file(file: PersistedBlockFile) -> Result<RamPrefixBlock> {
    if file.file_version != SSD_BLOCK_FILE_VERSION {
        return Err(Error::InvalidManifest(format!(
            "unsupported SSD block file version {}",
            file.file_version
        )));
    }
    file.block.verify_checksum()?;
    file.manifest.validate_for_block(&file.block)?;
    Ok(file.block)
}

fn fork_hash(
    label: &str,
    parent: Option<&str>,
    shared_prefix_blocks: &[BlockId],
    private_suffix_blocks: &[BlockId],
) -> String {
    #[derive(Serialize)]
    struct Inputs<'a> {
        label: &'a str,
        parent: Option<&'a str>,
        shared_prefix_blocks: &'a [BlockId],
        private_suffix_blocks: &'a [BlockId],
    }

    sha256_json(&Inputs {
        label,
        parent,
        shared_prefix_blocks,
        private_suffix_blocks,
    })
    .expect("fork hash inputs are serializable")
}

fn sha256_json<T: Serialize>(value: &T) -> Result<String> {
    let bytes = serde_json::to_vec(value)?;
    Ok(sha256_bytes(b"gemma4d:json:v1\0", &bytes))
}

fn sha256_bytes(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_m09_status() {
        assert_eq!(CRATE_NAME, "gemma4d-kv");
        assert_eq!(bootstrap_status(), "m09-kv-compression-research");
    }

    #[test]
    fn namespace_hash_changes_for_required_fields() {
        let base = KvNamespace::fixture(1024);
        let base_hash = base.namespace_hash().expect("hash");

        let mut wrong_model = base.clone();
        wrong_model.model_id = "other-model".to_owned();
        assert_ne!(base_hash, wrong_model.namespace_hash().expect("hash"));

        let mut wrong_template = base.clone();
        wrong_template.chat_template_sha256 = "other-template".to_owned();
        assert_ne!(base_hash, wrong_template.namespace_hash().expect("hash"));

        let mut wrong_prompt = base.clone();
        wrong_prompt.prompt_token_hash = "other-prompt".to_owned();
        assert_ne!(base_hash, wrong_prompt.namespace_hash().expect("hash"));
    }

    #[test]
    fn adapter_identity_and_weight_hash_partition_namespace_and_blocks() {
        let block_size = NonZeroU64::new(1024).expect("non-zero");
        let mut first = KvNamespace::fixture(1024);
        first.adapter_id = Some("rust-coding-r16-v1".to_owned());
        first.adapter_weight_hash = Some("adapter-weight-hash-a".to_owned());
        let mut second = first.clone();
        second.adapter_id = Some("sql-r16-v1".to_owned());
        second.adapter_weight_hash = Some("adapter-weight-hash-b".to_owned());

        assert_ne!(
            first.namespace_hash().expect("first hash"),
            second.namespace_hash().expect("second hash")
        );

        let first_key = KvBlockKey::new(&first, 0, block_size, 0, 1024).expect("first key");
        let second_key = KvBlockKey::new(&second, 0, block_size, 0, 1024).expect("second key");
        assert_ne!(first_key.namespace_hash, second_key.namespace_hash);
        assert_ne!(first_key.block_id, second_key.block_id);
    }

    #[test]
    fn adapter_namespace_mismatch_rejects_ram_restore() {
        let block_size = NonZeroU64::new(1024).expect("non-zero");
        let mut namespace = KvNamespace::fixture(1024);
        namespace.adapter_id = Some("rust-coding-r16-v1".to_owned());
        namespace.adapter_weight_hash = Some("adapter-weight-hash-a".to_owned());
        let observation = fresh_prefill_fixture(1024);
        let block = RamPrefixBlock::from_observation(
            namespace.clone(),
            0,
            block_size,
            0,
            observation,
            estimated_bf16_kv_bytes(1024),
        )
        .expect("adapter block");
        let key = block.key.clone();
        let mut wrong_namespace = namespace;
        wrong_namespace.adapter_weight_hash = Some("adapter-weight-hash-b".to_owned());
        let mut cache = RamPrefixCache::new(NonZeroU64::new(block.byte_len * 2).expect("non-zero"));
        cache.insert(block).expect("insert");

        let err = cache
            .restore(&key, &wrong_namespace)
            .expect_err("wrong adapter namespace should reject");
        assert!(matches!(err, Error::NamespaceMismatch { .. }));
    }

    #[test]
    fn adapter_namespace_mismatch_rejects_ssd_restore() {
        let root = temp_cache_dir("adapter_namespace");
        let block_size = NonZeroU64::new(1024).expect("non-zero");
        let mut namespace = KvNamespace::fixture(1024);
        namespace.adapter_id = Some("rust-coding-r16-v1".to_owned());
        namespace.adapter_weight_hash = Some("adapter-weight-hash-a".to_owned());
        let observation = fresh_prefill_fixture(1024);
        let block = RamPrefixBlock::from_observation(
            namespace.clone(),
            0,
            block_size,
            0,
            observation,
            estimated_bf16_kv_bytes(1024),
        )
        .expect("adapter block");
        let key = block.key.clone();
        let mut wrong_namespace = namespace;
        wrong_namespace.adapter_id = Some("sql-r16-v1".to_owned());
        wrong_namespace.adapter_weight_hash = Some("adapter-weight-hash-b".to_owned());
        let mut cache =
            SsdPrefixCache::open(&root, NonZeroU64::new(8 * 1024 * 1024).expect("non-zero"))
                .expect("cache");
        cache.write_block(&block).expect("write");

        let err = cache
            .restore_before_prefill(&key, &wrong_namespace)
            .expect_err("wrong adapter namespace should reject");
        assert!(matches!(err, Error::NamespaceMismatch { .. }));
        assert_eq!(cache.accounting().namespace_rejections, 1);
        cleanup_temp_dir(root);
    }

    #[test]
    fn restore_matches_fresh_prefill_for_m07_context_lengths() {
        for sequence_len in [1024, 4096, 8192, 16384] {
            let block = fixture_block(sequence_len, NonZeroU64::new(16384).expect("non-zero"))
                .expect("fixture block");
            let key = block.key.clone();
            let namespace = block.namespace.clone();
            let fresh = fresh_prefill_fixture(sequence_len);
            let mut cache =
                RamPrefixCache::new(NonZeroU64::new(64 * 1024 * 1024 * 1024).expect("non-zero"));
            cache.insert(block).expect("insert");

            let restored = cache.restore(&key, &namespace).expect("restore");
            assert_eq!(restored.observation, fresh);
            assert_eq!(restored.sequence_end, sequence_len);
            assert!(restored.native_handle.is_some());
        }
    }

    #[test]
    fn wrong_namespace_and_corruption_are_rejected() {
        let block =
            fixture_block(1024, NonZeroU64::new(1024).expect("non-zero")).expect("fixture block");
        let key = block.key.clone();
        let namespace = block.namespace.clone();
        let mut wrong_namespace = namespace.clone();
        wrong_namespace.model_id = "wrong-model".to_owned();

        let mut cache = RamPrefixCache::new(NonZeroU64::new(block.byte_len * 2).expect("non-zero"));
        cache.insert(block).expect("insert");
        let err = cache
            .restore(&key, &wrong_namespace)
            .expect_err("wrong namespace should reject");
        assert!(matches!(err, Error::NamespaceMismatch { .. }));

        let mut corrupted =
            fixture_block(2048, NonZeroU64::new(2048).expect("non-zero")).expect("fixture block");
        corrupted.corrupt_checksum_for_test();
        let corrupted_key = corrupted.key.clone();
        let corrupted_namespace = corrupted.namespace.clone();
        cache.insert(corrupted).expect("insert corrupted");
        let err = cache
            .restore(&corrupted_key, &corrupted_namespace)
            .expect_err("corrupted block should reject");
        assert!(matches!(err, Error::ChecksumMismatch { .. }));
        assert_eq!(cache.accounting().restore_failures, 2);
    }

    #[test]
    fn ram_lru_evicts_to_budget_and_reports_accounting() {
        let first =
            fixture_block(1024, NonZeroU64::new(1024).expect("non-zero")).expect("first block");
        let second =
            fixture_block(2048, NonZeroU64::new(2048).expect("non-zero")).expect("second block");
        let first_id = first.key.block_id.clone();
        let second_id = second.key.block_id.clone();
        let budget = first.byte_len.max(second.byte_len);
        let mut cache = RamPrefixCache::new(NonZeroU64::new(budget).expect("non-zero"));

        cache.insert(first).expect("insert first");
        let evicted = cache.insert(second).expect("insert second");
        assert_eq!(evicted, vec![first_id.clone()]);
        assert!(!cache.contains(&first_id));
        assert!(cache.contains(&second_id));

        let summary = cache.accounting();
        assert_eq!(summary.resident_blocks, 1);
        assert_eq!(summary.evictions, 1);
        assert!(!summary.ssd_enabled);
    }

    #[test]
    fn copy_on_write_fork_keeps_prefix_shared_and_suffix_private() {
        let shared =
            fixture_block(1024, NonZeroU64::new(1024).expect("non-zero")).expect("shared block");
        let private =
            fixture_block(2048, NonZeroU64::new(2048).expect("non-zero")).expect("private block");

        let root = ConversationFork::root(vec![shared.key.block_id.clone()]);
        let fork = root.fork(vec![private.key.block_id.clone()]);

        assert_eq!(fork.parent_fork_id.as_deref(), Some(root.fork_id.as_str()));
        assert_eq!(fork.shared_prefix_blocks, root.shared_prefix_blocks);
        assert_eq!(fork.private_suffix_blocks, vec![private.key.block_id]);
        assert_ne!(fork.fork_id, root.fork_id);
    }

    #[test]
    fn native_logical_handle_round_trips_metadata() {
        let block =
            fixture_block(4096, NonZeroU64::new(4096).expect("non-zero")).expect("fixture block");
        let handle = block.native_handle.as_ref().expect("native handle");
        assert_eq!(handle.sequence_start, block.key.sequence_start);
        assert_eq!(handle.sequence_end, block.key.sequence_end);
        assert_eq!(handle.namespace_hash, block.key.namespace_hash);
        assert_eq!(handle.byte_len, block.byte_len);
    }

    #[test]
    fn ssd_manifest_records_required_namespace_and_layout_fields() {
        let block =
            fixture_block(4096, NonZeroU64::new(4096).expect("non-zero")).expect("fixture block");
        let manifest = PersistedKvManifest::from_block(&block).expect("manifest");

        assert_eq!(manifest.manifest_version, SSD_MANIFEST_VERSION);
        assert_eq!(manifest.block_file_version, SSD_BLOCK_FILE_VERSION);
        assert_eq!(manifest.kv_layout_version, KV_LAYOUT_VERSION);
        assert_eq!(manifest.model_id, block.namespace.model_id);
        assert_eq!(manifest.model_revision, block.namespace.model_revision);
        assert_eq!(manifest.weights_sha256, block.namespace.weights_sha256);
        assert_eq!(
            manifest.quantization_sha256,
            block.namespace.quantization_sha256
        );
        assert_eq!(manifest.tokenizer_sha256, block.namespace.tokenizer_sha256);
        assert_eq!(
            manifest.chat_template_sha256,
            block.namespace.chat_template_sha256
        );
        assert_eq!(
            manifest.prompt_token_hash,
            block.namespace.prompt_token_hash
        );
        assert_eq!(manifest.raw_prompt_hash, block.namespace.raw_prompt_hash);
        assert_eq!(manifest.namespace_hash, block.key.namespace_hash);
        assert_eq!(manifest.block_id, block.key.block_id);
        assert_eq!(manifest.layers.len(), 48);
        assert!(
            manifest
                .layers
                .iter()
                .any(|layer| layer.attention_type == AttentionType::Full)
        );
        assert!(
            manifest
                .layers
                .iter()
                .any(|layer| layer.attention_type == AttentionType::Sliding)
        );
    }

    #[test]
    fn ssd_restore_before_prefill_matches_fresh_for_m08_context_lengths() {
        let root = temp_cache_dir("restore_matrix");
        let block_size = NonZeroU64::new(16 * 1024).expect("non-zero");
        let mut cache =
            SsdPrefixCache::open(&root, NonZeroU64::new(32 * 1024 * 1024).expect("non-zero"))
                .expect("cache");

        for sequence_len in [1024, 4096, 8192, 16384] {
            let block = fixture_block(sequence_len, block_size).expect("fixture block");
            let key = block.key.clone();
            let namespace = block.namespace.clone();
            let fresh = fresh_prefill_fixture(sequence_len);
            cache.write_block(&block).expect("write");

            let restored = cache
                .restore_before_prefill(&key, &namespace)
                .expect("ssd restore before prefill");
            assert_eq!(restored.observation, fresh);
            assert_eq!(restored.sequence_end, sequence_len);
            assert!(restored.native_handle.is_none());
        }

        let summary = cache.accounting();
        assert_eq!(summary.hits, 4);
        assert_eq!(summary.reads, 4);
        assert_eq!(summary.writes, 4);
        assert!(summary.bytes_read > 0);
        assert!(summary.bytes_written > 0);
        assert_eq!(summary.mid_decode_fetches, 0);
        assert!(summary.ssd_enabled);
        cleanup_temp_dir(root);
    }

    #[test]
    fn ssd_wrong_namespace_and_corrupt_block_are_rejected() {
        let root = temp_cache_dir("rejects");
        let mut cache =
            SsdPrefixCache::open(&root, NonZeroU64::new(8 * 1024 * 1024).expect("non-zero"))
                .expect("cache");
        let block =
            fixture_block(1024, NonZeroU64::new(1024).expect("non-zero")).expect("fixture block");
        let key = block.key.clone();
        let namespace = block.namespace.clone();
        let entry = cache.write_block(&block).expect("write");

        let mut wrong_namespace = namespace.clone();
        wrong_namespace.model_id = "wrong-model".to_owned();
        let err = cache
            .restore_before_prefill(&key, &wrong_namespace)
            .expect_err("wrong namespace should reject");
        assert!(matches!(err, Error::NamespaceMismatch { .. }));

        let path = cache.entry_path(&entry);
        let mut file = serde_json::from_slice::<PersistedBlockFile>(
            &std::fs::read(&path).expect("read persisted block"),
        )
        .expect("block file");
        file.manifest.block_checksum = "corrupted".to_owned();
        std::fs::write(&path, serde_json::to_vec_pretty(&file).expect("json")).expect("write");

        let err = cache
            .restore_before_prefill(&key, &namespace)
            .expect_err("corruption should reject");
        assert!(matches!(err, Error::ChecksumMismatch { .. }));

        let summary = cache.accounting();
        assert_eq!(summary.namespace_rejections, 1);
        assert_eq!(summary.corruptions, 1);
        assert_eq!(summary.restore_failures, 2);
        cleanup_temp_dir(root);
    }

    #[test]
    fn ssd_lru_evicts_to_disk_budget() {
        let root = temp_cache_dir("eviction");
        let first =
            fixture_block(1024, NonZeroU64::new(1024).expect("non-zero")).expect("first block");
        let second =
            fixture_block(2048, NonZeroU64::new(2048).expect("non-zero")).expect("second block");
        let first_id = first.key.block_id.clone();
        let second_id = second.key.block_id.clone();
        let first_size = serialized_block_bytes(&first).expect("first bytes").len() as u64;
        let second_size = serialized_block_bytes(&second).expect("second bytes").len() as u64;
        let budget = first_size.max(second_size);
        let mut cache =
            SsdPrefixCache::open(&root, NonZeroU64::new(budget).expect("non-zero")).expect("cache");

        cache.write_block(&first).expect("write first");
        cache.write_block(&second).expect("write second");

        assert!(!cache.contains(&first_id));
        assert!(cache.contains(&second_id));
        let summary = cache.accounting();
        assert_eq!(summary.stored_blocks, 1);
        assert_eq!(summary.evictions, 1);
        cleanup_temp_dir(root);
    }

    #[test]
    fn ssd_mid_decode_restore_is_rejected_without_fetching() {
        let root = temp_cache_dir("mid_decode");
        let mut cache =
            SsdPrefixCache::open(&root, NonZeroU64::new(8 * 1024 * 1024).expect("non-zero"))
                .expect("cache");
        let block =
            fixture_block(1024, NonZeroU64::new(1024).expect("non-zero")).expect("fixture block");
        let key = block.key.clone();
        let namespace = block.namespace.clone();
        cache.write_block(&block).expect("write");

        let err = cache
            .restore_for_phase(&key, &namespace, SsdRestorePhase::MidDecode)
            .expect_err("mid-decode restore should reject");
        assert!(matches!(err, Error::InvalidBlock(_)));
        let summary = cache.accounting();
        assert_eq!(summary.reads, 0);
        assert_eq!(summary.bytes_read, 0);
        assert_eq!(summary.mid_decode_fetches, 0);
        cleanup_temp_dir(root);
    }

    #[test]
    fn bf16_fallback_remains_default_cache_mode() {
        let namespace = KvNamespace::fixture(1024);
        assert_eq!(namespace.cache_mode, CacheMode::Bf16);
        let block =
            fixture_block(1024, NonZeroU64::new(1024).expect("non-zero")).expect("fixture block");
        assert_eq!(block.namespace.cache_mode, CacheMode::Bf16);
        assert!(
            block
                .layers
                .iter()
                .all(|layer| layer.compression == CacheMode::Bf16)
        );
    }

    #[test]
    fn q8_q4_change_namespace_and_manifest_compression_metadata() {
        let block_size = NonZeroU64::new(16 * 1024).expect("non-zero");
        let bf16 =
            fixture_block_with_mode(16 * 1024, block_size, CacheMode::Bf16).expect("bf16 block");
        let q8 = fixture_block_with_mode(16 * 1024, block_size, CacheMode::MlxAffineQ8)
            .expect("q8 block");
        let q4 = fixture_block_with_mode(16 * 1024, block_size, CacheMode::MlxAffineQ4)
            .expect("q4 block");

        assert_ne!(bf16.key.namespace_hash, q8.key.namespace_hash);
        assert_ne!(q8.key.namespace_hash, q4.key.namespace_hash);
        assert_ne!(bf16.key.block_id, q8.key.block_id);
        assert_ne!(q8.key.block_id, q4.key.block_id);

        let q8_manifest = PersistedKvManifest::from_block(&q8).expect("q8 manifest");
        let q4_manifest = PersistedKvManifest::from_block(&q4).expect("q4 manifest");
        assert_eq!(q8_manifest.compression.mode, CacheMode::MlxAffineQ8);
        assert_eq!(q8_manifest.compression.bits_per_value, 8);
        assert!(q8_manifest.compression.namespace_hash_includes_mode);
        assert_eq!(q4_manifest.compression.mode, CacheMode::MlxAffineQ4);
        assert_eq!(q4_manifest.compression.bits_per_value, 4);
        assert!(q4.byte_len < q8.byte_len);
        assert!(q8.byte_len < bf16.byte_len);
    }

    #[test]
    fn q8_q4_quality_gates_record_quality_and_memory_deltas() {
        for mode in [CacheMode::MlxAffineQ8, CacheMode::MlxAffineQ4] {
            let summary =
                evaluate_compression_fixture(16 * 1024, CompressionWorkload::JsonToolFixture, mode);
            assert!(
                summary.accepted,
                "{mode:?} should pass fixture gate: {summary:?}"
            );
            assert!(summary.greedy_agreement);
            assert!(summary.logit_cosine >= summary.gate.min_logit_cosine);
            assert!(summary.memory_delta_bytes < 0);
            assert!(summary.memory_reduction > 0.0);
        }
    }

    #[cfg(feature = "planar-iso-experiments")]
    #[test]
    fn planar_iso_interface_stays_experimental_by_default() {
        let candidates = ExperimentalCompressionPlan::candidates();
        assert_eq!(candidates.len(), 4);
        assert!(candidates.iter().all(|candidate| !candidate.accepted));
        assert!(
            candidates
                .iter()
                .all(|candidate| candidate.feature_flag == "planar-iso-experiments")
        );
    }

    fn temp_cache_dir(label: &str) -> std::path::PathBuf {
        let unique = format!(
            "gemma4d-kv-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        );
        std::env::temp_dir().join(unique)
    }

    fn cleanup_temp_dir(path: std::path::PathBuf) {
        let _ = std::fs::remove_dir_all(path);
    }
}
