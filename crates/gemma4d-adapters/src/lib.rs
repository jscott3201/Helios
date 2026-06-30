#![doc = "Dynamic LoRA/QLoRA adapter manifests, trust policy, registry state, and request routing."]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fmt, fs,
    path::{Path, PathBuf},
    time::Instant,
};

pub const CRATE_NAME: &str = "gemma4d-adapters";
pub const REGISTRY_FILE: &str = "registry.json";

pub type Result<T> = std::result::Result<T, Error>;

pub fn bootstrap_status() -> &'static str {
    "m10-dynamic-adapters"
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    AdapterNotFound { adapter_id: String },
    InvalidManifest(String),
    InvalidPeftConfig(String),
    InvalidSafetensors(String),
    Io(String),
    Json(String),
    UntrustedPath { root: PathBuf, candidate: PathBuf },
    UnsupportedAdapter(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AdapterNotFound { adapter_id } => write!(f, "adapter not found: {adapter_id}"),
            Self::InvalidManifest(message) => f.write_str(message),
            Self::InvalidPeftConfig(message) => f.write_str(message),
            Self::InvalidSafetensors(message) => f.write_str(message),
            Self::Io(message) => f.write_str(message),
            Self::Json(message) => f.write_str(message),
            Self::UntrustedPath { root, candidate } => write!(
                f,
                "adapter path {} is outside trusted root {}",
                candidate.display(),
                root.display()
            ),
            Self::UnsupportedAdapter(message) => f.write_str(message),
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
pub enum AdapterType {
    Lora,
    Qlora,
    Alora,
}

impl AdapterType {
    fn is_m10_supported(self) -> bool {
        matches!(self, Self::Lora | Self::Qlora)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceFormat {
    Peft,
    MlxLm,
    Internal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AdapterDType {
    Bf16,
    Fp16,
    Fp32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MtpSupport {
    #[default]
    Unknown,
    False,
    True,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdapterManifest {
    pub adapter_id: String,
    pub adapter_type: AdapterType,
    pub base_model_id: String,
    pub base_weight_hash: String,
    pub tokenizer_hash: String,
    pub chat_template_hash: String,
    pub rank: u32,
    pub alpha: f64,
    pub target_modules: Vec<String>,
    pub adapter_weight_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_format: Option<SourceFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_model_revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dropout: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<AdapterDType>,
    #[serde(default)]
    pub modules_to_save: Vec<String>,
    #[serde(default)]
    pub requires_tokenizer_changes: bool,
    #[serde(default)]
    pub supports_mtp: MtpSupport,
}

impl AdapterManifest {
    pub fn from_json_str(value: &str) -> Result<Self> {
        let manifest: Self = serde_json::from_str(value)?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<()> {
        require_non_empty(&self.adapter_id, "adapter_id")?;
        require_non_empty(&self.base_model_id, "base_model_id")?;
        require_non_empty(&self.base_weight_hash, "base_weight_hash")?;
        require_non_empty(&self.tokenizer_hash, "tokenizer_hash")?;
        require_non_empty(&self.chat_template_hash, "chat_template_hash")?;
        require_non_empty(&self.adapter_weight_hash, "adapter_weight_hash")?;
        if !self.adapter_type.is_m10_supported() {
            return Err(Error::UnsupportedAdapter(format!(
                "adapter_type {:?} is not supported by M10 standard LoRA routing",
                self.adapter_type
            )));
        }
        if self.rank == 0 || self.rank > 256 {
            return Err(Error::InvalidManifest(format!(
                "rank must be between 1 and 256, got {}",
                self.rank
            )));
        }
        if self.alpha <= 0.0 || !self.alpha.is_finite() {
            return Err(Error::InvalidManifest(format!(
                "alpha must be finite and positive, got {}",
                self.alpha
            )));
        }
        if self.target_modules.is_empty() {
            return Err(Error::InvalidManifest(
                "target_modules must not be empty".to_owned(),
            ));
        }
        if self
            .target_modules
            .iter()
            .any(|module| module.trim().is_empty())
        {
            return Err(Error::InvalidManifest(
                "target_modules must not contain empty module names".to_owned(),
            ));
        }
        if !self.modules_to_save.is_empty() {
            return Err(Error::UnsupportedAdapter(
                "modules_to_save is rejected for M10 standard LoRA MVP".to_owned(),
            ));
        }
        if self.requires_tokenizer_changes {
            return Err(Error::UnsupportedAdapter(
                "adapters requiring tokenizer changes are rejected".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterCompatibility {
    pub base_model_id: String,
    pub base_weight_hash: String,
    pub tokenizer_hash: String,
    pub chat_template_hash: String,
}

impl AdapterCompatibility {
    pub fn validate(&self, manifest: &AdapterManifest) -> Result<()> {
        compare_manifest_field(
            "base_model_id",
            &self.base_model_id,
            &manifest.base_model_id,
        )?;
        compare_manifest_field(
            "base_weight_hash",
            &self.base_weight_hash,
            &manifest.base_weight_hash,
        )?;
        compare_manifest_field(
            "tokenizer_hash",
            &self.tokenizer_hash,
            &manifest.tokenizer_hash,
        )?;
        compare_manifest_field(
            "chat_template_hash",
            &self.chat_template_hash,
            &manifest.chat_template_hash,
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrustedPathPolicy {
    root: PathBuf,
}

impl TrustedPathPolicy {
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().canonicalize()?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn require_trusted(&self, candidate: impl AsRef<Path>) -> Result<PathBuf> {
        let candidate = candidate.as_ref().canonicalize()?;
        if candidate.starts_with(&self.root) {
            Ok(candidate)
        } else {
            Err(Error::UntrustedPath {
                root: self.root.clone(),
                candidate,
            })
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetensorsValidation {
    pub tensor_count: usize,
    pub lora_a_tensors: usize,
    pub lora_b_tensors: usize,
    pub resident_bytes: u64,
    pub shape_validation_result: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportedAdapter {
    pub manifest: AdapterManifest,
    pub source_path: PathBuf,
    pub config_path: PathBuf,
    pub weights_path: PathBuf,
    pub validation: SafetensorsValidation,
    pub load_latency_us: u128,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegistryEntry {
    pub adapter: ImportedAdapter,
    pub loaded: bool,
    pub pinned: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterSummary {
    pub adapter_id: String,
    pub display_name: Option<String>,
    pub adapter_type: AdapterType,
    pub source_path: PathBuf,
    pub loaded: bool,
    pub pinned: bool,
    pub active: bool,
    pub resident_bytes: u64,
    pub load_latency_us: u128,
    pub target_modules: Vec<String>,
    pub supports_mtp: MtpSupport,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterRoute {
    pub active_adapter_id: Option<String>,
    pub adapter_weight_hash: Option<String>,
    pub mtp_enabled: bool,
    pub mtp_disable_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
struct RegistryIndex {
    entries: Vec<RegistryEntry>,
    active_adapter_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AdapterRegistry {
    registry_dir: PathBuf,
    index: RegistryIndex,
}

impl AdapterRegistry {
    pub fn open(registry_dir: impl AsRef<Path>) -> Result<Self> {
        let registry_dir = registry_dir.as_ref().to_path_buf();
        fs::create_dir_all(&registry_dir)?;
        let index_path = registry_dir.join(REGISTRY_FILE);
        let index = if index_path.exists() {
            let raw = fs::read_to_string(&index_path)?;
            serde_json::from_str(&raw)?
        } else {
            RegistryIndex::default()
        };
        Ok(Self {
            registry_dir,
            index,
        })
    }

    pub fn import_peft(
        &mut self,
        source_dir: impl AsRef<Path>,
        trusted: &TrustedPathPolicy,
        expected: &AdapterCompatibility,
    ) -> Result<ImportedAdapter> {
        let started = Instant::now();
        let source_path = trusted.require_trusted(source_dir)?;
        let config_path = source_path.join("adapter_config.json");
        let weights_path = source_path.join("adapter_model.safetensors");
        let config = PeftAdapterConfig::from_path(&config_path)?;
        let validation = validate_safetensors(&weights_path)?;
        let manifest = config.to_manifest(
            &source_path,
            expected,
            &sha256_file(&weights_path)?,
            Some(validation.clone()),
        )?;
        expected.validate(&manifest)?;
        manifest.validate()?;
        let imported = ImportedAdapter {
            manifest,
            source_path,
            config_path,
            weights_path,
            validation,
            load_latency_us: started.elapsed().as_micros(),
        };
        self.upsert_entry(imported.clone(), true)?;
        Ok(imported)
    }

    pub fn load(&mut self, adapter_id: &str) -> Result<AdapterSummary> {
        let active = self.index.active_adapter_id.clone();
        let entry = self.entry_mut(adapter_id)?;
        entry.loaded = true;
        let summary = summary_for_entry(entry, active.as_deref());
        self.persist()?;
        Ok(summary)
    }

    pub fn unload(&mut self, adapter_id: &str) -> Result<AdapterSummary> {
        if self.index.active_adapter_id.as_deref() == Some(adapter_id) {
            self.index.active_adapter_id = None;
        }
        let active = self.index.active_adapter_id.clone();
        let entry = self.entry_mut(adapter_id)?;
        entry.loaded = false;
        let summary = summary_for_entry(entry, active.as_deref());
        self.persist()?;
        Ok(summary)
    }

    pub fn pin(&mut self, adapter_id: &str) -> Result<AdapterSummary> {
        let active = self.index.active_adapter_id.clone();
        let entry = self.entry_mut(adapter_id)?;
        entry.pinned = true;
        let summary = summary_for_entry(entry, active.as_deref());
        self.persist()?;
        Ok(summary)
    }

    pub fn activate_request(&mut self, adapter_id: Option<&str>) -> Result<AdapterRoute> {
        let Some(adapter_id) = adapter_id else {
            self.index.active_adapter_id = None;
            self.persist()?;
            return Ok(AdapterRoute {
                active_adapter_id: None,
                adapter_weight_hash: None,
                mtp_enabled: true,
                mtp_disable_reason: None,
            });
        };

        let entry = self.entry(adapter_id)?;
        if !entry.loaded {
            return Err(Error::AdapterNotFound {
                adapter_id: adapter_id.to_owned(),
            });
        }
        let adapter_weight_hash = entry.adapter.manifest.adapter_weight_hash.clone();
        self.index.active_adapter_id = Some(adapter_id.to_owned());
        self.persist()?;
        Ok(AdapterRoute {
            active_adapter_id: Some(adapter_id.to_owned()),
            adapter_weight_hash: Some(adapter_weight_hash),
            mtp_enabled: false,
            mtp_disable_reason: Some(
                "MTP is disabled for active standard LoRA adapters until per-adapter exactness is verified"
                    .to_owned(),
            ),
        })
    }

    pub fn summaries(&self) -> Vec<AdapterSummary> {
        self.index
            .entries
            .iter()
            .map(|entry| summary_for_entry(entry, self.index.active_adapter_id.as_deref()))
            .collect()
    }

    pub fn total_resident_bytes(&self) -> u64 {
        self.index
            .entries
            .iter()
            .filter(|entry| entry.loaded)
            .map(|entry| entry.adapter.validation.resident_bytes)
            .sum()
    }

    pub fn active_adapter_id(&self) -> Option<&str> {
        self.index.active_adapter_id.as_deref()
    }

    fn upsert_entry(&mut self, adapter: ImportedAdapter, loaded: bool) -> Result<()> {
        if let Some(existing) = self
            .index
            .entries
            .iter_mut()
            .find(|entry| entry.adapter.manifest.adapter_id == adapter.manifest.adapter_id)
        {
            let pinned = existing.pinned;
            *existing = RegistryEntry {
                adapter,
                loaded,
                pinned,
            };
        } else {
            self.index.entries.push(RegistryEntry {
                adapter,
                loaded,
                pinned: false,
            });
        }
        self.persist()
    }

    fn entry(&self, adapter_id: &str) -> Result<&RegistryEntry> {
        self.index
            .entries
            .iter()
            .find(|entry| entry.adapter.manifest.adapter_id == adapter_id)
            .ok_or_else(|| Error::AdapterNotFound {
                adapter_id: adapter_id.to_owned(),
            })
    }

    fn entry_mut(&mut self, adapter_id: &str) -> Result<&mut RegistryEntry> {
        self.index
            .entries
            .iter_mut()
            .find(|entry| entry.adapter.manifest.adapter_id == adapter_id)
            .ok_or_else(|| Error::AdapterNotFound {
                adapter_id: adapter_id.to_owned(),
            })
    }

    fn persist(&self) -> Result<()> {
        let raw = serde_json::to_string_pretty(&self.index)?;
        fs::write(self.registry_dir.join(REGISTRY_FILE), raw)?;
        Ok(())
    }
}

pub fn fixture_generate_token(base_token: u32, route: &AdapterRoute) -> u32 {
    if let Some(adapter_id) = route.active_adapter_id.as_deref() {
        let delta = adapter_id
            .as_bytes()
            .iter()
            .fold(0u32, |acc, byte| acc.wrapping_add(u32::from(*byte)))
            % 97
            + 1;
        base_token.wrapping_add(delta)
    } else {
        base_token
    }
}

#[derive(Debug, Deserialize)]
struct PeftAdapterConfig {
    peft_type: String,
    #[serde(default)]
    base_model_name_or_path: Option<String>,
    r: u32,
    lora_alpha: f64,
    #[serde(default)]
    lora_dropout: Option<f64>,
    target_modules: StringList,
    #[serde(default)]
    modules_to_save: Option<StringList>,
    #[serde(default)]
    gemma4d: Option<PeftGemma4dMetadata>,
}

impl PeftAdapterConfig {
    fn from_path(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path)?;
        serde_json::from_str(&raw).map_err(|error| Error::InvalidPeftConfig(error.to_string()))
    }

    fn to_manifest(
        &self,
        source_path: &Path,
        expected: &AdapterCompatibility,
        adapter_weight_hash: &str,
        validation: Option<SafetensorsValidation>,
    ) -> Result<AdapterManifest> {
        if !self.peft_type.eq_ignore_ascii_case("LORA") {
            return Err(Error::UnsupportedAdapter(format!(
                "PEFT peft_type {} is not supported by M10",
                self.peft_type
            )));
        }
        let modules_to_save = self
            .modules_to_save
            .as_ref()
            .map(StringList::values)
            .unwrap_or_default();
        if !modules_to_save.is_empty() {
            return Err(Error::UnsupportedAdapter(
                "PEFT modules_to_save is rejected for M10 standard LoRA MVP".to_owned(),
            ));
        }

        let metadata = self.gemma4d.as_ref();
        let base_model_id = metadata
            .and_then(|metadata| metadata.base_model_id.clone())
            .or_else(|| self.base_model_name_or_path.clone())
            .unwrap_or_else(|| expected.base_model_id.clone());
        let manifest = AdapterManifest {
            adapter_id: metadata
                .and_then(|metadata| metadata.adapter_id.clone())
                .unwrap_or_else(|| fallback_adapter_id(source_path)),
            adapter_type: metadata
                .and_then(|metadata| metadata.adapter_type)
                .unwrap_or(AdapterType::Lora),
            base_model_id,
            base_weight_hash: metadata
                .and_then(|metadata| metadata.base_weight_hash.clone())
                .unwrap_or_else(|| expected.base_weight_hash.clone()),
            tokenizer_hash: metadata
                .and_then(|metadata| metadata.tokenizer_hash.clone())
                .unwrap_or_else(|| expected.tokenizer_hash.clone()),
            chat_template_hash: metadata
                .and_then(|metadata| metadata.chat_template_hash.clone())
                .unwrap_or_else(|| expected.chat_template_hash.clone()),
            rank: self.r,
            alpha: self.lora_alpha,
            target_modules: self.target_modules.values(),
            adapter_weight_hash: adapter_weight_hash.to_owned(),
            display_name: metadata.and_then(|metadata| metadata.display_name.clone()),
            source_format: Some(SourceFormat::Peft),
            source_path: Some(source_path.to_path_buf()),
            base_model_revision: metadata.and_then(|metadata| metadata.base_model_revision.clone()),
            dropout: self.lora_dropout,
            dtype: metadata.and_then(|metadata| metadata.dtype),
            modules_to_save,
            requires_tokenizer_changes: metadata
                .and_then(|metadata| metadata.requires_tokenizer_changes)
                .unwrap_or(false),
            supports_mtp: metadata
                .and_then(|metadata| metadata.supports_mtp)
                .unwrap_or_default(),
        };
        if let Some(validation) = validation {
            validate_targets_present(&manifest.target_modules, &validation)?;
        }
        Ok(manifest)
    }
}

#[derive(Debug, Deserialize)]
struct PeftGemma4dMetadata {
    #[serde(default)]
    adapter_id: Option<String>,
    #[serde(default)]
    display_name: Option<String>,
    #[serde(default)]
    adapter_type: Option<AdapterType>,
    #[serde(default)]
    base_model_id: Option<String>,
    #[serde(default)]
    base_model_revision: Option<String>,
    #[serde(default)]
    base_weight_hash: Option<String>,
    #[serde(default)]
    tokenizer_hash: Option<String>,
    #[serde(default)]
    chat_template_hash: Option<String>,
    #[serde(default)]
    dtype: Option<AdapterDType>,
    #[serde(default)]
    requires_tokenizer_changes: Option<bool>,
    #[serde(default)]
    supports_mtp: Option<MtpSupport>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum StringList {
    One(String),
    Many(Vec<String>),
}

impl StringList {
    fn values(&self) -> Vec<String> {
        match self {
            Self::One(value) => vec![value.clone()],
            Self::Many(values) => values.clone(),
        }
    }
}

fn validate_safetensors(path: &Path) -> Result<SafetensorsValidation> {
    let bytes = fs::read(path)?;
    if bytes.len() < 8 {
        return Err(Error::InvalidSafetensors(
            "safetensors file is too short for header length".to_owned(),
        ));
    }
    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&bytes[..8]);
    let header_len = u64::from_le_bytes(len_bytes);
    let header_len_usize = usize::try_from(header_len)
        .map_err(|_| Error::InvalidSafetensors("safetensors header is too large".to_owned()))?;
    let header_start = 8usize;
    let header_end = header_start.checked_add(header_len_usize).ok_or_else(|| {
        Error::InvalidSafetensors("safetensors header length overflow".to_owned())
    })?;
    if header_end > bytes.len() {
        return Err(Error::InvalidSafetensors(
            "safetensors header extends past end of file".to_owned(),
        ));
    }
    let data_len = bytes.len() - header_end;
    let header: serde_json::Value = serde_json::from_slice(&bytes[header_start..header_end])
        .map_err(|error| Error::InvalidSafetensors(format!("invalid safetensors JSON: {error}")))?;
    let Some(object) = header.as_object() else {
        return Err(Error::InvalidSafetensors(
            "safetensors header must be a JSON object".to_owned(),
        ));
    };

    let mut tensor_count = 0usize;
    let mut lora_a_tensors = 0usize;
    let mut lora_b_tensors = 0usize;
    for (name, metadata) in object {
        if name == "__metadata__" {
            continue;
        }
        tensor_count += 1;
        validate_tensor_metadata(name, metadata, data_len)?;
        if name.contains("lora_A") {
            lora_a_tensors += 1;
        }
        if name.contains("lora_B") {
            lora_b_tensors += 1;
        }
    }

    if tensor_count == 0 {
        return Err(Error::InvalidSafetensors(
            "safetensors file contains no tensors".to_owned(),
        ));
    }
    if lora_a_tensors == 0 || lora_b_tensors == 0 {
        return Err(Error::InvalidSafetensors(
            "safetensors file must include lora_A and lora_B tensors".to_owned(),
        ));
    }

    Ok(SafetensorsValidation {
        tensor_count,
        lora_a_tensors,
        lora_b_tensors,
        resident_bytes: bytes.len() as u64,
        shape_validation_result: "header_only_lora_tensors_present".to_owned(),
    })
}

fn validate_tensor_metadata(
    name: &str,
    metadata: &serde_json::Value,
    data_len: usize,
) -> Result<()> {
    let Some(object) = metadata.as_object() else {
        return Err(Error::InvalidSafetensors(format!(
            "tensor {name} metadata must be an object"
        )));
    };
    let dtype = object
        .get("dtype")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| Error::InvalidSafetensors(format!("tensor {name} missing dtype")))?;
    if dtype.trim().is_empty() {
        return Err(Error::InvalidSafetensors(format!(
            "tensor {name} dtype is empty"
        )));
    }
    let shape = object
        .get("shape")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Error::InvalidSafetensors(format!("tensor {name} missing shape")))?;
    if shape.is_empty() {
        return Err(Error::InvalidSafetensors(format!(
            "tensor {name} shape must not be empty"
        )));
    }
    if shape.iter().any(|dim| dim.as_u64().is_none()) {
        return Err(Error::InvalidSafetensors(format!(
            "tensor {name} shape contains a non-integer dimension"
        )));
    }
    let offsets = object
        .get("data_offsets")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Error::InvalidSafetensors(format!("tensor {name} missing data_offsets")))?;
    if offsets.len() != 2 {
        return Err(Error::InvalidSafetensors(format!(
            "tensor {name} data_offsets must have two values"
        )));
    }
    let start = offsets[0].as_u64().ok_or_else(|| {
        Error::InvalidSafetensors(format!("tensor {name} data_offsets[0] is not an integer"))
    })?;
    let end = offsets[1].as_u64().ok_or_else(|| {
        Error::InvalidSafetensors(format!("tensor {name} data_offsets[1] is not an integer"))
    })?;
    let data_len = data_len as u64;
    if start >= end || end > data_len {
        return Err(Error::InvalidSafetensors(format!(
            "tensor {name} data_offsets [{start}, {end}] exceed data length {data_len}"
        )));
    }
    Ok(())
}

fn validate_targets_present(
    target_modules: &[String],
    validation: &SafetensorsValidation,
) -> Result<()> {
    if target_modules.is_empty() {
        return Err(Error::InvalidManifest(
            "target_modules must not be empty".to_owned(),
        ));
    }
    if validation.lora_a_tensors < target_modules.len()
        || validation.lora_b_tensors < target_modules.len()
    {
        return Err(Error::InvalidSafetensors(format!(
            "safetensors has {} lora_A and {} lora_B tensors for {} target modules",
            validation.lora_a_tensors,
            validation.lora_b_tensors,
            target_modules.len()
        )));
    }
    Ok(())
}

fn summary_for_entry(entry: &RegistryEntry, active_adapter_id: Option<&str>) -> AdapterSummary {
    let manifest = &entry.adapter.manifest;
    AdapterSummary {
        adapter_id: manifest.adapter_id.clone(),
        display_name: manifest.display_name.clone(),
        adapter_type: manifest.adapter_type,
        source_path: entry.adapter.source_path.clone(),
        loaded: entry.loaded,
        pinned: entry.pinned,
        active: active_adapter_id == Some(manifest.adapter_id.as_str()),
        resident_bytes: if entry.loaded {
            entry.adapter.validation.resident_bytes
        } else {
            0
        },
        load_latency_us: entry.adapter.load_latency_us,
        target_modules: manifest.target_modules.clone(),
        supports_mtp: manifest.supports_mtp,
    }
}

fn require_non_empty(value: &str, field: &str) -> Result<()> {
    if value.trim().is_empty() {
        Err(Error::InvalidManifest(format!("{field} must not be empty")))
    } else {
        Ok(())
    }
}

fn compare_manifest_field(field: &str, expected: &str, actual: &str) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(Error::InvalidManifest(format!(
            "{field} mismatch: expected {expected}, adapter declares {actual}"
        )))
    }
}

fn fallback_adapter_id(source_path: &Path) -> String {
    source_path
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("unnamed-adapter")
        .to_owned()
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path)?;
    Ok(sha256_bytes(&bytes))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn reports_m10_status() {
        assert_eq!(CRATE_NAME, "gemma4d-adapters");
        assert_eq!(bootstrap_status(), "m10-dynamic-adapters");
    }

    #[test]
    fn manifest_parser_rejects_unsupported_adapter_features() {
        let raw = r#"{
            "adapter_id":"bad",
            "adapter_type":"alora",
            "base_model_id":"model",
            "base_weight_hash":"weights",
            "tokenizer_hash":"tokenizer",
            "chat_template_hash":"template",
            "rank":16,
            "alpha":32.0,
            "target_modules":["q_proj"],
            "adapter_weight_hash":"adapter"
        }"#;
        let err = AdapterManifest::from_json_str(raw).expect_err("aLoRA rejected");
        assert!(err.to_string().contains("not supported"));
    }

    #[test]
    fn peft_adapter_imports_and_routes_one_active_adapter() {
        let fixture = AdapterFixture::new("peft-imports");
        let adapter_dir = fixture.write_adapter("rust-coding-r16-v1", None);
        let trusted = TrustedPathPolicy::new(&fixture.trusted_root).expect("trusted root");
        let mut registry = AdapterRegistry::open(&fixture.registry_dir).expect("registry");

        let imported = registry
            .import_peft(&adapter_dir, &trusted, &fixture.compatibility())
            .expect("adapter imports");

        assert_eq!(imported.manifest.adapter_id, "rust-coding-r16-v1");
        assert_eq!(imported.validation.tensor_count, 4);
        assert_eq!(registry.summaries().len(), 1);
        assert!(registry.summaries()[0].loaded);

        let route = registry
            .activate_request(Some("rust-coding-r16-v1"))
            .expect("active route");
        assert_eq!(
            route.active_adapter_id.as_deref(),
            Some("rust-coding-r16-v1")
        );
        assert!(!route.mtp_enabled);
        assert!(route.mtp_disable_reason.is_some());

        let base_route = registry.activate_request(None).expect("base route");
        assert!(base_route.mtp_enabled);
        assert_eq!(fixture_generate_token(42, &base_route), 42);
        assert_ne!(fixture_generate_token(42, &route), 42);
    }

    #[test]
    fn wrong_base_tokenizer_and_template_are_rejected() {
        for (case, field, value) in [
            ("wrong-base", "base_model_id", "other-model"),
            ("wrong-tokenizer", "tokenizer_hash", "other-tokenizer"),
            ("wrong-template", "chat_template_hash", "other-template"),
        ] {
            let fixture = AdapterFixture::new(case);
            let adapter_dir =
                fixture.write_adapter("mismatch", Some(format!(r#""{field}":"{value}""#).as_str()));
            let trusted = TrustedPathPolicy::new(&fixture.trusted_root).expect("trusted root");
            let mut registry = AdapterRegistry::open(&fixture.registry_dir).expect("registry");
            let err = registry
                .import_peft(&adapter_dir, &trusted, &fixture.compatibility())
                .expect_err("compat mismatch");
            assert!(err.to_string().contains("mismatch"));
        }
    }

    #[test]
    fn modules_to_save_and_tokenizer_changes_are_rejected() {
        let fixture = AdapterFixture::new("unsupported-features");
        let modules_dir =
            fixture.write_adapter_config("modules", r#""modules_to_save":["lm_head"]"#);
        let tokenizer_dir =
            fixture.write_adapter("tokenizer", Some(r#""requires_tokenizer_changes":true"#));
        let trusted = TrustedPathPolicy::new(&fixture.trusted_root).expect("trusted root");
        let mut registry = AdapterRegistry::open(&fixture.registry_dir).expect("registry");

        let modules_err = registry
            .import_peft(&modules_dir, &trusted, &fixture.compatibility())
            .expect_err("modules_to_save rejected");
        assert!(modules_err.to_string().contains("modules_to_save"));

        let tokenizer_err = registry
            .import_peft(&tokenizer_dir, &trusted, &fixture.compatibility())
            .expect_err("tokenizer changes rejected");
        assert!(tokenizer_err.to_string().contains("tokenizer changes"));
    }

    #[test]
    fn untrusted_path_is_rejected() {
        let fixture = AdapterFixture::new("untrusted");
        let outside = fixture.outside_root.join("outside-adapter");
        fs::create_dir_all(&outside).expect("outside dir");
        write_peft_config(&outside, "outside-adapter", "", false);
        write_safetensors(&outside.join("adapter_model.safetensors"));
        let trusted = TrustedPathPolicy::new(&fixture.trusted_root).expect("trusted root");
        let mut registry = AdapterRegistry::open(&fixture.registry_dir).expect("registry");

        let err = registry
            .import_peft(&outside, &trusted, &fixture.compatibility())
            .expect_err("untrusted path rejected");
        assert!(matches!(err, Error::UntrustedPath { .. }));
    }

    #[test]
    fn load_unload_and_pin_update_registry_summary() {
        let fixture = AdapterFixture::new("load-unload-pin");
        let adapter_dir = fixture.write_adapter("pin-me", None);
        let trusted = TrustedPathPolicy::new(&fixture.trusted_root).expect("trusted root");
        let mut registry = AdapterRegistry::open(&fixture.registry_dir).expect("registry");
        registry
            .import_peft(&adapter_dir, &trusted, &fixture.compatibility())
            .expect("adapter imports");

        let pinned = registry.pin("pin-me").expect("pin");
        assert!(pinned.pinned);
        let unloaded = registry.unload("pin-me").expect("unload");
        assert!(!unloaded.loaded);
        assert_eq!(unloaded.resident_bytes, 0);
        let loaded = registry.load("pin-me").expect("load");
        assert!(loaded.loaded);
        assert!(loaded.resident_bytes > 0);

        let reopened = AdapterRegistry::open(&fixture.registry_dir).expect("reopen");
        let summary = reopened.summaries().pop().expect("summary");
        assert!(summary.loaded);
        assert!(summary.pinned);
    }

    #[test]
    fn invalid_safetensors_header_is_rejected() {
        let fixture = AdapterFixture::new("invalid-safetensors");
        let adapter_dir = fixture.write_adapter("broken", None);
        fs::write(adapter_dir.join("adapter_model.safetensors"), b"bad").expect("bad weights");
        let trusted = TrustedPathPolicy::new(&fixture.trusted_root).expect("trusted root");
        let mut registry = AdapterRegistry::open(&fixture.registry_dir).expect("registry");

        let err = registry
            .import_peft(&adapter_dir, &trusted, &fixture.compatibility())
            .expect_err("bad safetensors rejected");
        assert!(matches!(err, Error::InvalidSafetensors(_)));
    }

    struct AdapterFixture {
        trusted_root: PathBuf,
        outside_root: PathBuf,
        registry_dir: PathBuf,
    }

    impl AdapterFixture {
        fn new(name: &str) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos();
            let root = std::env::temp_dir().join(format!("gemma4d-adapters-{name}-{nonce}"));
            let trusted_root = root.join("trusted");
            let outside_root = root.join("outside");
            let registry_dir = root.join("registry");
            fs::create_dir_all(&trusted_root).expect("trusted root");
            fs::create_dir_all(&outside_root).expect("outside root");
            fs::create_dir_all(&registry_dir).expect("registry dir");
            Self {
                trusted_root,
                outside_root,
                registry_dir,
            }
        }

        fn compatibility(&self) -> AdapterCompatibility {
            AdapterCompatibility {
                base_model_id: "mlx-community/gemma-4-12B-it-4bit".to_owned(),
                base_weight_hash: "base-weight-hash".to_owned(),
                tokenizer_hash: "tokenizer-hash".to_owned(),
                chat_template_hash: "chat-template-hash".to_owned(),
            }
        }

        fn write_adapter(&self, adapter_id: &str, metadata_extra: Option<&str>) -> PathBuf {
            self.write_adapter_config(adapter_id, metadata_extra.unwrap_or(""))
        }

        fn write_adapter_config(&self, adapter_id: &str, extra: &str) -> PathBuf {
            let adapter_dir = self.trusted_root.join(adapter_id);
            fs::create_dir_all(&adapter_dir).expect("adapter dir");
            write_peft_config(
                &adapter_dir,
                adapter_id,
                extra,
                !extra.contains("modules_to_save"),
            );
            write_safetensors(&adapter_dir.join("adapter_model.safetensors"));
            adapter_dir
        }
    }

    fn write_peft_config(adapter_dir: &Path, adapter_id: &str, extra: &str, include_gemma4d: bool) {
        let metadata = if include_gemma4d {
            let mut fields = vec![
                format!(r#""adapter_id": "{adapter_id}""#),
                r#""display_name": "Rust coding fixture""#.to_owned(),
                r#""adapter_type": "lora""#.to_owned(),
                r#""dtype": "bf16""#.to_owned(),
                r#""supports_mtp": "unknown""#.to_owned(),
            ];
            if !extra.contains(r#""base_model_id""#) {
                fields.push(r#""base_model_id": "mlx-community/gemma-4-12B-it-4bit""#.to_owned());
            }
            if !extra.contains(r#""base_weight_hash""#) {
                fields.push(r#""base_weight_hash": "base-weight-hash""#.to_owned());
            }
            if !extra.contains(r#""tokenizer_hash""#) {
                fields.push(r#""tokenizer_hash": "tokenizer-hash""#.to_owned());
            }
            if !extra.contains(r#""chat_template_hash""#) {
                fields.push(r#""chat_template_hash": "chat-template-hash""#.to_owned());
            }
            if !extra.trim().is_empty() {
                fields.push(extra.to_owned());
            }
            format!(
                r#",
  "gemma4d": {{
    {}
  }}"#,
                fields.join(",\n    ")
            )
        } else {
            format!(",\n  {extra}")
        };
        let raw = format!(
            r#"{{
  "peft_type": "LORA",
  "base_model_name_or_path": "mlx-community/gemma-4-12B-it-4bit",
  "r": 16,
  "lora_alpha": 32.0,
  "lora_dropout": 0.05,
  "target_modules": ["q_proj", "v_proj"]{metadata}
}}"#
        );
        fs::write(adapter_dir.join("adapter_config.json"), raw).expect("adapter config");
    }

    fn write_safetensors(path: &Path) {
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
        fs::write(path, bytes).expect("safetensors");
    }
}
