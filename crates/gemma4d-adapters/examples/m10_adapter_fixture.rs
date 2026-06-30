use std::{
    env, fs,
    num::NonZeroU64,
    path::{Path, PathBuf},
};

use gemma4d_adapters::{
    AdapterCompatibility, AdapterRegistry, Error as AdapterError, TrustedPathPolicy,
    fixture_generate_token,
};
use gemma4d_kv::{
    Error as KvError, KvBlockKey, KvNamespace, RamPrefixBlock, RamPrefixCache,
    estimated_bf16_kv_bytes, fresh_prefill_fixture,
};
use serde::Serialize;

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    milestone: &'static str,
    status: &'static str,
    commands: Vec<&'static str>,
    fixture_root: String,
    registry_dir: String,
    import: ImportReport,
    rejection: RejectionReport,
    routing: RoutingReport,
    registry: RegistryReport,
    kv_namespace: KvNamespaceReport,
}

#[derive(Debug, Serialize)]
struct ImportReport {
    adapter_id: String,
    tensor_count: usize,
    resident_bytes: u64,
    load_latency_us: u128,
    shape_validation_result: String,
}

#[derive(Debug, Serialize)]
struct RejectionReport {
    wrong_base_rejected: bool,
    wrong_tokenizer_rejected: bool,
    wrong_template_rejected: bool,
    modules_to_save_rejected: bool,
    untrusted_path_rejected: bool,
}

#[derive(Debug, Serialize)]
struct RoutingReport {
    one_active_adapter_per_request: bool,
    base_output_unchanged_when_disabled: bool,
    adapter_changes_fixture_output: bool,
    mtp_disabled_with_adapter: bool,
    mtp_disable_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct RegistryReport {
    loaded_after_import: bool,
    pinned_after_pin: bool,
    loaded_after_unload: bool,
    loaded_after_load: bool,
    total_resident_bytes_after_load: u64,
}

#[derive(Debug, Serialize)]
struct KvNamespaceReport {
    namespace_hashes_unique_by_adapter: bool,
    block_ids_unique_by_adapter: bool,
    wrong_adapter_restore_rejected: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_path = parse_out_path()?;
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let out_dir = out_path.parent().unwrap_or_else(|| Path::new("."));
    let fixture_root = out_dir.join("fixtures");
    let trusted_root = fixture_root.join("trusted");
    let registry_dir = out_dir.join("registry");
    reset_dir(&fixture_root)?;
    reset_dir(&registry_dir)?;
    fs::create_dir_all(&trusted_root)?;

    let compatibility = compatibility();
    let valid_dir = write_adapter(
        &trusted_root,
        "rust-coding-r16-v1",
        MetadataOverride::default(),
    )?;
    let policy = TrustedPathPolicy::new(&trusted_root)?;
    let mut registry = AdapterRegistry::open(&registry_dir)?;
    let imported = registry.import_peft(&valid_dir, &policy, &compatibility)?;

    let active_route = registry.activate_request(Some("rust-coding-r16-v1"))?;
    let base_route = registry.activate_request(None)?;
    let base_output_unchanged_when_disabled = fixture_generate_token(42, &base_route) == 42;
    let adapter_changes_fixture_output = fixture_generate_token(42, &active_route) != 42;
    let mtp_disabled_with_adapter = !active_route.mtp_enabled;

    let loaded_after_import = registry
        .summaries()
        .iter()
        .any(|summary| summary.adapter_id == "rust-coding-r16-v1" && summary.loaded);
    let pinned_after_pin = registry.pin("rust-coding-r16-v1")?.pinned;
    let loaded_after_unload = registry.unload("rust-coding-r16-v1")?.loaded;
    let loaded_after_load = registry.load("rust-coding-r16-v1")?.loaded;
    let total_resident_bytes_after_load = registry.total_resident_bytes();

    let rejection = RejectionReport {
        wrong_base_rejected: rejection_case(
            &trusted_root,
            &policy,
            &registry_dir,
            &compatibility,
            "wrong-base",
            MetadataOverride {
                base_model_id: Some("other-model"),
                ..MetadataOverride::default()
            },
        )?,
        wrong_tokenizer_rejected: rejection_case(
            &trusted_root,
            &policy,
            &registry_dir,
            &compatibility,
            "wrong-tokenizer",
            MetadataOverride {
                tokenizer_hash: Some("other-tokenizer"),
                ..MetadataOverride::default()
            },
        )?,
        wrong_template_rejected: rejection_case(
            &trusted_root,
            &policy,
            &registry_dir,
            &compatibility,
            "wrong-template",
            MetadataOverride {
                chat_template_hash: Some("other-template"),
                ..MetadataOverride::default()
            },
        )?,
        modules_to_save_rejected: modules_to_save_rejection(
            &trusted_root,
            &policy,
            &registry_dir,
            &compatibility,
        )?,
        untrusted_path_rejected: untrusted_rejection(
            &fixture_root,
            &policy,
            &registry_dir,
            &compatibility,
        )?,
    };
    let kv_namespace = kv_namespace_report(
        "rust-coding-r16-v1",
        &imported.manifest.adapter_weight_hash,
        "sql-r16-v1",
        "other-adapter-weight-hash",
    )?;

    let passed = loaded_after_import
        && pinned_after_pin
        && !loaded_after_unload
        && loaded_after_load
        && base_output_unchanged_when_disabled
        && adapter_changes_fixture_output
        && mtp_disabled_with_adapter
        && rejection.wrong_base_rejected
        && rejection.wrong_tokenizer_rejected
        && rejection.wrong_template_rejected
        && rejection.modules_to_save_rejected
        && rejection.untrusted_path_rejected
        && kv_namespace.namespace_hashes_unique_by_adapter
        && kv_namespace.block_ids_unique_by_adapter
        && kv_namespace.wrong_adapter_restore_rejected;

    let report = Report {
        schema_version: 1,
        milestone: "M10",
        status: if passed { "passed" } else { "failed" },
        commands: vec![
            "cargo test -p gemma4d-adapters --all-targets",
            "cargo test -p gemma4d-kv --all-targets",
            "cargo run -p gemma4d-adapters --example m10_adapter_fixture -- --out benchmarks/out/M10/adapter-fixture.json",
        ],
        fixture_root: fixture_root.display().to_string(),
        registry_dir: registry_dir.display().to_string(),
        import: ImportReport {
            adapter_id: imported.manifest.adapter_id,
            tensor_count: imported.validation.tensor_count,
            resident_bytes: imported.validation.resident_bytes,
            load_latency_us: imported.load_latency_us,
            shape_validation_result: imported.validation.shape_validation_result,
        },
        rejection,
        routing: RoutingReport {
            one_active_adapter_per_request: active_route.active_adapter_id.is_some()
                && base_route.active_adapter_id.is_none(),
            base_output_unchanged_when_disabled,
            adapter_changes_fixture_output,
            mtp_disabled_with_adapter,
            mtp_disable_reason: active_route.mtp_disable_reason,
        },
        registry: RegistryReport {
            loaded_after_import,
            pinned_after_pin,
            loaded_after_unload,
            loaded_after_load,
            total_resident_bytes_after_load,
        },
        kv_namespace,
    };

    fs::write(&out_path, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "M10 adapter fixture: adapter={} tensors={} {}",
        report.import.adapter_id, report.import.tensor_count, report.status
    );
    println!("evidence: {}", out_path.display());
    if passed {
        Ok(())
    } else {
        Err("M10 adapter fixture failed".into())
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct MetadataOverride {
    base_model_id: Option<&'static str>,
    base_weight_hash: Option<&'static str>,
    tokenizer_hash: Option<&'static str>,
    chat_template_hash: Option<&'static str>,
}

fn compatibility() -> AdapterCompatibility {
    AdapterCompatibility {
        base_model_id: "mlx-community/gemma-4-12B-it-4bit".to_owned(),
        base_weight_hash: "base-weight-hash".to_owned(),
        tokenizer_hash: "tokenizer-hash".to_owned(),
        chat_template_hash: "chat-template-hash".to_owned(),
    }
}

fn rejection_case(
    trusted_root: &Path,
    policy: &TrustedPathPolicy,
    registry_dir: &Path,
    compatibility: &AdapterCompatibility,
    adapter_id: &str,
    override_metadata: MetadataOverride,
) -> Result<bool, Box<dyn std::error::Error>> {
    let adapter_dir = write_adapter(trusted_root, adapter_id, override_metadata)?;
    let mut registry = AdapterRegistry::open(registry_dir)?;
    Ok(matches!(
        registry.import_peft(&adapter_dir, policy, compatibility),
        Err(AdapterError::InvalidManifest(_))
    ))
}

fn modules_to_save_rejection(
    trusted_root: &Path,
    policy: &TrustedPathPolicy,
    registry_dir: &Path,
    compatibility: &AdapterCompatibility,
) -> Result<bool, Box<dyn std::error::Error>> {
    let adapter_dir = trusted_root.join("modules-to-save");
    fs::create_dir_all(&adapter_dir)?;
    write_peft_config(
        &adapter_dir,
        "modules-to-save",
        MetadataOverride::default(),
        true,
    )?;
    write_safetensors(&adapter_dir.join("adapter_model.safetensors"))?;
    let mut registry = AdapterRegistry::open(registry_dir)?;
    Ok(matches!(
        registry.import_peft(&adapter_dir, policy, compatibility),
        Err(AdapterError::UnsupportedAdapter(_))
    ))
}

fn untrusted_rejection(
    fixture_root: &Path,
    policy: &TrustedPathPolicy,
    registry_dir: &Path,
    compatibility: &AdapterCompatibility,
) -> Result<bool, Box<dyn std::error::Error>> {
    let outside_root = fixture_root.join("outside");
    fs::create_dir_all(&outside_root)?;
    let adapter_dir = write_adapter(
        &outside_root,
        "outside-adapter",
        MetadataOverride::default(),
    )?;
    let mut registry = AdapterRegistry::open(registry_dir)?;
    Ok(matches!(
        registry.import_peft(&adapter_dir, policy, compatibility),
        Err(AdapterError::UntrustedPath { .. })
    ))
}

fn kv_namespace_report(
    adapter_a: &str,
    weight_hash_a: &str,
    adapter_b: &str,
    weight_hash_b: &str,
) -> Result<KvNamespaceReport, Box<dyn std::error::Error>> {
    let block_size = NonZeroU64::new(1024).expect("non-zero");
    let mut namespace_a = KvNamespace::fixture(1024);
    namespace_a.adapter_id = Some(adapter_a.to_owned());
    namespace_a.adapter_weight_hash = Some(weight_hash_a.to_owned());
    let mut namespace_b = KvNamespace::fixture(1024);
    namespace_b.adapter_id = Some(adapter_b.to_owned());
    namespace_b.adapter_weight_hash = Some(weight_hash_b.to_owned());
    let namespace_hashes_unique_by_adapter =
        namespace_a.namespace_hash()? != namespace_b.namespace_hash()?;
    let key_a = KvBlockKey::new(&namespace_a, 0, block_size, 0, 1024)?;
    let key_b = KvBlockKey::new(&namespace_b, 0, block_size, 0, 1024)?;
    let block_ids_unique_by_adapter = key_a.block_id != key_b.block_id;

    let block = RamPrefixBlock::from_observation(
        namespace_a,
        0,
        block_size,
        0,
        fresh_prefill_fixture(1024),
        estimated_bf16_kv_bytes(1024),
    )?;
    let key = block.key.clone();
    let mut cache = RamPrefixCache::new(NonZeroU64::new(block.byte_len * 2).expect("non-zero"));
    cache.insert(block)?;
    let wrong_adapter_restore_rejected = matches!(
        cache.restore(&key, &namespace_b),
        Err(KvError::NamespaceMismatch { .. })
    );

    Ok(KvNamespaceReport {
        namespace_hashes_unique_by_adapter,
        block_ids_unique_by_adapter,
        wrong_adapter_restore_rejected,
    })
}

fn write_adapter(
    trusted_root: &Path,
    adapter_id: &str,
    override_metadata: MetadataOverride,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let adapter_dir = trusted_root.join(adapter_id);
    fs::create_dir_all(&adapter_dir)?;
    write_peft_config(&adapter_dir, adapter_id, override_metadata, false)?;
    write_safetensors(&adapter_dir.join("adapter_model.safetensors"))?;
    Ok(adapter_dir)
}

fn write_peft_config(
    adapter_dir: &Path,
    adapter_id: &str,
    override_metadata: MetadataOverride,
    modules_to_save: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let modules_to_save = if modules_to_save {
        r#",
  "modules_to_save": ["lm_head"]"#
    } else {
        ""
    };
    let raw = format!(
        r#"{{
  "peft_type": "LORA",
  "base_model_name_or_path": "mlx-community/gemma-4-12B-it-4bit",
  "r": 16,
  "lora_alpha": 32.0,
  "lora_dropout": 0.05,
  "target_modules": ["q_proj", "v_proj"]{modules_to_save},
  "gemma4d": {{
    "adapter_id": "{adapter_id}",
    "base_model_id": "{}",
    "base_weight_hash": "{}",
    "tokenizer_hash": "{}",
    "chat_template_hash": "{}",
    "adapter_type": "lora",
    "dtype": "bf16",
    "supports_mtp": "unknown"
  }}
}}"#,
        override_metadata
            .base_model_id
            .unwrap_or("mlx-community/gemma-4-12B-it-4bit"),
        override_metadata
            .base_weight_hash
            .unwrap_or("base-weight-hash"),
        override_metadata.tokenizer_hash.unwrap_or("tokenizer-hash"),
        override_metadata
            .chat_template_hash
            .unwrap_or("chat-template-hash"),
    );
    fs::write(adapter_dir.join("adapter_config.json"), raw)?;
    Ok(())
}

fn write_safetensors(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
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
    let header = serde_json::to_vec(&header)?;
    let mut bytes = Vec::with_capacity(8 + header.len() + 2048);
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&header);
    bytes.extend(vec![0u8; 2048]);
    fs::write(path, bytes)?;
    Ok(())
}

fn reset_dir(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    match fs::remove_dir_all(path) {
        Ok(()) => {}
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {}
        Err(source) => return Err(Box::new(source)),
    }
    fs::create_dir_all(path)?;
    Ok(())
}

fn parse_out_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let mut out = None;
    while let Some(arg) = args.next() {
        if arg == "--out" {
            out = args.next().map(PathBuf::from);
        }
    }
    out.ok_or_else(|| "usage: m10_adapter_fixture --out <path>".into())
}
