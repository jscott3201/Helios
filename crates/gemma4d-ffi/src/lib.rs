#![doc = "Safe Rust wrappers for the narrow Gemma4D native C ABI."]

use std::{
    ffi::{CStr, CString, NulError},
    fmt,
    marker::PhantomData,
    num::NonZeroU32,
    os::raw::c_char,
    ptr::{self, NonNull},
};

mod raw {
    use super::c_char;

    pub type Gemma4Status = i32;

    pub const GEMMA4_OK: Gemma4Status = 0;
    pub const GEMMA4_ERR_INVALID_ARGUMENT: Gemma4Status = 1;
    pub const GEMMA4_ERR_UNSUPPORTED_CONFIG: Gemma4Status = 2;
    pub const GEMMA4_ERR_MODEL_LOAD: Gemma4Status = 3;
    pub const GEMMA4_ERR_RUNTIME: Gemma4Status = 4;
    pub const GEMMA4_ERR_MEMORY_GUARD: Gemma4Status = 5;
    pub const GEMMA4_ERR_CACHE: Gemma4Status = 6;
    pub const GEMMA4_ERR_ADAPTER: Gemma4Status = 7;

    #[repr(C)]
    pub struct Gemma4Target {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct Gemma4Drafter {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct Gemma4KvCache {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct Gemma4KvSnapshot {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct Gemma4VersionInfo {
        pub abi_version: u32,
        pub backend_name: *const c_char,
        pub backend_version: *const c_char,
    }

    #[repr(C)]
    pub struct Gemma4LoadConfig {
        pub model_path: *const c_char,
        pub model_id: *const c_char,
        pub model_revision: *const c_char,
        pub expected_architecture: *const c_char,
        pub max_context_tokens: u32,
        pub allow_unsupported_config: bool,
    }

    #[repr(C)]
    pub struct Gemma4KvPolicy {
        pub active_mode: i32,
        pub ram_prefix_mode: i32,
        pub ssd_prefix_mode: i32,
        pub block_size_tokens: u32,
        pub quantized_kv_start: u32,
        pub compress_global_layers: bool,
        pub compress_sliding_layers: bool,
        pub keep_mtp_shared_layers_bf16: bool,
        pub allow_active_compressed_decode: bool,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct Gemma4StepResult {
        pub greedy_token: i32,
        pub greedy_logit: f32,
        pub peak_memory_gb: f32,
        pub peak_rss_mb: f32,
        pub sequence_len: u64,
        pub active_kv_bytes: u64,
        pub accepted_draft_count: u32,
        pub committed_count: u32,
        pub committed_tokens: [i32; 4],
        pub native_last_hidden: *mut std::ffi::c_void,
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct Gemma4KvSnapshotInfo {
        pub sequence_len: u64,
        pub active_kv_bytes: u64,
        pub token_count: u64,
        pub has_last_step: bool,
    }

    // SAFETY: These declarations mirror `native/gemma4_mlx/include/gemma4_mlx.h`.
    unsafe extern "C" {
        pub fn gemma4_runtime_version(out: *mut Gemma4VersionInfo) -> Gemma4Status;
        pub fn gemma4_get_last_error(buffer: *mut c_char, buffer_len: usize) -> Gemma4Status;
        pub fn gemma4_load_target(
            config: *const Gemma4LoadConfig,
            out: *mut *mut Gemma4Target,
        ) -> Gemma4Status;
        pub fn gemma4_free_target(target: *mut Gemma4Target) -> Gemma4Status;
        pub fn gemma4_kv_create(
            policy: *const Gemma4KvPolicy,
            out: *mut *mut Gemma4KvCache,
        ) -> Gemma4Status;
        pub fn gemma4_kv_free(cache: *mut Gemma4KvCache) -> Gemma4Status;
        pub fn gemma4_kv_reset(cache: *mut Gemma4KvCache) -> Gemma4Status;
        pub fn gemma4_kv_last_step(
            cache: *const Gemma4KvCache,
            out: *mut Gemma4StepResult,
        ) -> Gemma4Status;
        pub fn gemma4_kv_snapshot_export(
            cache: *const Gemma4KvCache,
            out: *mut *mut Gemma4KvSnapshot,
        ) -> Gemma4Status;
        pub fn gemma4_kv_snapshot_import(
            cache: *mut Gemma4KvCache,
            snapshot: *const Gemma4KvSnapshot,
        ) -> Gemma4Status;
        pub fn gemma4_kv_snapshot_info(
            snapshot: *const Gemma4KvSnapshot,
            out: *mut Gemma4KvSnapshotInfo,
        ) -> Gemma4Status;
        pub fn gemma4_kv_snapshot_free(snapshot: *mut Gemma4KvSnapshot) -> Gemma4Status;
        pub fn gemma4_prefill(
            target: *mut Gemma4Target,
            cache: *mut Gemma4KvCache,
            tokens: *const i32,
            token_count: usize,
            out: *mut Gemma4StepResult,
        ) -> Gemma4Status;
        pub fn gemma4_decode_one(
            target: *mut Gemma4Target,
            cache: *mut Gemma4KvCache,
            token: i32,
            out: *mut Gemma4StepResult,
        ) -> Gemma4Status;
        pub fn gemma4_load_drafter(
            config: *const Gemma4LoadConfig,
            target: *mut Gemma4Target,
            out: *mut *mut Gemma4Drafter,
        ) -> Gemma4Status;
        pub fn gemma4_free_drafter(drafter: *mut Gemma4Drafter) -> Gemma4Status;
        pub fn gemma4_mtp_draft_block(
            drafter: *mut Gemma4Drafter,
            cache: *mut Gemma4KvCache,
            block_size: u32,
            out_tokens: *mut i32,
            inout_count: *mut usize,
        ) -> Gemma4Status;
        pub fn gemma4_verify_tokens(
            target: *mut Gemma4Target,
            cache: *mut Gemma4KvCache,
            draft_tokens: *const i32,
            draft_count: usize,
            out: *mut Gemma4StepResult,
        ) -> Gemma4Status;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Ok,
    InvalidArgument,
    UnsupportedConfig,
    ModelLoad,
    Runtime,
    MemoryGuard,
    Cache,
    Adapter,
    Unknown(i32),
}

impl Status {
    fn from_raw(raw: raw::Gemma4Status) -> Self {
        match raw {
            raw::GEMMA4_OK => Self::Ok,
            raw::GEMMA4_ERR_INVALID_ARGUMENT => Self::InvalidArgument,
            raw::GEMMA4_ERR_UNSUPPORTED_CONFIG => Self::UnsupportedConfig,
            raw::GEMMA4_ERR_MODEL_LOAD => Self::ModelLoad,
            raw::GEMMA4_ERR_RUNTIME => Self::Runtime,
            raw::GEMMA4_ERR_MEMORY_GUARD => Self::MemoryGuard,
            raw::GEMMA4_ERR_CACHE => Self::Cache,
            raw::GEMMA4_ERR_ADAPTER => Self::Adapter,
            other => Self::Unknown(other),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    status: Status,
    message: String,
}

impl Error {
    pub fn status(&self) -> Status {
        self.status
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.status, self.message)
    }
}

impl std::error::Error for Error {}

impl From<NulError> for Error {
    fn from(error: NulError) -> Self {
        Self {
            status: Status::InvalidArgument,
            message: format!("string contains interior NUL byte: {error}"),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionInfo {
    pub abi_version: u32,
    pub backend_name: String,
    pub backend_version: String,
}

pub fn runtime_version() -> Result<VersionInfo> {
    let mut raw_version = raw::Gemma4VersionInfo {
        abi_version: 0,
        backend_name: ptr::null(),
        backend_version: ptr::null(),
    };

    // SAFETY: `raw_version` is a valid, writable out pointer for the duration of the call.
    check(unsafe { raw::gemma4_runtime_version(&mut raw_version) })?;

    Ok(VersionInfo {
        abi_version: raw_version.abi_version,
        backend_name: cstr_to_string(raw_version.backend_name),
        backend_version: cstr_to_string(raw_version.backend_version),
    })
}

#[derive(Debug, Clone)]
pub struct LoadConfig {
    pub model_path: String,
    pub model_id: Option<String>,
    pub model_revision: Option<String>,
    pub expected_architecture: Option<String>,
    pub max_context_tokens: NonZeroU32,
    pub allow_unsupported_config: bool,
}

impl LoadConfig {
    pub fn smoke(model_path: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            model_id: Some("gemma4d-smoke".to_owned()),
            model_revision: None,
            expected_architecture: Some("gemma4".to_owned()),
            max_context_tokens: NonZeroU32::new(1).expect("1 is non-zero"),
            allow_unsupported_config: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum KvMode {
    Bf16 = 0,
    MlxAffineQ8 = 1,
    MlxAffineQ4 = 2,
}

#[derive(Debug, Clone)]
pub struct KvPolicy {
    pub active_mode: KvMode,
    pub ram_prefix_mode: KvMode,
    pub ssd_prefix_mode: KvMode,
    pub block_size_tokens: NonZeroU32,
    pub quantized_kv_start: u32,
    pub compress_global_layers: bool,
    pub compress_sliding_layers: bool,
    pub keep_mtp_shared_layers_bf16: bool,
    pub allow_active_compressed_decode: bool,
}

impl Default for KvPolicy {
    fn default() -> Self {
        Self {
            active_mode: KvMode::Bf16,
            ram_prefix_mode: KvMode::Bf16,
            ssd_prefix_mode: KvMode::Bf16,
            block_size_tokens: NonZeroU32::new(1024).expect("1024 is non-zero"),
            quantized_kv_start: 0,
            compress_global_layers: false,
            compress_sliding_layers: false,
            keep_mtp_shared_layers_bf16: true,
            allow_active_compressed_decode: false,
        }
    }
}

#[derive(Debug)]
pub struct Target {
    ptr: NonNull<raw::Gemma4Target>,
}

impl Target {
    pub fn load(config: &LoadConfig) -> Result<Self> {
        let model_path = CString::new(config.model_path.as_str())?;
        let model_id = optional_cstring(config.model_id.as_deref())?;
        let model_revision = optional_cstring(config.model_revision.as_deref())?;
        let expected_architecture = optional_cstring(config.expected_architecture.as_deref())?;

        let raw_config = raw::Gemma4LoadConfig {
            model_path: model_path.as_ptr(),
            model_id: optional_ptr(&model_id),
            model_revision: optional_ptr(&model_revision),
            expected_architecture: optional_ptr(&expected_architecture),
            max_context_tokens: config.max_context_tokens.get(),
            allow_unsupported_config: config.allow_unsupported_config,
        };

        let mut out = ptr::null_mut();
        // SAFETY: `raw_config` points to C strings that live through the call, and `out` is writable.
        check(unsafe { raw::gemma4_load_target(&raw_config, &mut out) })?;
        let ptr = NonNull::new(out).ok_or_else(|| Error {
            status: Status::Runtime,
            message: "gemma4_load_target returned OK with a null target".to_owned(),
        })?;
        Ok(Self { ptr })
    }
}

impl Drop for Target {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` is an owned target handle returned by `gemma4_load_target`.
        let status = unsafe { raw::gemma4_free_target(self.ptr.as_ptr()) };
        debug_assert_eq!(Status::from_raw(status), Status::Ok);
    }
}

#[derive(Debug)]
pub struct Drafter<'target> {
    ptr: NonNull<raw::Gemma4Drafter>,
    _target: PhantomData<&'target Target>,
}

impl<'target> Drafter<'target> {
    pub fn load(config: &LoadConfig, target: &'target Target) -> Result<Self> {
        let model_path = CString::new(config.model_path.as_str())?;
        let model_id = optional_cstring(config.model_id.as_deref())?;
        let model_revision = optional_cstring(config.model_revision.as_deref())?;
        let expected_architecture = optional_cstring(config.expected_architecture.as_deref())?;

        let raw_config = raw::Gemma4LoadConfig {
            model_path: model_path.as_ptr(),
            model_id: optional_ptr(&model_id),
            model_revision: optional_ptr(&model_revision),
            expected_architecture: optional_ptr(&expected_architecture),
            max_context_tokens: config.max_context_tokens.get(),
            allow_unsupported_config: config.allow_unsupported_config,
        };

        let mut out = ptr::null_mut();
        // SAFETY: `raw_config` C strings live through the call; `target` is a valid handle; `out` is writable.
        check(unsafe { raw::gemma4_load_drafter(&raw_config, target.ptr.as_ptr(), &mut out) })?;
        let ptr = NonNull::new(out).ok_or_else(|| Error {
            status: Status::Runtime,
            message: "gemma4_load_drafter returned OK with a null drafter".to_owned(),
        })?;
        Ok(Self {
            ptr,
            _target: PhantomData,
        })
    }
}

impl Drop for Drafter<'_> {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` is an owned drafter handle returned by `gemma4_load_drafter`.
        let status = unsafe { raw::gemma4_free_drafter(self.ptr.as_ptr()) };
        debug_assert_eq!(Status::from_raw(status), Status::Ok);
    }
}

#[derive(Debug)]
pub struct KvCache {
    ptr: NonNull<raw::Gemma4KvCache>,
}

impl KvCache {
    pub fn create(policy: &KvPolicy) -> Result<Self> {
        let raw_policy = raw::Gemma4KvPolicy {
            active_mode: policy.active_mode as i32,
            ram_prefix_mode: policy.ram_prefix_mode as i32,
            ssd_prefix_mode: policy.ssd_prefix_mode as i32,
            block_size_tokens: policy.block_size_tokens.get(),
            quantized_kv_start: policy.quantized_kv_start,
            compress_global_layers: policy.compress_global_layers,
            compress_sliding_layers: policy.compress_sliding_layers,
            keep_mtp_shared_layers_bf16: policy.keep_mtp_shared_layers_bf16,
            allow_active_compressed_decode: policy.allow_active_compressed_decode,
        };

        let mut out = ptr::null_mut();
        // SAFETY: `raw_policy` is a valid policy and `out` is writable for the duration of the call.
        check(unsafe { raw::gemma4_kv_create(&raw_policy, &mut out) })?;
        let ptr = NonNull::new(out).ok_or_else(|| Error {
            status: Status::Runtime,
            message: "gemma4_kv_create returned OK with a null cache".to_owned(),
        })?;
        Ok(Self { ptr })
    }

    pub fn reset(&mut self) -> Result<()> {
        // SAFETY: `self.ptr` is an owned KV cache handle returned by `gemma4_kv_create`.
        check(unsafe { raw::gemma4_kv_reset(self.ptr.as_ptr()) })
    }

    pub fn last_step(&self) -> Result<StepResult> {
        let mut out = raw::Gemma4StepResult::default();
        // SAFETY: `self.ptr` is an owned KV cache handle and `out` is writable.
        check(unsafe { raw::gemma4_kv_last_step(self.ptr.as_ptr(), &mut out) })?;
        Ok(out.into())
    }

    pub fn export_snapshot(&self) -> Result<KvSnapshot> {
        let mut out = ptr::null_mut();
        // SAFETY: `self.ptr` is an owned KV cache handle and `out` is writable.
        check(unsafe { raw::gemma4_kv_snapshot_export(self.ptr.as_ptr(), &mut out) })?;
        let ptr = NonNull::new(out).ok_or_else(|| Error {
            status: Status::Runtime,
            message: "gemma4_kv_snapshot_export returned OK with a null snapshot".to_owned(),
        })?;
        Ok(KvSnapshot { ptr })
    }

    pub fn import_snapshot(&mut self, snapshot: &KvSnapshot) -> Result<()> {
        // SAFETY: handles come from safe wrappers and remain valid for the duration of the call.
        check(unsafe { raw::gemma4_kv_snapshot_import(self.ptr.as_ptr(), snapshot.ptr.as_ptr()) })
    }
}

impl Drop for KvCache {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` is an owned KV cache handle returned by `gemma4_kv_create`.
        let status = unsafe { raw::gemma4_kv_free(self.ptr.as_ptr()) };
        debug_assert_eq!(Status::from_raw(status), Status::Ok);
    }
}

#[derive(Debug)]
pub struct KvSnapshot {
    ptr: NonNull<raw::Gemma4KvSnapshot>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvSnapshotInfo {
    pub sequence_len: u64,
    pub active_kv_bytes: u64,
    pub token_count: u64,
    pub has_last_step: bool,
}

impl KvSnapshot {
    pub fn info(&self) -> Result<KvSnapshotInfo> {
        let mut out = raw::Gemma4KvSnapshotInfo::default();
        // SAFETY: `self.ptr` is an owned snapshot handle and `out` is writable.
        check(unsafe { raw::gemma4_kv_snapshot_info(self.ptr.as_ptr(), &mut out) })?;
        Ok(KvSnapshotInfo {
            sequence_len: out.sequence_len,
            active_kv_bytes: out.active_kv_bytes,
            token_count: out.token_count,
            has_last_step: out.has_last_step,
        })
    }
}

impl Drop for KvSnapshot {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` is an owned snapshot handle returned by `gemma4_kv_snapshot_export`.
        let status = unsafe { raw::gemma4_kv_snapshot_free(self.ptr.as_ptr()) };
        debug_assert_eq!(Status::from_raw(status), Status::Ok);
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NativeLastHiddenView {
    ptr: NonNull<std::ffi::c_void>,
}

impl NativeLastHiddenView {
    /// Returns the opaque native hidden-state pointer.
    ///
    /// The pointer is owned by the KV cache that produced the `StepResult` and is valid only until
    /// that cache is reset, freed, or advanced by another native prefill/decode/verify call.
    pub fn as_ptr(self) -> *mut std::ffi::c_void {
        self.ptr.as_ptr()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepResult {
    pub greedy_token: i32,
    pub greedy_logit: f32,
    pub peak_memory_gb: f32,
    pub peak_rss_mb: f32,
    pub sequence_len: u64,
    pub active_kv_bytes: u64,
    pub accepted_draft_count: u32,
    pub committed_count: u32,
    pub committed_tokens: [i32; 4],
    pub native_last_hidden: Option<NativeLastHiddenView>,
}

impl StepResult {
    pub fn committed_tokens(&self) -> &[i32] {
        &self.committed_tokens
            [..self.committed_count.min(self.committed_tokens.len() as u32) as usize]
    }
}

impl From<raw::Gemma4StepResult> for StepResult {
    fn from(value: raw::Gemma4StepResult) -> Self {
        Self {
            greedy_token: value.greedy_token,
            greedy_logit: value.greedy_logit,
            peak_memory_gb: value.peak_memory_gb,
            peak_rss_mb: value.peak_rss_mb,
            sequence_len: value.sequence_len,
            active_kv_bytes: value.active_kv_bytes,
            accepted_draft_count: value.accepted_draft_count,
            committed_count: value
                .committed_count
                .min(value.committed_tokens.len() as u32),
            committed_tokens: value.committed_tokens,
            native_last_hidden: NonNull::new(value.native_last_hidden)
                .map(|ptr| NativeLastHiddenView { ptr }),
        }
    }
}

pub fn prefill(target: &Target, cache: &mut KvCache, tokens: &[i32]) -> Result<StepResult> {
    let mut out = raw::Gemma4StepResult::default();
    let token_ptr = if tokens.is_empty() {
        ptr::null()
    } else {
        tokens.as_ptr()
    };

    // SAFETY: handles come from safe wrappers; token pointer/count describe `tokens`; `out` is writable.
    check(unsafe {
        raw::gemma4_prefill(
            target.ptr.as_ptr(),
            cache.ptr.as_ptr(),
            token_ptr,
            tokens.len(),
            &mut out,
        )
    })?;

    Ok(out.into())
}

pub fn decode_one(target: &Target, cache: &mut KvCache, token: i32) -> Result<StepResult> {
    let mut out = raw::Gemma4StepResult::default();
    // SAFETY: handles come from safe wrappers and `out` is writable for the duration of the call.
    check(unsafe {
        raw::gemma4_decode_one(target.ptr.as_ptr(), cache.ptr.as_ptr(), token, &mut out)
    })?;

    Ok(out.into())
}

pub fn draft_block(
    drafter: &Drafter<'_>,
    cache: &mut KvCache,
    block_size: NonZeroU32,
) -> Result<Vec<i32>> {
    let mut tokens = vec![0; usize::try_from(block_size.get()).expect("u32 fits usize")];
    let mut count = tokens.len();
    // SAFETY: handles come from safe wrappers; `tokens` is writable and `count` starts at capacity.
    check(unsafe {
        raw::gemma4_mtp_draft_block(
            drafter.ptr.as_ptr(),
            cache.ptr.as_ptr(),
            block_size.get(),
            tokens.as_mut_ptr(),
            &mut count,
        )
    })?;
    tokens.truncate(count);
    Ok(tokens)
}

pub fn verify_tokens(
    target: &Target,
    cache: &mut KvCache,
    draft_tokens: &[i32],
) -> Result<StepResult> {
    let mut out = raw::Gemma4StepResult::default();
    let token_ptr = if draft_tokens.is_empty() {
        ptr::null()
    } else {
        draft_tokens.as_ptr()
    };

    // SAFETY: handles come from safe wrappers; token pointer/count describe `draft_tokens`; `out` is writable.
    check(unsafe {
        raw::gemma4_verify_tokens(
            target.ptr.as_ptr(),
            cache.ptr.as_ptr(),
            token_ptr,
            draft_tokens.len(),
            &mut out,
        )
    })?;

    Ok(out.into())
}

pub fn smoke_prefill(target: &Target, cache: &mut KvCache, tokens: &[i32]) -> Result<()> {
    prefill(target, cache, tokens).map(|_| ())
}

pub fn smoke_decode_one(target: &Target, cache: &mut KvCache, token: i32) -> Result<()> {
    decode_one(target, cache, token).map(|_| ())
}

fn check(status: raw::Gemma4Status) -> Result<()> {
    match Status::from_raw(status) {
        Status::Ok => Ok(()),
        status => Err(Error {
            status,
            message: last_error_message(),
        }),
    }
}

fn last_error_message() -> String {
    let mut buffer = [0 as c_char; 512];
    // SAFETY: `buffer` is writable and its length matches the value passed to native code.
    let status = unsafe { raw::gemma4_get_last_error(buffer.as_mut_ptr(), buffer.len()) };
    if Status::from_raw(status) != Status::Ok {
        return "native error unavailable".to_owned();
    }

    // SAFETY: native code always writes a NUL-terminated string into non-empty buffers.
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn cstr_to_string(value: *const c_char) -> String {
    if value.is_null() {
        return String::new();
    }

    // SAFETY: successful native version query returns pointers to static NUL-terminated strings.
    unsafe { CStr::from_ptr(value) }
        .to_string_lossy()
        .into_owned()
}

fn optional_cstring(value: Option<&str>) -> Result<Option<CString>> {
    value.map(CString::new).transpose().map_err(Into::into)
}

fn optional_ptr(value: &Option<CString>) -> *const c_char {
    value.as_ref().map_or(ptr::null(), |value| value.as_ptr())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        io::Write,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn runtime_version_reports_smoke_backend() {
        let version = runtime_version().expect("runtime version should be available");
        assert_eq!(version.abi_version, 1);
        assert_eq!(version.backend_name, "gemma4_mlx");
        assert!(version.backend_version.starts_with("m03-"));
    }

    #[test]
    fn target_and_kv_lifecycle_work_without_model_loading() {
        let target = Target::load(&LoadConfig::smoke("/tmp/gemma4d-smoke")).expect("target handle");
        let mut cache = KvCache::create(&KvPolicy::default()).expect("kv cache handle");
        cache.reset().expect("kv reset");
        let drafter = Drafter::load(&LoadConfig::smoke("/tmp/gemma4d-smoke-drafter"), &target)
            .expect("drafter handle");
        drop(drafter);
        drop(cache);
        drop(target);
    }

    #[test]
    fn empty_kv_cache_rejects_snapshot_export_and_last_step() {
        let cache = KvCache::create(&KvPolicy::default()).expect("kv cache handle");

        let export_error = cache
            .export_snapshot()
            .expect_err("empty cache should not export a native snapshot");
        assert_eq!(export_error.status(), Status::Cache);

        let last_step_error = cache
            .last_step()
            .expect_err("empty cache should not have a native last step");
        assert_eq!(last_step_error.status(), Status::Cache);
    }

    #[test]
    fn invalid_target_config_returns_error_message() {
        let err = Target::load(&LoadConfig::smoke("")).expect_err("empty model path should fail");
        assert_eq!(err.status(), Status::InvalidArgument);
        assert!(err.message().contains("model_path"));
    }

    #[test]
    fn execution_stubs_return_unsupported_config() {
        let target = Target::load(&LoadConfig::smoke("/tmp/gemma4d-smoke")).expect("target handle");
        let mut cache = KvCache::create(&KvPolicy::default()).expect("kv cache handle");

        let prefill =
            smoke_prefill(&target, &mut cache, &[1, 2, 3]).expect_err("M01 prefill is a stub");
        assert_eq!(prefill.status(), Status::UnsupportedConfig);
        assert!(
            prefill
                .message()
                .contains("requires a loaded Gemma 4 target model")
        );

        let decode = smoke_decode_one(&target, &mut cache, 1).expect_err("M01 decode is a stub");
        assert_eq!(decode.status(), Status::UnsupportedConfig);
        assert!(
            decode
                .message()
                .contains("requires a loaded Gemma 4 target model")
        );
    }

    #[test]
    fn strict_load_reports_missing_model_path() {
        let config = LoadConfig {
            model_path: "/tmp/gemma4d-missing-model-path-for-test".to_owned(),
            model_id: Some("mlx-community/gemma-4-12B-it-4bit".to_owned()),
            model_revision: None,
            expected_architecture: Some("gemma4".to_owned()),
            max_context_tokens: NonZeroU32::new(8192).expect("non-zero"),
            allow_unsupported_config: false,
        };

        let err = Target::load(&config).expect_err("strict load should reject missing model path");
        assert_eq!(err.status(), Status::ModelLoad);
        assert!(err.message().contains("model_path does not exist"));
    }

    #[test]
    #[ignore = "M03 defers native chunked/KV parity while the native graph uses full recompute"]
    fn chunked_prefill_matches_unchunked_for_full_model() {
        panic!("pending full-model chunked prefill equivalence test");
    }

    #[test]
    fn native_graph_prefills_one_token_when_explicitly_enabled() {
        if std::env::var_os("GEMMA4D_FULL_MODEL_TESTS").is_none()
            || std::env::var_os("GEMMA4D_USE_NATIVE_GRAPH").is_none()
        {
            return;
        }

        let version = runtime_version().expect("runtime version should be available");
        assert_ne!(
            version.backend_version, "m03-smoke-no-mlx",
            "native graph full-model tests require GEMMA4D_REQUIRE_MLX=1 at build time"
        );

        let model_path = std::env::var("GEMMA4D_MODEL_PATH")
            .unwrap_or_else(|_| workspace_path("artifacts/models/gemma-4-12B-it-4bit"));
        if !std::path::Path::new(&model_path).exists() {
            return;
        }

        let config = LoadConfig {
            model_path,
            model_id: Some("mlx-community/gemma-4-12B-it-4bit".to_owned()),
            model_revision: None,
            expected_architecture: Some("gemma4".to_owned()),
            max_context_tokens: NonZeroU32::new(8192).expect("non-zero"),
            allow_unsupported_config: false,
        };

        let target = Target::load(&config).expect("native target model should load");
        let mut cache = KvCache::create(&KvPolicy::default()).expect("kv cache handle");
        let step = prefill(&target, &mut cache, &[9259]).expect("native prefill should run");
        assert_eq!(step.sequence_len, 1);
        assert_eq!(step.greedy_token, 236772);
        assert!(step.peak_memory_gb > 0.0);
        assert!(step.active_kv_bytes > 0);
        assert!(step.native_last_hidden.is_some());

        let mut baseline_cache = KvCache::create(&KvPolicy::default()).expect("kv cache handle");
        let baseline_first =
            prefill(&target, &mut baseline_cache, &[9259]).expect("baseline prefill should run");
        let baseline_second = decode_one(&target, &mut baseline_cache, baseline_first.greedy_token)
            .expect("baseline second token should run");
        let baseline_third = decode_one(&target, &mut baseline_cache, baseline_second.greedy_token)
            .expect("baseline third token should run");
        let baseline_fourth = decode_one(&target, &mut baseline_cache, baseline_third.greedy_token)
            .expect("baseline fourth token should run");

        let mut verify_accept_cache =
            KvCache::create(&KvPolicy::default()).expect("kv cache handle");
        let verify_accept_first = prefill(&target, &mut verify_accept_cache, &[9259])
            .expect("verify accept prefill should run");
        let verify_accept = verify_tokens(
            &target,
            &mut verify_accept_cache,
            &[
                verify_accept_first.greedy_token,
                baseline_second.greedy_token,
            ],
        )
        .expect("native verify should accept matching block-size-2 draft");
        assert_eq!(verify_accept.sequence_len, 3);
        assert_eq!(verify_accept.greedy_token, baseline_third.greedy_token);
        assert_eq!(verify_accept.accepted_draft_count, 2);
        assert_eq!(
            verify_accept.committed_tokens(),
            &[
                verify_accept_first.greedy_token,
                baseline_second.greedy_token
            ]
        );
        assert!(verify_accept.native_last_hidden.is_some());

        let rejected_token = if baseline_first.greedy_token == 0 {
            1
        } else {
            0
        };
        let mut verify_reject_cache =
            KvCache::create(&KvPolicy::default()).expect("kv cache handle");
        let _ = prefill(&target, &mut verify_reject_cache, &[9259])
            .expect("verify reject prefill should run");
        let verify_reject = verify_tokens(&target, &mut verify_reject_cache, &[rejected_token])
            .expect("native verify should rollback a rejected draft token");
        assert_eq!(verify_reject.sequence_len, 2);
        assert_eq!(verify_reject.greedy_token, baseline_second.greedy_token);
        assert_eq!(verify_reject.accepted_draft_count, 0);
        assert_eq!(
            verify_reject.committed_tokens(),
            &[baseline_first.greedy_token]
        );
        assert!(verify_reject.native_last_hidden.is_some());
        let reject_follow_up = decode_one(
            &target,
            &mut verify_reject_cache,
            verify_reject.greedy_token,
        )
        .expect("decode after rejected draft should continue from fallback token");
        assert_eq!(reject_follow_up.sequence_len, 3);
        assert_eq!(reject_follow_up.greedy_token, baseline_third.greedy_token);

        let assistant_model_path =
            std::env::var("GEMMA4D_ASSISTANT_MODEL_PATH").unwrap_or_else(|_| {
                workspace_path("artifacts/models/gemma-4-12B-it-qat-assistant-4bit")
            });
        if !std::path::Path::new(&assistant_model_path).exists() {
            return;
        }
        let drafter_config = LoadConfig {
            model_path: assistant_model_path,
            model_id: Some("mlx-community/gemma-4-12B-it-qat-assistant-4bit".to_owned()),
            model_revision: None,
            expected_architecture: Some("gemma4_unified_assistant".to_owned()),
            max_context_tokens: NonZeroU32::new(8192).expect("non-zero"),
            allow_unsupported_config: false,
        };
        let drafter =
            Drafter::load(&drafter_config, &target).expect("assistant manifest should load");
        let draft_after_prefill =
            draft_block(&drafter, &mut cache, NonZeroU32::new(1).expect("non-zero"))
                .expect("assistant prefill block-size-1 draft should run");
        assert_eq!(draft_after_prefill.len(), 1);
        assert!((0..262144).contains(&draft_after_prefill[0]));

        let verified_draft_after_prefill = verify_tokens(&target, &mut cache, &draft_after_prefill)
            .expect("native verify should advance after assistant prefill draft");
        assert_eq!(verified_draft_after_prefill.sequence_len, 2);
        assert_eq!(
            verified_draft_after_prefill.greedy_token,
            baseline_second.greedy_token
        );
        assert!(verified_draft_after_prefill.peak_memory_gb > 0.0);
        assert!(verified_draft_after_prefill.native_last_hidden.is_some());

        let draft_one = draft_block(&drafter, &mut cache, NonZeroU32::new(1).expect("non-zero"))
            .expect("assistant block-size-1 draft should run");
        assert_eq!(draft_one.len(), 1);
        assert!((0..262144).contains(&draft_one[0]));

        let draft_two = draft_block(&drafter, &mut cache, NonZeroU32::new(2).expect("non-zero"))
            .expect("assistant block-size-2 draft should run");
        assert_eq!(draft_two.len(), 2);
        assert!(draft_two.iter().all(|token| (0..262144).contains(token)));

        let verified_draft_two = verify_tokens(&target, &mut cache, &draft_two)
            .expect("native verify should advance after assistant block-size-2 draft");
        assert!((3..=4).contains(&verified_draft_two.sequence_len));
        let expected_next = if verified_draft_two.sequence_len == 3 {
            baseline_third.greedy_token
        } else {
            baseline_fourth.greedy_token
        };
        assert_eq!(verified_draft_two.greedy_token, expected_next);
        assert!(verified_draft_two.native_last_hidden.is_some());

        assert_eq!(baseline_third.sequence_len, 3);
    }

    #[test]
    fn raw_null_pointers_return_invalid_argument() {
        // SAFETY: This deliberately passes null pointers to validate native argument checks.
        let status = unsafe { raw::gemma4_runtime_version(ptr::null_mut()) };
        assert_eq!(Status::from_raw(status), Status::InvalidArgument);
        assert!(last_error_message().contains("runtime_version"));

        // SAFETY: This deliberately passes null pointers to validate native argument checks.
        let status = unsafe { raw::gemma4_free_target(ptr::null_mut()) };
        assert_eq!(Status::from_raw(status), Status::InvalidArgument);
        assert!(last_error_message().contains("free_target"));

        // SAFETY: This deliberately passes null pointers to validate native argument checks.
        let status = unsafe { raw::gemma4_kv_create(ptr::null(), ptr::null_mut()) };
        assert_eq!(Status::from_raw(status), Status::InvalidArgument);
        assert!(last_error_message().contains("kv_create"));

        // SAFETY: This deliberately passes null pointers to validate native argument checks.
        let status =
            unsafe { raw::gemma4_load_drafter(ptr::null(), ptr::null_mut(), ptr::null_mut()) };
        assert_eq!(Status::from_raw(status), Status::InvalidArgument);
        assert!(last_error_message().contains("load_drafter"));

        // SAFETY: This deliberately passes null pointers to validate native argument checks.
        let status = unsafe { raw::gemma4_free_drafter(ptr::null_mut()) };
        assert_eq!(Status::from_raw(status), Status::InvalidArgument);
        assert!(last_error_message().contains("free_drafter"));

        // SAFETY: This deliberately passes null pointers to validate native argument checks.
        let status = unsafe {
            raw::gemma4_mtp_draft_block(
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        assert_eq!(Status::from_raw(status), Status::InvalidArgument);
        assert!(last_error_message().contains("mtp_draft_block"));

        // SAFETY: This deliberately passes null pointers to validate native argument checks.
        let status = unsafe {
            raw::gemma4_verify_tokens(
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null(),
                0,
                ptr::null_mut(),
            )
        };
        assert_eq!(Status::from_raw(status), Status::InvalidArgument);
        assert!(last_error_message().contains("verify_tokens"));
    }

    #[test]
    fn drafter_strict_load_reports_missing_path() {
        let target = Target::load(&LoadConfig::smoke("/tmp/gemma4d-smoke")).expect("target handle");
        let config = LoadConfig {
            model_path: "/tmp/gemma4d-missing-drafter".to_owned(),
            model_id: Some("mlx-community/gemma-4-12B-it-qat-assistant-4bit".to_owned()),
            model_revision: None,
            expected_architecture: Some("gemma4".to_owned()),
            max_context_tokens: NonZeroU32::new(8192).expect("non-zero"),
            allow_unsupported_config: false,
        };

        let err = Drafter::load(&config, &target).expect_err("strict drafter load should fail");
        assert_eq!(err.status(), Status::ModelLoad);
        assert!(err.message().contains("model_path does not exist"));
    }

    #[test]
    fn drafter_strict_load_accepts_assistant_manifest_but_requires_hidden_views() {
        let fixture = write_assistant_fixture();
        let target = Target::load(&LoadConfig::smoke("/tmp/gemma4d-smoke")).expect("target handle");
        let config = LoadConfig {
            model_path: fixture.to_string_lossy().into_owned(),
            model_id: Some("mlx-community/gemma-4-12B-it-qat-assistant-4bit".to_owned()),
            model_revision: None,
            expected_architecture: Some("gemma4_unified_assistant".to_owned()),
            max_context_tokens: NonZeroU32::new(8192).expect("non-zero"),
            allow_unsupported_config: false,
        };

        let drafter = Drafter::load(&config, &target).expect("assistant manifest should load");
        let mut cache = KvCache::create(&KvPolicy::default()).expect("kv cache handle");
        let err = draft_block(&drafter, &mut cache, NonZeroU32::new(1).expect("non-zero"))
            .expect_err("native MTP draft should require hidden views");
        assert_eq!(err.status(), Status::UnsupportedConfig);
        assert!(err.message().contains("last target hidden/shared views"));

        fs::remove_dir_all(fixture).expect("remove assistant fixture");
    }

    #[test]
    fn smoke_drafter_draft_block_is_unsupported() {
        let target = Target::load(&LoadConfig::smoke("/tmp/gemma4d-smoke")).expect("target handle");
        let drafter = Drafter::load(&LoadConfig::smoke("/tmp/gemma4d-smoke-drafter"), &target)
            .expect("drafter handle");
        let mut cache = KvCache::create(&KvPolicy::default()).expect("kv cache handle");

        let err = draft_block(&drafter, &mut cache, NonZeroU32::new(2).expect("non-zero"))
            .expect_err("smoke drafter does not execute");
        assert_eq!(err.status(), Status::UnsupportedConfig);
        assert!(err.message().contains("loaded Gemma 4 MTP assistant"));
    }

    #[test]
    fn verify_tokens_rejects_empty_draft() {
        let target = Target::load(&LoadConfig::smoke("/tmp/gemma4d-smoke")).expect("target handle");
        let mut cache = KvCache::create(&KvPolicy::default()).expect("kv cache handle");

        let err = verify_tokens(&target, &mut cache, &[]).expect_err("empty verify should fail");
        assert_eq!(err.status(), Status::InvalidArgument);
        assert!(err.message().contains("at least one draft token"));
    }

    fn write_assistant_fixture() -> PathBuf {
        let mut dir = std::env::temp_dir();
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        dir.push(format!("gemma4d-assistant-fixture-{unique}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create assistant fixture dir");
        fs::write(dir.join("config.json"), assistant_config_json())
            .expect("write assistant config");
        fs::write(dir.join("tokenizer.json"), "{}").expect("write assistant tokenizer");
        write_safetensors_header(&dir.join("model.safetensors"), assistant_tensor_keys());
        dir
    }

    fn workspace_path(relative: &str) -> String {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("../..");
        path.push(relative);
        path.to_string_lossy().into_owned()
    }

    fn assistant_config_json() -> &'static str {
        r#"{
  "architectures": ["Gemma4UnifiedAssistantForCausalLM"],
  "backbone_hidden_size": 3840,
  "model_type": "gemma4_unified_assistant",
  "quantization": {
    "group_size": 64,
    "bits": 4,
    "mode": "affine"
  },
  "text_config": {
    "attention_k_eq_v": true,
    "hidden_size": 1024,
    "intermediate_size": 8192,
    "model_type": "gemma4_unified_text",
    "num_attention_heads": 16,
    "num_global_key_value_heads": 1,
    "num_hidden_layers": 4,
    "num_key_value_heads": 8,
    "num_kv_shared_layers": 4,
    "sliding_window": 1024,
    "tie_word_embeddings": true,
    "vocab_size": 262144
  }
}"#
    }

    fn assistant_tensor_keys() -> Vec<String> {
        let mut keys = vec![
            "model.embed_tokens.weight".to_owned(),
            "model.embed_tokens.scales".to_owned(),
            "model.embed_tokens.biases".to_owned(),
            "model.norm.weight".to_owned(),
            "pre_projection.weight".to_owned(),
            "pre_projection.scales".to_owned(),
            "pre_projection.biases".to_owned(),
            "post_projection.weight".to_owned(),
            "post_projection.scales".to_owned(),
            "post_projection.biases".to_owned(),
        ];
        for layer in 0..4 {
            let base = format!("model.layers.{layer}");
            keys.extend([
                format!("{base}.input_layernorm.weight"),
                format!("{base}.post_attention_layernorm.weight"),
                format!("{base}.pre_feedforward_layernorm.weight"),
                format!("{base}.post_feedforward_layernorm.weight"),
                format!("{base}.layer_scalar"),
                format!("{base}.self_attn.q_norm.weight"),
            ]);
            for projection in [
                "self_attn.q_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ] {
                let prefix = format!("{base}.{projection}");
                keys.extend([
                    format!("{prefix}.weight"),
                    format!("{prefix}.scales"),
                    format!("{prefix}.biases"),
                ]);
            }
        }
        keys
    }

    fn write_safetensors_header(path: &std::path::Path, keys: Vec<String>) {
        let mut header = "{\"__metadata__\":{\"format\":\"mlx\"}".to_owned();
        for key in keys {
            header.push_str(",\"");
            header.push_str(&key);
            header.push_str("\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,0]}");
        }
        header.push('}');

        let mut file = fs::File::create(path).expect("create safetensors fixture");
        file.write_all(&(header.len() as u64).to_le_bytes())
            .expect("write safetensors header length");
        file.write_all(header.as_bytes())
            .expect("write safetensors header");
    }
}
