use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::path::Path;

use serde_json::Value;
use sha2::{Digest, Sha256};
use tracing::warn;

pub const REQUIRED_POSTERIOR_EPOCH: u32 = 1;
pub const A14_EPOCH_GUARD_ERROR_PREFIX: &str = "contaminated_pre_a14_state:";
pub const A14_BYPASS_ENV: &str = "SAGE_BOOT_BYPASS_EPOCH_GUARD";
pub const POSTERIOR_EPOCH_FILENAME: &str = "posterior_epoch.json";
pub const CONTAMINATED_MARKER_FILENAME: &str = "_CONTAMINATED.json";
pub const TOPOLOGY_STATE_MANIFEST_FILENAME: &str = "topology_state_manifest.json";
pub const TOPOLOGY_STATE_MANIFEST_TYPE: &str = "YGN-SAGE_A14_ACTIVE_TOPOLOGY_STATE_MANIFEST";
pub const SMMU_STATE_FILENAME: &str = "smmu_state.json";

const A14_TOPOLOGY_STATE_FILES: [&str; 8] = [
    "bandit_state.db",
    "bandit_state.db-wal",
    "bandit_state.db-shm",
    "archive_state.db",
    "archive_state.db-wal",
    "archive_state.db-shm",
    "engine_extras.json",
    SMMU_STATE_FILENAME,
];

#[derive(Debug, Clone, PartialEq, Eq)]
enum EpochStatus {
    Missing,
    Match(u32),
    Mismatch(u32),
    Malformed(String),
}

impl EpochStatus {
    fn file_epoch_for_log(&self) -> String {
        match self {
            Self::Missing => "missing".to_string(),
            Self::Match(epoch) | Self::Mismatch(epoch) => epoch.to_string(),
            Self::Malformed(_) => "malformed".to_string(),
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct TopologyStateManifest {
    pub manifest_type: String,
    pub epoch: u32,
    pub state_generation_id: String,
    pub created_at_utc: String,
    pub writer: String,
    pub state_files: Vec<TopologyStateFileEntry>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct TopologyStateFileEntry {
    pub name: String,
    pub sha256: String,
    pub size_bytes: u64,
}

pub fn validate_epoch_for_load(dir_path: &Path) -> Result<(), String> {
    let state_files = topology_state_files(dir_path);

    if dir_path.join(CONTAMINATED_MARKER_FILENAME).exists() {
        if bypass_enabled() {
            let epoch_status = read_epoch_status(dir_path);
            warn_bypass(dir_path, &state_files, &epoch_status);
            return Ok(());
        }
        return Err(poison_pill_error(dir_path));
    }

    let epoch_status = read_epoch_status(dir_path);

    if state_files.is_empty() {
        if !matches!(epoch_status, EpochStatus::Missing) {
            warn!(
                event = "a14_epoch_guard_epoch_without_state",
                layer = "rust",
                state_dir = %dir_path.display(),
                required_epoch = REQUIRED_POSTERIOR_EPOCH,
                file_epoch = %epoch_status.file_epoch_for_log(),
                "posterior_epoch_json_present_without_topology_state"
            );
        }
        return Ok(());
    }

    match epoch_status {
        EpochStatus::Match(REQUIRED_POSTERIOR_EPOCH) => {
            let manifest_result = load_topology_state_manifest(dir_path)
                .and_then(|manifest| verify_state_files_against_manifest(dir_path, &manifest));
            match manifest_result {
                Ok(()) => Ok(()),
                Err(error) => allow_bypass_or_error(
                    dir_path,
                    &state_files,
                    &EpochStatus::Match(REQUIRED_POSTERIOR_EPOCH),
                    error,
                ),
            }
        }
        EpochStatus::Match(epoch) | EpochStatus::Mismatch(epoch) => allow_bypass_or_error(
            dir_path,
            &state_files,
            &EpochStatus::Mismatch(epoch),
            epoch_mismatch_error(dir_path, &state_files, epoch),
        ),
        EpochStatus::Missing => allow_bypass_or_error(
            dir_path,
            &state_files,
            &EpochStatus::Missing,
            missing_epoch_error(dir_path, &state_files),
        ),
        EpochStatus::Malformed(reason) => allow_bypass_or_error(
            dir_path,
            &state_files,
            &EpochStatus::Malformed(reason.clone()),
            malformed_epoch_error(dir_path, &state_files, &reason),
        ),
    }
}

pub fn validate_epoch_for_save(dir_path: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir_path)
        .map_err(|err| format!("create dir {}: {}", dir_path.display(), err))?;

    if bypass_enabled() {
        return Err(bypass_save_error(dir_path));
    }

    let state_files = topology_state_files(dir_path);
    if dir_path.join(CONTAMINATED_MARKER_FILENAME).exists() {
        return Err(poison_pill_error(dir_path));
    }

    match read_epoch_status(dir_path) {
        EpochStatus::Match(REQUIRED_POSTERIOR_EPOCH) => {
            if state_files.is_empty() {
                Ok(())
            } else {
                load_topology_state_manifest(dir_path)
                    .and_then(|manifest| verify_state_files_against_manifest(dir_path, &manifest))
            }
        }
        EpochStatus::Match(epoch) | EpochStatus::Mismatch(epoch) => {
            Err(epoch_mismatch_error(dir_path, &state_files, epoch))
        }
        EpochStatus::Missing if state_files.is_empty() => write_clean_epoch_marker(dir_path),
        EpochStatus::Missing => Err(missing_epoch_error(dir_path, &state_files)),
        EpochStatus::Malformed(reason) => {
            Err(malformed_epoch_error(dir_path, &state_files, &reason))
        }
    }
}

pub fn load_topology_state_manifest(dir_path: &Path) -> Result<TopologyStateManifest, String> {
    let manifest_path = dir_path.join(TOPOLOGY_STATE_MANIFEST_FILENAME);
    if !manifest_path.exists() {
        return Err(manifest_missing_error(
            dir_path,
            &topology_state_files(dir_path),
        ));
    }
    let raw = std::fs::read_to_string(&manifest_path)
        .map_err(|err| manifest_malformed_error(dir_path, &format!("read error: {err}")))?;
    serde_json::from_str(&raw).map_err(|err| manifest_malformed_error(dir_path, &err.to_string()))
}

pub fn verify_state_files_against_manifest(
    dir_path: &Path,
    manifest: &TopologyStateManifest,
) -> Result<(), String> {
    let state_files = topology_state_files(dir_path);

    if manifest.manifest_type != TOPOLOGY_STATE_MANIFEST_TYPE {
        return Err(manifest_malformed_error(
            dir_path,
            "manifest_type must be YGN-SAGE_A14_ACTIVE_TOPOLOGY_STATE_MANIFEST",
        ));
    }
    if manifest.epoch != REQUIRED_POSTERIOR_EPOCH {
        return Err(manifest_epoch_mismatch_error(
            dir_path,
            &state_files,
            manifest.epoch,
        ));
    }

    let allowed: BTreeSet<String> = A14_TOPOLOGY_STATE_FILES
        .iter()
        .map(|name| (*name).to_string())
        .collect();
    let mut by_name: BTreeMap<String, TopologyStateFileEntry> = BTreeMap::new();
    for entry in manifest.state_files.iter().cloned() {
        if !allowed.contains(&entry.name) {
            let mut manifest_names: Vec<String> = by_name.keys().cloned().collect();
            manifest_names.push(entry.name);
            manifest_names.sort();
            return Err(manifest_file_set_error(
                dir_path,
                &state_files,
                &manifest_names,
            ));
        }
        if by_name.contains_key(&entry.name) {
            return Err(manifest_malformed_error(
                dir_path,
                &format!("duplicate state file entry: {}", entry.name),
            ));
        }
        if !is_sha256_hex(&entry.sha256) {
            return Err(manifest_malformed_error(
                dir_path,
                &format!("state_files[].sha256 malformed for {}", entry.name),
            ));
        }
        by_name.insert(entry.name.clone(), entry);
    }

    let expected_names: BTreeSet<String> = state_files.iter().cloned().collect();
    let manifest_names_set: BTreeSet<String> = by_name.keys().cloned().collect();
    if expected_names != manifest_names_set {
        let manifest_names: Vec<String> = by_name.keys().cloned().collect();
        return Err(manifest_file_set_error(
            dir_path,
            &state_files,
            &manifest_names,
        ));
    }

    for name in &state_files {
        let path = dir_path.join(name);
        let metadata = std::fs::metadata(&path)
            .map_err(|err| manifest_malformed_error(dir_path, &err.to_string()))?;
        let entry = by_name
            .get(name)
            .expect("manifest set equality should guarantee an entry");
        let actual_size = metadata.len();
        if actual_size != entry.size_bytes {
            return Err(manifest_size_mismatch_error(
                dir_path,
                name,
                entry.size_bytes,
                actual_size,
            ));
        }
        let actual_sha256 =
            sha256_file(&path).map_err(|err| manifest_malformed_error(dir_path, &err))?;
        if actual_sha256 != entry.sha256 {
            return Err(manifest_sha256_mismatch_error(
                dir_path,
                name,
                &entry.sha256,
                &actual_sha256,
            ));
        }
    }

    Ok(())
}

pub fn write_topology_state_manifest(dir_path: &Path, writer: &str) -> Result<(), String> {
    let mut entries = Vec::new();
    for name in topology_state_files(dir_path) {
        let path = dir_path.join(&name);
        let size_bytes = std::fs::metadata(&path)
            .map_err(|err| format!("stat topology state file {}: {}", path.display(), err))?
            .len();
        entries.push(TopologyStateFileEntry {
            name,
            sha256: sha256_file(&path)?,
            size_bytes,
        });
    }

    let manifest = TopologyStateManifest {
        manifest_type: TOPOLOGY_STATE_MANIFEST_TYPE.to_string(),
        epoch: REQUIRED_POSTERIOR_EPOCH,
        state_generation_id: ulid::Ulid::new().to_string(),
        created_at_utc: chrono::Utc::now().to_rfc3339(),
        writer: writer.to_string(),
        state_files: entries,
    };
    let mut bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|err| format!("serialize topology state manifest: {err}"))?;
    bytes.push(b'\n');
    write_bytes_atomic(&dir_path.join(TOPOLOGY_STATE_MANIFEST_FILENAME), &bytes)
}

fn allow_bypass_or_error(
    dir_path: &Path,
    state_files: &[String],
    epoch_status: &EpochStatus,
    error: String,
) -> Result<(), String> {
    if bypass_enabled() {
        warn_bypass(dir_path, state_files, epoch_status);
        return Ok(());
    }
    Err(error)
}

fn topology_state_files(dir_path: &Path) -> Vec<String> {
    A14_TOPOLOGY_STATE_FILES
        .iter()
        .filter(|name| dir_path.join(name).exists())
        .map(|name| (*name).to_string())
        .collect()
}

fn read_epoch_status(dir_path: &Path) -> EpochStatus {
    let epoch_path = dir_path.join(POSTERIOR_EPOCH_FILENAME);
    if !epoch_path.exists() {
        return EpochStatus::Missing;
    }

    let raw = match std::fs::read_to_string(&epoch_path) {
        Ok(raw) => raw,
        Err(err) => return EpochStatus::Malformed(short_reason(format!("read error: {err}"))),
    };
    let value: Value = match serde_json::from_str(&raw) {
        Ok(value) => value,
        Err(err) => return EpochStatus::Malformed(short_reason(err.to_string())),
    };
    let Some(epoch_value) = value.get("epoch") else {
        return EpochStatus::Malformed("missing integer field epoch".to_string());
    };
    let Some(epoch) = epoch_value.as_u64() else {
        return EpochStatus::Malformed("epoch must be an integer".to_string());
    };
    if epoch > u32::MAX as u64 {
        return EpochStatus::Malformed("epoch exceeds u32 range".to_string());
    }

    let epoch = epoch as u32;
    if epoch == REQUIRED_POSTERIOR_EPOCH {
        EpochStatus::Match(epoch)
    } else {
        EpochStatus::Mismatch(epoch)
    }
}

fn write_clean_epoch_marker(dir_path: &Path) -> Result<(), String> {
    let epoch_path = dir_path.join(POSTERIOR_EPOCH_FILENAME);
    let payload = serde_json::json!({
        "epoch": REQUIRED_POSTERIOR_EPOCH,
        "started_utc": chrono::Utc::now().to_rfc3339(),
        "reason": "auto-created clean topology posterior epoch before first save_state",
        "policy": "all bandit/MAP-Elites updates for this state are post-A14 clean-epoch updates",
    });
    let mut bytes = serde_json::to_vec_pretty(&payload)
        .map_err(|err| format!("serialize posterior epoch marker: {err}"))?;
    bytes.push(b'\n');
    write_bytes_atomic(&epoch_path, &bytes)
}

fn missing_epoch_error(dir_path: &Path, state_files: &[String]) -> String {
    format!(
        "{A14_EPOCH_GUARD_ERROR_PREFIX} explicit epoch required when topology state exists; \
         state_dir={}; state_files={}; epoch_file=missing; required_epoch={}; bypass_env={}",
        dir_path.display(),
        csv(state_files),
        REQUIRED_POSTERIOR_EPOCH,
        A14_BYPASS_ENV
    )
}

fn epoch_mismatch_error(dir_path: &Path, state_files: &[String], file_epoch: u32) -> String {
    format!(
        "{A14_EPOCH_GUARD_ERROR_PREFIX} epoch mismatch: file={} required={}; \
         state_dir={}; state_files={}; bypass_env={}",
        file_epoch,
        REQUIRED_POSTERIOR_EPOCH,
        dir_path.display(),
        csv(state_files),
        A14_BYPASS_ENV
    )
}

fn malformed_epoch_error(dir_path: &Path, state_files: &[String], reason: &str) -> String {
    format!(
        "{A14_EPOCH_GUARD_ERROR_PREFIX} posterior_epoch.json malformed: {}; \
         state_dir={}; state_files={}; required_epoch={}; bypass_env={}",
        short_reason(reason.to_string()),
        dir_path.display(),
        csv(state_files),
        REQUIRED_POSTERIOR_EPOCH,
        A14_BYPASS_ENV
    )
}

fn poison_pill_error(dir_path: &Path) -> String {
    format!(
        "{A14_EPOCH_GUARD_ERROR_PREFIX} poison pill marker present in state dir: {}; \
         state_dir={}; bypass_env={}",
        CONTAMINATED_MARKER_FILENAME,
        dir_path.display(),
        A14_BYPASS_ENV
    )
}

fn bypass_save_error(dir_path: &Path) -> String {
    format!(
        "{A14_EPOCH_GUARD_ERROR_PREFIX} save disabled while {}=1; state_dir={}",
        A14_BYPASS_ENV,
        dir_path.display()
    )
}

fn manifest_missing_error(dir_path: &Path, state_files: &[String]) -> String {
    format!(
        "{A14_EPOCH_GUARD_ERROR_PREFIX} state files present but {} missing; \
         state_dir={}; state_files={}; required_epoch={}; bypass_env={}",
        TOPOLOGY_STATE_MANIFEST_FILENAME,
        dir_path.display(),
        csv(state_files),
        REQUIRED_POSTERIOR_EPOCH,
        A14_BYPASS_ENV
    )
}

fn manifest_malformed_error(dir_path: &Path, reason: &str) -> String {
    format!(
        "{A14_EPOCH_GUARD_ERROR_PREFIX} {} malformed: {}; state_dir={}; \
         required_epoch={}; bypass_env={}",
        TOPOLOGY_STATE_MANIFEST_FILENAME,
        short_reason(reason.to_string()),
        dir_path.display(),
        REQUIRED_POSTERIOR_EPOCH,
        A14_BYPASS_ENV
    )
}

fn manifest_epoch_mismatch_error(dir_path: &Path, state_files: &[String], epoch: u32) -> String {
    format!(
        "{A14_EPOCH_GUARD_ERROR_PREFIX} {} epoch mismatch: file={} required={}; \
         state_dir={}; state_files={}; bypass_env={}",
        TOPOLOGY_STATE_MANIFEST_FILENAME,
        epoch,
        REQUIRED_POSTERIOR_EPOCH,
        dir_path.display(),
        csv(state_files),
        A14_BYPASS_ENV
    )
}

fn manifest_file_set_error(
    dir_path: &Path,
    state_files: &[String],
    manifest_names: &[String],
) -> String {
    format!(
        "{A14_EPOCH_GUARD_ERROR_PREFIX} {} state file set mismatch; state_dir={}; \
         state_files={}; manifest_files={}; required_epoch={}; bypass_env={}",
        TOPOLOGY_STATE_MANIFEST_FILENAME,
        dir_path.display(),
        csv(state_files),
        csv(manifest_names),
        REQUIRED_POSTERIOR_EPOCH,
        A14_BYPASS_ENV
    )
}

fn manifest_size_mismatch_error(dir_path: &Path, name: &str, expected: u64, actual: u64) -> String {
    format!(
        "{A14_EPOCH_GUARD_ERROR_PREFIX} {} size_bytes mismatch on {}; \
         expected={} actual={}; state_dir={}; bypass_env={}",
        TOPOLOGY_STATE_MANIFEST_FILENAME,
        name,
        expected,
        actual,
        dir_path.display(),
        A14_BYPASS_ENV
    )
}

fn manifest_sha256_mismatch_error(
    dir_path: &Path,
    name: &str,
    expected: &str,
    actual: &str,
) -> String {
    format!(
        "{A14_EPOCH_GUARD_ERROR_PREFIX} {} sha256 mismatch on {}; expected={} actual={}; \
         state_dir={}; bypass_env={}",
        TOPOLOGY_STATE_MANIFEST_FILENAME,
        name,
        expected,
        actual,
        dir_path.display(),
        A14_BYPASS_ENV
    )
}

fn csv(state_files: &[String]) -> String {
    state_files.join(",")
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file =
        std::fs::File::open(path).map_err(|err| format!("open {}: {}", path.display(), err))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|err| format!("read {}: {}", path.display(), err))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn write_bytes_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let tmp_path = path.with_file_name(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("state"),
        ulid::Ulid::new()
    ));

    // Inner closure: if anything fails, the outer cleanup-on-failure
    // block below removes the .tmp file before propagating. Cycle-13
    // B Q4 follow-up (cgpro deep VERIFY 2026-05-06): pre-fix, a
    // failure on rename (e.g. EACCES on Windows AV scanner, EXDEV
    // cross-device, ENOSPC parent dir) propagated the error but
    // left `.<name>.<ulid>.tmp` on disk forever. Each retry gets a
    // fresh ulid suffix, so the leak grows monotonically. Clean up
    // best-effort to keep state dirs tidy.
    let result: Result<(), String> = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)
            .map_err(|err| format!("create temp file {}: {}", tmp_path.display(), err))?;
        file.write_all(bytes)
            .map_err(|err| format!("write temp file {}: {}", tmp_path.display(), err))?;
        file.sync_all()
            .map_err(|err| format!("sync temp file {}: {}", tmp_path.display(), err))?;
        drop(file);
        std::fs::rename(&tmp_path, path)
            .map_err(|err| format!("install {}: {}", path.display(), err))?;
        Ok(())
    })();

    // Best-effort cleanup on failure path. On rename success the
    // .tmp no longer exists (rename moved it); remove_file returns
    // NotFound which we swallow. On any earlier failure the .tmp
    // does exist and remove_file should clear it. We swallow this
    // error too because the original failure is what callers need
    // to see — a leaked .tmp is recoverable; a misleading error
    // message is not.
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp_path);
    }

    result
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn short_reason(reason: String) -> String {
    let normalized = reason.replace(['\r', '\n', ';'], " ");
    normalized.chars().take(180).collect()
}

fn bypass_enabled() -> bool {
    std::env::var(A14_BYPASS_ENV).is_ok_and(|value| value == "1")
}

fn operator_id() -> String {
    std::env::var("SAGE_OPERATOR_ID")
        .or_else(|_| std::env::var("USER"))
        .or_else(|_| std::env::var("USERNAME"))
        .unwrap_or_else(|_| "unknown".to_string())
}

fn bypass_reason() -> String {
    std::env::var("SAGE_BOOT_BYPASS_REASON").unwrap_or_else(|_| "unspecified".to_string())
}

fn warn_bypass(dir_path: &Path, state_files: &[String], epoch_status: &EpochStatus) {
    warn!(
        event = "a14_epoch_guard_bypass",
        layer = "rust",
        state_dir = %dir_path.display(),
        required_epoch = REQUIRED_POSTERIOR_EPOCH,
        file_epoch = %epoch_status.file_epoch_for_log(),
        state_files = %csv(state_files),
        operator = %operator_id(),
        reason = %bypass_reason(),
        "a14_epoch_guard_bypass"
    );
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, OnceLock};

    // Cycle-11 CI debug 2026-05-05: TOPOLOGY_STATE_MANIFEST_FILENAME
    // is used at line 840 inside `#[cfg(feature = "cognitive")]` test
    // — clippy --fix wrongly removed it on a default-features build
    // where the cognitive test isn't compiled. `#[allow(unused_imports)]`
    // keeps the default clippy clean while preserving the cognitive
    // build.
    #[allow(unused_imports)]
    use super::{
        validate_epoch_for_load, validate_epoch_for_save, write_topology_state_manifest,
        A14_BYPASS_ENV, A14_EPOCH_GUARD_ERROR_PREFIX, CONTAMINATED_MARKER_FILENAME,
        POSTERIOR_EPOCH_FILENAME, TOPOLOGY_STATE_MANIFEST_FILENAME,
    };

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let guard = LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("epoch guard test env lock poisoned");
        std::env::remove_var(A14_BYPASS_ENV);
        guard
    }

    fn temp_state_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sage_a14_epoch_{}", ulid::Ulid::new()));
        std::fs::create_dir_all(&dir).expect("create temp state dir");
        dir
    }

    fn touch(path: &Path) {
        std::fs::write(path, b"legacy-state").expect("write test state file");
    }

    fn write_epoch(dir: &Path, epoch_json: &str) {
        std::fs::write(dir.join(POSTERIOR_EPOCH_FILENAME), epoch_json)
            .expect("write epoch fixture");
    }

    fn assert_a14_error(result: Result<(), String>) -> String {
        let err = result.expect_err("expected A14 epoch guard error");
        assert!(
            err.starts_with(A14_EPOCH_GUARD_ERROR_PREFIX),
            "unexpected error: {err}"
        );
        err
    }

    #[test]
    fn test_load_state_fails_closed_on_missing_epoch_with_state() {
        let _guard = env_lock();
        let dir = temp_state_dir();
        touch(&dir.join("bandit_state.db"));

        let err = assert_a14_error(validate_epoch_for_load(&dir));
        assert!(err.contains("epoch_file=missing"));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_load_state_fails_closed_on_epoch_zero() {
        let _guard = env_lock();
        let dir = temp_state_dir();
        touch(&dir.join("bandit_state.db"));
        write_epoch(&dir, r#"{"epoch":0}"#);

        let err = assert_a14_error(validate_epoch_for_load(&dir));
        assert!(err.contains("epoch mismatch: file=0 required=1"));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_load_state_fails_closed_on_epoch_mismatch() {
        let _guard = env_lock();
        let dir = temp_state_dir();
        touch(&dir.join("archive_state.db"));
        write_epoch(&dir, r#"{"epoch":2}"#);

        let err = assert_a14_error(validate_epoch_for_load(&dir));
        assert!(err.contains("epoch mismatch: file=2 required=1"));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_load_state_fails_closed_on_malformed_json() {
        let _guard = env_lock();
        let dir = temp_state_dir();
        touch(&dir.join("engine_extras.json"));
        write_epoch(&dir, r#"{"epoch":"1"}"#);

        let err = assert_a14_error(validate_epoch_for_load(&dir));
        assert!(err.contains("posterior_epoch.json malformed"));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_load_state_loads_normally_on_epoch_match() {
        let _guard = env_lock();
        let dir = temp_state_dir();
        touch(&dir.join("engine_extras.json"));
        write_epoch(&dir, r#"{"epoch":1,"reason":"clean"}"#);
        write_topology_state_manifest(&dir, "test").expect("write manifest fixture");

        validate_epoch_for_load(&dir).expect("matching epoch should load");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_load_state_fails_closed_on_restore_over_valid_epoch() {
        let _guard = env_lock();
        let dir = temp_state_dir();
        write_epoch(&dir, r#"{"epoch":1,"reason":"post-reset clean epoch"}"#);
        touch(&dir.join("bandit_state.db"));

        let err = assert_a14_error(validate_epoch_for_load(&dir));
        assert!(err.contains("topology_state_manifest"));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_load_state_fails_closed_on_manifest_sha_mismatch() {
        let _guard = env_lock();
        let dir = temp_state_dir();
        touch(&dir.join("bandit_state.db"));
        write_epoch(&dir, r#"{"epoch":1,"reason":"clean"}"#);
        write_topology_state_manifest(&dir, "test").expect("write manifest fixture");
        std::fs::write(dir.join("bandit_state.db"), b"tampered!!!!").expect("tamper state file");

        let err = assert_a14_error(validate_epoch_for_load(&dir));
        assert!(err.contains("topology_state_manifest.json sha256 mismatch on bandit_state.db"));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_load_state_fails_closed_on_manifest_file_set_mismatch() {
        let _guard = env_lock();
        let dir = temp_state_dir();
        touch(&dir.join("bandit_state.db"));
        write_epoch(&dir, r#"{"epoch":1,"reason":"clean"}"#);
        write_topology_state_manifest(&dir, "test").expect("write manifest fixture");
        touch(&dir.join("archive_state.db"));

        let err = assert_a14_error(validate_epoch_for_load(&dir));
        assert!(err.contains("topology_state_manifest.json state file set mismatch"));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_write_manifest_replaces_existing_manifest() {
        let _guard = env_lock();
        let dir = temp_state_dir();
        touch(&dir.join("bandit_state.db"));

        write_topology_state_manifest(&dir, "first").expect("write first manifest");
        write_topology_state_manifest(&dir, "second").expect("replace manifest atomically");

        validate_epoch_for_save(&dir).expect_err("epoch is still required for save");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_load_state_cold_start_clean() {
        let _guard = env_lock();
        let dir = temp_state_dir();

        validate_epoch_for_load(&dir).expect("empty state dir is a clean cold start");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_load_state_warns_on_epoch_without_state() {
        let _guard = env_lock();
        let dir = temp_state_dir();
        write_epoch(&dir, r#"{"epoch":1}"#);

        validate_epoch_for_load(&dir).expect("epoch without state should cold-start");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(feature = "cognitive")]
    #[test]
    fn test_load_state_fails_before_sqlite_deserialization() {
        let _guard = env_lock();
        let dir = temp_state_dir();
        touch(&dir.join("bandit_state.db"));

        let mut engine = crate::topology::engine::TopologyEngine::new();
        let err = engine
            .load_state(dir.to_str().expect("utf-8 temp path"))
            .expect_err("guard should fail before SQLite opens the corrupt DB");
        assert!(
            err.starts_with(A14_EPOCH_GUARD_ERROR_PREFIX),
            "unexpected error: {err}"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_load_state_detects_wal_shm_as_state() {
        let _guard = env_lock();
        let dir = temp_state_dir();
        touch(&dir.join("archive_state.db-wal"));

        let err = assert_a14_error(validate_epoch_for_load(&dir));
        assert!(err.contains("archive_state.db-wal"));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_load_state_fails_on_contaminated_marker_present() {
        let _guard = env_lock();
        let dir = temp_state_dir();
        std::fs::write(dir.join(CONTAMINATED_MARKER_FILENAME), "{}").expect("write poison marker");

        let err = assert_a14_error(validate_epoch_for_load(&dir));
        assert!(err.contains("poison pill marker present"));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_load_state_bypass_allows_forensic_load_and_warns() {
        let _guard = env_lock();
        let dir = temp_state_dir();
        touch(&dir.join("bandit_state.db"));
        std::env::set_var(A14_BYPASS_ENV, "1");

        validate_epoch_for_load(&dir).expect("forensic bypass should allow load only");

        std::env::remove_var(A14_BYPASS_ENV);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(feature = "cognitive")]
    #[test]
    fn test_save_state_creates_epoch_on_clean_first_save() {
        let _guard = env_lock();
        let dir = temp_state_dir();

        let engine = crate::topology::engine::TopologyEngine::new();
        engine
            .save_state(dir.to_str().expect("utf-8 temp path"))
            .expect("clean first save should create epoch before persistence");

        let epoch = std::fs::read_to_string(dir.join(POSTERIOR_EPOCH_FILENAME))
            .expect("posterior epoch should be created");
        assert!(epoch.contains(r#""epoch": 1"#) || epoch.contains(r#""epoch":1"#));
        assert!(dir.join("bandit_state.db").exists());
        assert!(dir.join(TOPOLOGY_STATE_MANIFEST_FILENAME).exists());
        validate_epoch_for_load(&dir).expect("freshly saved state should validate");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(feature = "cognitive")]
    #[test]
    fn test_save_state_refuses_under_bypass_with_state() {
        let _guard = env_lock();
        let dir = temp_state_dir();
        touch(&dir.join("bandit_state.db"));
        std::env::set_var(A14_BYPASS_ENV, "1");

        let engine = crate::topology::engine::TopologyEngine::new();
        let err = engine
            .save_state(dir.to_str().expect("utf-8 temp path"))
            .expect_err("save must be disabled under forensic bypass");
        assert!(err.starts_with(A14_EPOCH_GUARD_ERROR_PREFIX));
        assert!(err.contains("save disabled while SAGE_BOOT_BYPASS_EPOCH_GUARD=1"));

        std::env::remove_var(A14_BYPASS_ENV);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(feature = "cognitive")]
    #[test]
    fn test_save_state_refuses_missing_epoch_when_state_exists() {
        let _guard = env_lock();
        let dir = temp_state_dir();
        touch(&dir.join("bandit_state.db"));

        let engine = crate::topology::engine::TopologyEngine::new();
        let err = engine
            .save_state(dir.to_str().expect("utf-8 temp path"))
            .expect_err("save must not persist over legacy state with missing epoch");
        assert!(err.starts_with(A14_EPOCH_GUARD_ERROR_PREFIX));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_save_state_refuses_malformed_or_mismatched_epoch() {
        let _guard = env_lock();

        let malformed = temp_state_dir();
        write_epoch(&malformed, r#"{"epoch":"1"}"#);
        let malformed_err = assert_a14_error(validate_epoch_for_save(&malformed));
        assert!(malformed_err.contains("posterior_epoch.json malformed"));

        let mismatched = temp_state_dir();
        write_epoch(&mismatched, r#"{"epoch":2}"#);
        let mismatched_err = assert_a14_error(validate_epoch_for_save(&mismatched));
        assert!(mismatched_err.contains("epoch mismatch: file=2 required=1"));

        let _ = std::fs::remove_dir_all(malformed);
        let _ = std::fs::remove_dir_all(mismatched);
    }

    // Cycle-13 B Q4 follow-up (cgpro deep VERIFY 2026-05-06):
    // `write_bytes_atomic` previously leaked `.<name>.<ulid>.tmp` on
    // the rename-failure path. Validate the cleanup-on-failure
    // behaviour here.
    #[test]
    fn test_write_bytes_atomic_success_no_tmp_file_left() {
        let dir = temp_state_dir();
        let target = dir.join("target.json");

        super::write_bytes_atomic(&target, b"{\"hello\":\"world\"}\n").expect("write succeeds");

        assert!(target.exists(), "target file should exist after success");
        assert_eq!(
            std::fs::read(&target).unwrap(),
            b"{\"hello\":\"world\"}\n",
            "target bytes match"
        );

        // No leftover .tmp files in the dir on success.
        let leftover_tmps: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .map(|s| s.ends_with(".tmp"))
                    .unwrap_or(false)
            })
            .collect();
        assert!(
            leftover_tmps.is_empty(),
            "no .tmp files should remain on success path; found: {:?}",
            leftover_tmps
                .iter()
                .map(|e| e.file_name())
                .collect::<Vec<_>>()
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_write_bytes_atomic_cleans_tmp_on_rename_failure() {
        // We force a rename failure by making the destination a
        // directory: `std::fs::rename(file, dir)` fails on every OS
        // we care about (EISDIR-class on Linux/macOS,
        // ERROR_ACCESS_DENIED on Windows). Pre-fix the .tmp would
        // leak; post-fix the cleanup block removes it.
        let dir = temp_state_dir();
        let blocker_dir = dir.join("blocked_path");
        std::fs::create_dir(&blocker_dir).expect("create blocker dir");

        let result = super::write_bytes_atomic(&blocker_dir, b"any bytes");
        assert!(result.is_err(), "rename onto a directory must fail");

        // No `.tmp` file should remain in the parent dir.
        let leftover_tmps: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter_map(|entry| {
                let name = entry.file_name().to_str()?.to_string();
                if name.ends_with(".tmp") {
                    Some(name)
                } else {
                    None
                }
            })
            .collect();
        assert!(
            leftover_tmps.is_empty(),
            "rename failure must clean up .tmp; leaked: {:?}",
            leftover_tmps
        );

        let _ = std::fs::remove_dir_all(dir);
    }
}
