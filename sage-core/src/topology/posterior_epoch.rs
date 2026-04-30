use std::path::Path;

use serde_json::Value;
use tracing::warn;

pub const REQUIRED_POSTERIOR_EPOCH: u32 = 1;
pub const A14_EPOCH_GUARD_ERROR_PREFIX: &str = "contaminated_pre_a14_state:";
pub const A14_BYPASS_ENV: &str = "SAGE_BOOT_BYPASS_EPOCH_GUARD";
pub const POSTERIOR_EPOCH_FILENAME: &str = "posterior_epoch.json";
pub const CONTAMINATED_MARKER_FILENAME: &str = "_CONTAMINATED.json";

const A14_TOPOLOGY_STATE_FILES: [&str; 7] = [
    "bandit_state.db",
    "bandit_state.db-wal",
    "bandit_state.db-shm",
    "archive_state.db",
    "archive_state.db-wal",
    "archive_state.db-shm",
    "engine_extras.json",
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
        EpochStatus::Match(REQUIRED_POSTERIOR_EPOCH) => Ok(()),
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

    let state_files = topology_state_files(dir_path);
    if dir_path.join(CONTAMINATED_MARKER_FILENAME).exists() {
        return Err(poison_pill_error(dir_path));
    }

    match read_epoch_status(dir_path) {
        EpochStatus::Match(REQUIRED_POSTERIOR_EPOCH) => Ok(()),
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
    let tmp_path = dir_path.join(format!(
        ".{}.{}.tmp",
        POSTERIOR_EPOCH_FILENAME,
        ulid::Ulid::new()
    ));
    let payload = serde_json::json!({
        "epoch": REQUIRED_POSTERIOR_EPOCH,
        "started_utc": chrono::Utc::now().to_rfc3339(),
        "reason": "auto-created clean topology posterior epoch before first save_state",
        "policy": "all bandit/MAP-Elites updates for this state are post-A14 clean-epoch updates",
    });
    let bytes = serde_json::to_vec_pretty(&payload)
        .map_err(|err| format!("serialize posterior epoch marker: {err}"))?;
    std::fs::write(&tmp_path, bytes)
        .map_err(|err| format!("write posterior epoch temp file: {err}"))?;
    std::fs::rename(&tmp_path, &epoch_path)
        .map_err(|err| format!("install posterior epoch marker: {err}"))?;
    Ok(())
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

fn csv(state_files: &[String]) -> String {
    state_files.join(",")
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

    use super::{
        validate_epoch_for_load, validate_epoch_for_save, A14_BYPASS_ENV,
        A14_EPOCH_GUARD_ERROR_PREFIX, CONTAMINATED_MARKER_FILENAME, POSTERIOR_EPOCH_FILENAME,
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

        validate_epoch_for_load(&dir).expect("matching epoch should load");

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
}
