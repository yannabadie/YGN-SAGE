//! SQLite persistence for ContextualBandit posteriors.
//! Behind `cognitive` feature flag (requires rusqlite).

use super::bandit::*;
use rusqlite::{params, Connection};

/// SQLite schema for bandit arm posteriors.
const CREATE_TABLE: &str = r#"
    CREATE TABLE IF NOT EXISTS bandit_arms (
        model_id TEXT NOT NULL,
        template TEXT NOT NULL,
        quality_alpha REAL NOT NULL,
        quality_beta REAL NOT NULL,
        cost_shape REAL NOT NULL,
        cost_rate REAL NOT NULL,
        latency_shape REAL NOT NULL,
        latency_rate REAL NOT NULL,
        observation_count INTEGER NOT NULL,
        updated_at TEXT NOT NULL DEFAULT (datetime('now')),
        PRIMARY KEY (model_id, template)
    )
"#;

/// Bandit config table (stores decay_factor, exploration_bonus).
const CREATE_CONFIG_TABLE: &str = r#"
    CREATE TABLE IF NOT EXISTS bandit_config (
        key TEXT PRIMARY KEY,
        value REAL NOT NULL
    )
"#;

/// Save the full bandit state (config + all arm posteriors) to SQLite.
///
/// Uses WAL journal mode for concurrent read access. Arms are upserted
/// via `INSERT OR REPLACE`, so calling save twice updates existing rows.
pub fn save_bandit(bandit: &ContextualBandit, path: &str) -> Result<(), String> {
    let conn = Connection::open(path).map_err(|e| format!("SQLite open: {}", e))?;
    conn.execute_batch("PRAGMA journal_mode=WAL;")
        .map_err(|e| format!("WAL: {}", e))?;
    conn.execute_batch(CREATE_TABLE)
        .map_err(|e| format!("Create table: {}", e))?;
    conn.execute_batch(CREATE_CONFIG_TABLE)
        .map_err(|e| format!("Create config table: {}", e))?;

    // Save config
    conn.execute(
        "INSERT OR REPLACE INTO bandit_config (key, value) VALUES ('decay_factor', ?1)",
        params![bandit.decay_factor()],
    )
    .map_err(|e| format!("Save decay_factor: {}", e))?;
    conn.execute(
        "INSERT OR REPLACE INTO bandit_config (key, value) VALUES ('exploration_bonus', ?1)",
        params![bandit.exploration_bonus()],
    )
    .map_err(|e| format!("Save exploration_bonus: {}", e))?;

    // Upsert each arm posterior
    let mut stmt = conn
        .prepare(
            "INSERT OR REPLACE INTO bandit_arms \
             (model_id, template, quality_alpha, quality_beta, \
              cost_shape, cost_rate, latency_shape, latency_rate, observation_count) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        )
        .map_err(|e| format!("Prepare: {}", e))?;

    for arm in bandit.arms_iter() {
        stmt.execute(params![
            arm.key.model_id,
            arm.key.template,
            arm.quality.alpha,
            arm.quality.beta,
            arm.cost.shape,
            arm.cost.rate,
            arm.latency.shape,
            arm.latency.rate,
            arm.observation_count,
        ])
        .map_err(|e| format!("Insert arm: {}", e))?;
    }

    Ok(())
}

/// Load bandit state from SQLite, reconstructing config and all arm posteriors.
///
/// If config keys are missing, falls back to defaults (decay=0.995, exploration=0.1).
pub fn load_bandit(path: &str) -> Result<ContextualBandit, String> {
    let conn = Connection::open(path).map_err(|e| format!("SQLite open: {}", e))?;
    conn.execute_batch("PRAGMA journal_mode=WAL;")
        .map_err(|e| format!("WAL: {}", e))?;

    // Ensure tables exist (in case we open a fresh DB)
    conn.execute_batch(CREATE_TABLE)
        .map_err(|e| format!("Create table: {}", e))?;
    conn.execute_batch(CREATE_CONFIG_TABLE)
        .map_err(|e| format!("Create config table: {}", e))?;

    // Load config with defaults
    let decay: f64 = conn
        .query_row(
            "SELECT value FROM bandit_config WHERE key = 'decay_factor'",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0.995);

    let exploration: f64 = conn
        .query_row(
            "SELECT value FROM bandit_config WHERE key = 'exploration_bonus'",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0.1);

    let mut bandit = ContextualBandit::create(decay, exploration);

    // Load arm posteriors
    let mut stmt = conn
        .prepare(
            "SELECT model_id, template, quality_alpha, quality_beta, \
             cost_shape, cost_rate, latency_shape, latency_rate, observation_count \
             FROM bandit_arms",
        )
        .map_err(|e| format!("Prepare: {}", e))?;

    let arms = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?, // model_id
                row.get::<_, String>(1)?, // template
                row.get::<_, f64>(2)?,    // quality_alpha
                row.get::<_, f64>(3)?,    // quality_beta
                row.get::<_, f64>(4)?,    // cost_shape
                row.get::<_, f64>(5)?,    // cost_rate
                row.get::<_, f64>(6)?,    // latency_shape
                row.get::<_, f64>(7)?,    // latency_rate
                row.get::<_, u32>(8)?,    // observation_count
            ))
        })
        .map_err(|e| format!("Query: {}", e))?;

    for arm_result in arms {
        let (model_id, template, qa, qb, cs, cr, ls, lr, obs) =
            arm_result.map_err(|e| format!("Row: {}", e))?;
        bandit.restore_arm(model_id, template, qa, qb, cs, cr, ls, lr, obs);
    }

    Ok(bandit)
}
