//! Deterministic parser for iGSM (Integer Grade School Math) problems.
//!
//! Converts natural language word problems (Meta's Physics of Language Models
//! format) into equation systems solvable by `solve_equation_system()`.
//!
//! No LLM needed — the grammar is fixed and deterministic.
//! Based on SatLM (NeurIPS 2023): separating formalization from solving.

use pyo3::prelude::*;
use std::collections::HashMap;

use super::smt::solve_equation_system;

/// Normalize entity reference to a variable name.
/// "each Zoo A's Pelican" → "zoo_a_pelican"
/// "each Bread's Rye" → "bread_rye"
fn normalize_ref(s: &str) -> String {
    let s = s.trim();
    // Remove "each " prefix
    let s = s.strip_prefix("each ").unwrap_or(s);
    // Remove possessive 's
    let s = s.replace("'s ", "_").replace("'s", "_");
    // Lowercase, spaces to underscores, remove non-alphanumeric except underscore
    s.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
        .collect::<String>()
        .split('_')
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("_")
}

/// Parse a single iGSM statement into (variable_name, expression).
///
/// Patterns:
/// - "The number of each X equals N." → (x, "N")
/// - "...equals N times as much as each Y." → (x, "N * y")
/// - "...equals the sum of each Y and each Z." → (x, "y + z")
/// - "...equals the difference of each Y and each Z." → (x, "y - z")
/// - "...equals N more than each Y." → (x, "y + N")
/// - "...equals each Y." → (x, "y")
fn parse_statement(stmt: &str) -> Option<(String, String)> {
    let stmt = stmt.trim().trim_end_matches('.');

    // Must start with "The number of each"
    let rest = stmt.strip_prefix("The number of ")?;

    // Split at " equals "
    let parts: Vec<&str> = rest.splitn(2, " equals ").collect();
    if parts.len() != 2 {
        return None;
    }

    let var_name = normalize_ref(parts[0]);
    let rhs = parts[1].trim();

    // Pattern: "N times as much as the sum/difference of REF1 and REF2" (compound)
    if rhs.contains("times as much as the sum of ") {
        let sub: Vec<&str> = rhs.splitn(2, " times as much as the sum of ").collect();
        if sub.len() == 2 {
            let n = sub[0].trim().parse::<i64>().ok()?;
            let parts: Vec<&str> = sub[1].splitn(2, " and ").collect();
            if parts.len() == 2 {
                let ref1 = normalize_ref(parts[0]);
                let ref2 = normalize_ref(parts[1]);
                return Some((var_name, format!("{} * ({} + {})", n, ref1, ref2)));
            }
        }
    }
    if rhs.contains("times as much as the difference of ") {
        let sub: Vec<&str> = rhs.splitn(2, " times as much as the difference of ").collect();
        if sub.len() == 2 {
            let n = sub[0].trim().parse::<i64>().ok()?;
            let parts: Vec<&str> = sub[1].splitn(2, " and ").collect();
            if parts.len() == 2 {
                let ref1 = normalize_ref(parts[0]);
                let ref2 = normalize_ref(parts[1]);
                return Some((var_name, format!("{} * ({} - {})", n, ref1, ref2)));
            }
        }
    }

    // Pattern: "N times as much as each REF" (simple)
    if rhs.contains("times as much as") {
        let sub: Vec<&str> = rhs.splitn(2, " times as much as ").collect();
        if sub.len() == 2 {
            let n = sub[0].trim().parse::<i64>().ok()?;
            let ref_name = normalize_ref(sub[1]);
            return Some((var_name, format!("{} * {}", n, ref_name)));
        }
    }

    // Pattern: "the sum of each REF1 and each REF2"
    if rhs.starts_with("the sum of ") {
        let inner = rhs.strip_prefix("the sum of ")?;
        let parts: Vec<&str> = inner.splitn(2, " and ").collect();
        if parts.len() == 2 {
            let ref1 = normalize_ref(parts[0]);
            let ref2 = normalize_ref(parts[1]);
            return Some((var_name, format!("{} + {}", ref1, ref2)));
        }
    }

    // Pattern: "the difference of each REF1 and each REF2"
    if rhs.starts_with("the difference of ") {
        let inner = rhs.strip_prefix("the difference of ")?;
        let parts: Vec<&str> = inner.splitn(2, " and ").collect();
        if parts.len() == 2 {
            let ref1 = normalize_ref(parts[0]);
            let ref2 = normalize_ref(parts[1]);
            return Some((var_name, format!("{} - {}", ref1, ref2)));
        }
    }

    // Pattern: "N more than the sum of REF1 and REF2" (compound)
    if rhs.contains(" more than the sum of ") {
        let sub: Vec<&str> = rhs.splitn(2, " more than the sum of ").collect();
        if sub.len() == 2 {
            let n = sub[0].trim().parse::<i64>().ok()?;
            let inner = sub[1];
            let parts: Vec<&str> = inner.splitn(2, " and ").collect();
            if parts.len() == 2 {
                let ref1 = normalize_ref(parts[0]);
                let ref2 = normalize_ref(parts[1]);
                return Some((var_name, format!("{} + {} + {}", ref1, ref2, n)));
            }
        }
    }

    // Pattern: "N more than the difference of REF1 and REF2" (compound)
    if rhs.contains(" more than the difference of ") {
        let sub: Vec<&str> = rhs.splitn(2, " more than the difference of ").collect();
        if sub.len() == 2 {
            let n = sub[0].trim().parse::<i64>().ok()?;
            let inner = sub[1];
            let parts: Vec<&str> = inner.splitn(2, " and ").collect();
            if parts.len() == 2 {
                let ref1 = normalize_ref(parts[0]);
                let ref2 = normalize_ref(parts[1]);
                return Some((var_name, format!("{} - {} + {}", ref1, ref2, n)));
            }
        }
    }

    // Pattern: "N more than each REF" (simple)
    if rhs.contains(" more than ") {
        let sub: Vec<&str> = rhs.splitn(2, " more than ").collect();
        if sub.len() == 2 {
            let n = sub[0].trim().parse::<i64>().ok()?;
            let ref_name = normalize_ref(sub[1]);
            return Some((var_name, format!("{} + {}", ref_name, n)));
        }
    }

    // Pattern: plain integer "N"
    if let Ok(n) = rhs.parse::<i64>() {
        return Some((var_name, n.to_string()));
    }

    // Pattern: "each REF" (equality)
    if rhs.starts_with("each ") || rhs.contains("'s") {
        let ref_name = normalize_ref(rhs);
        return Some((var_name, ref_name));
    }

    None
}

/// Parse the question "How many X does Y have?" and return the target variable.
fn parse_question(text: &str) -> Option<String> {
    // "How many Pelican does Zoo B have?"
    // → target: find the entity that matches "Zoo B's Pelican" pattern
    let q = text.trim().trim_end_matches('?');
    let rest = q.strip_prefix("How many ")?;
    let parts: Vec<&str> = rest.splitn(2, " does ").collect();
    if parts.len() != 2 {
        return None;
    }
    let item = parts[0].trim();
    let entity = parts[1].strip_suffix(" have").unwrap_or(parts[1]).trim();
    // Construct the variable name: "entity_item"
    let var = format!(
        "{}_{}",
        entity.to_lowercase().replace(' ', "_"),
        item.to_lowercase().replace(' ', "_")
    );
    Some(var)
}

/// Parse and solve an iGSM problem in one call.
///
/// Takes the full problem text, extracts equations and question,
/// solves via topological evaluation, returns the answer.
///
/// Returns None if parsing fails or the target variable can't be solved.
#[pyfunction]
pub fn parse_and_solve_igsm(text: &str) -> Option<i64> {
    let mut equations = Vec::new();
    let mut target_var = None;

    for sentence in text.split('.') {
        let trimmed = sentence.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Try as statement
        if trimmed.starts_with("The number of") {
            if let Some(eq) = parse_statement(trimmed) {
                equations.push(eq);
                continue;
            }
        }

        // Try as question
        if trimmed.contains("How many") {
            if let Some(var) = parse_question(trimmed) {
                target_var = Some(var);
            }
        }
    }

    // Also check for question with ? (might be after last period)
    for part in text.split('?') {
        let trimmed = part.trim();
        if trimmed.contains("How many") && target_var.is_none() {
            if let Some(var) = parse_question(trimmed) {
                target_var = Some(var);
            }
        }
    }

    let target = target_var?;
    let solved = solve_equation_system(equations);

    // Direct match
    if let Some(&val) = solved.get(&target) {
        return Some(val);
    }

    // Fuzzy match: the target might be a partial match
    // e.g., target="zoo_b_pelican" matches solved key "zoo_b_pelican"
    // or target="region_population" matches "region_population"
    for (key, &val) in &solved {
        if key.ends_with(&target) || target.ends_with(key) {
            return Some(val);
        }
    }

    // Try matching by item name (last component)
    let target_parts: Vec<&str> = target.split('_').collect();
    if target_parts.len() >= 2 {
        let entity_part = &target_parts[..target_parts.len() - 1].join("_");
        let item_part = target_parts.last()?;
        for (key, &val) in &solved {
            if key.contains(entity_part) && key.contains(item_part) {
                return Some(val);
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_ref() {
        assert_eq!(normalize_ref("each Zoo A's Pelican"), "zoo_a_pelican");
        assert_eq!(normalize_ref("each Bread's Rye"), "bread_rye");
        assert_eq!(normalize_ref("each Save-A-Lot's Cakes"), "save_a_lot_cakes");
    }

    #[test]
    fn test_parse_direct_assignment() {
        let (var, expr) = parse_statement("The number of each Zoo A's Pelican equals 4").unwrap();
        assert_eq!(var, "zoo_a_pelican");
        assert_eq!(expr, "4");
    }

    #[test]
    fn test_parse_multiplication() {
        let (var, expr) = parse_statement(
            "The number of each Zoo B's Pelican equals 3 times as much as each Zoo A's Pelican"
        ).unwrap();
        assert_eq!(var, "zoo_b_pelican");
        assert_eq!(expr, "3 * zoo_a_pelican");
    }

    #[test]
    fn test_parse_sum() {
        let (var, expr) = parse_statement(
            "The number of each Store's Fruit equals the sum of each Store's Apple and each Store's Banana"
        ).unwrap();
        assert_eq!(var, "store_fruit");
        assert_eq!(expr, "store_apple + store_banana");
    }

    #[test]
    fn test_parse_difference() {
        let (var, expr) = parse_statement(
            "The number of each Lab A's Sample equals the difference of each Lab B's Sample and each Lab C's Sample"
        ).unwrap();
        assert_eq!(var, "lab_a_sample");
        assert_eq!(expr, "lab_b_sample - lab_c_sample");
    }

    #[test]
    fn test_parse_more_than() {
        let (var, expr) = parse_statement(
            "The number of each Farm B's Chicken equals 2 more than each Farm A's Chicken"
        ).unwrap();
        assert_eq!(var, "farm_b_chicken");
        assert_eq!(expr, "farm_a_chicken + 2");
    }

    #[test]
    fn test_parse_equality() {
        let (var, expr) = parse_statement(
            "The number of each Farm C's Chicken equals each Farm B's Chicken"
        ).unwrap();
        assert_eq!(var, "farm_c_chicken");
        assert_eq!(expr, "farm_b_chicken");
    }

    #[test]
    fn test_parse_question() {
        let var = parse_question("How many Pelican does Zoo B have").unwrap();
        assert_eq!(var, "zoo_b_pelican");
    }

    #[test]
    fn test_solve_simple() {
        let text = "The number of each Zoo A's Pelican equals 4. The number of each Zoo B's Pelican equals 3 times as much as each Zoo A's Pelican. How many Pelican does Zoo B have?";
        assert_eq!(parse_and_solve_igsm(text), Some(12));
    }

    #[test]
    fn test_solve_sum() {
        let text = "The number of each Store's Apple equals 7. The number of each Store's Banana equals 3 times as much as each Store's Apple. The number of each Store's Fruit equals the sum of each Store's Apple and each Store's Banana. How many Fruit does Store have?";
        assert_eq!(parse_and_solve_igsm(text), Some(28)); // 7 + 21 = 28
    }

    #[test]
    fn test_solve_chain() {
        let text = "The number of each Farm A's Chicken equals 10. The number of each Farm B's Chicken equals 2 more than each Farm A's Chicken. The number of each Farm C's Chicken equals each Farm B's Chicken. How many Chicken does Farm C have?";
        assert_eq!(parse_and_solve_igsm(text), Some(12));
    }

    #[test]
    fn test_solve_difference() {
        let text = "The number of each Lab B's Sample equals 20. The number of each Lab C's Sample equals 8. The number of each Lab A's Sample equals the difference of each Lab B's Sample and each Lab C's Sample. How many Sample does Lab A have?";
        assert_eq!(parse_and_solve_igsm(text), Some(12));
    }
}
