//! ModelRegistry — manages ModelCards and selects best models per CognitiveSystem.

use pyo3::prelude::*;
use std::collections::HashMap;

use super::model_card::{CognitiveSystem, ModelCard};

#[pyclass]
#[derive(Debug, Clone)]
pub struct ModelRegistry {
    cards: HashMap<String, ModelCard>,
}

#[pymethods]
impl ModelRegistry {
    #[staticmethod]
    pub fn from_toml_file(path: &str) -> PyResult<Self> {
        let cards_vec = ModelCard::load_from_file(path)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let cards = cards_vec.into_iter().map(|c| (c.id.clone(), c)).collect();
        Ok(Self { cards })
    }

    pub fn len(&self) -> usize {
        self.cards.len()
    }

    pub fn get(&self, id: &str) -> Option<ModelCard> {
        self.cards.get(id).cloned()
    }

    pub fn register(&mut self, card: ModelCard) {
        self.cards.insert(card.id.clone(), card);
    }

    pub fn unregister(&mut self, id: &str) {
        self.cards.remove(id);
    }

    /// Return all cards sorted by affinity for the given system (descending).
    pub fn select_for_system(&self, system: CognitiveSystem) -> Vec<ModelCard> {
        let mut candidates: Vec<_> = self.cards.values().cloned().collect();
        candidates.sort_by(|a, b| {
            b.affinity_for(system)
                .partial_cmp(&a.affinity_for(system))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates
    }

    pub fn list_ids(&self) -> Vec<String> {
        self.cards.keys().cloned().collect()
    }

    fn __repr__(&self) -> String {
        format!("ModelRegistry(models={})", self.cards.len())
    }

    fn __len__(&self) -> usize {
        self.cards.len()
    }
}

impl ModelRegistry {
    /// Load from a TOML string (for tests and embedding).
    pub fn from_toml_str(toml_str: &str) -> Result<Self, toml::de::Error> {
        let cards_vec = ModelCard::parse_toml(toml_str)?;
        let cards = cards_vec.into_iter().map(|c| (c.id.clone(), c)).collect();
        Ok(Self { cards })
    }
}
