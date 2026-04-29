//! JSON report — schema mirrors `llgan/long_rollout_eval.py` sidecar so existing
//! `RESULTS.md` tables consume cachesim output unchanged.

use serde::Serialize;

use crate::metrics::{PerSizeResult, RunResult};

#[derive(Debug, Clone, Serialize)]
pub struct PerSizeJson {
    pub size: usize,
    pub miss_ratio: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PolicyReport {
    pub policy: String,
    pub hrc_mae_vs_real: Option<f64>,
    pub reuse_access_rate: f64,
    pub stack_distance_median: Option<u64>,
    pub stack_distance_p90: Option<u64>,
    pub footprint_mean_per_stream: u64,
    pub n_accesses: u64,
    pub per_cache_size: Vec<PerSizeJson>,
}

impl From<&PerSizeResult> for PerSizeJson {
    fn from(p: &PerSizeResult) -> Self {
        Self {
            size: p.size,
            miss_ratio: p.miss_ratio,
        }
    }
}

pub fn build(run: &RunResult, hrc_mae_vs_real: Option<f64>) -> PolicyReport {
    PolicyReport {
        policy: run.policy.clone(),
        hrc_mae_vs_real,
        reuse_access_rate: run.reuse_access_rate,
        stack_distance_median: run.stack_distance_median,
        stack_distance_p90: run.stack_distance_p90,
        footprint_mean_per_stream: run.footprint,
        n_accesses: run.n_accesses,
        per_cache_size: run.per_cache_size.iter().map(PerSizeJson::from).collect(),
    }
}
