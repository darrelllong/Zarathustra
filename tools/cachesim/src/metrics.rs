//! Metrics computed against an `Iterator<Item = anyhow::Result<Access>>`.
//!
//! v1 metric set (matches `llgan/long_rollout_eval.py` sidecar):
//!   - per-cache-size miss ratio (HRC)
//!   - reuse-access rate
//!   - stack-distance median / p90  (LRU only; via Mattson)
//!   - footprint (unique objects seen)
//!   - hrc_mae_vs_real (computed at report time, not here)
//!
//! Bodies land alongside the LRU/Mattson implementation.

use crate::policy::{Outcome, Policy};
use crate::trace::Access;

#[derive(Debug, Clone)]
pub struct PerSizeResult {
    pub size: usize,
    pub miss_ratio: f64,
}

#[derive(Debug, Clone)]
pub struct RunResult {
    pub policy: String,
    pub n_accesses: u64,
    pub reuse_access_rate: f64,
    pub stack_distance_median: Option<u64>,
    pub stack_distance_p90: Option<u64>,
    pub footprint: u64,
    pub per_cache_size: Vec<PerSizeResult>,
}

pub fn run_single<I>(mut policy: Box<dyn Policy>, iter: I) -> anyhow::Result<RunResult>
where
    I: Iterator<Item = anyhow::Result<Access>>,
{
    let name = policy.name().to_string();
    let mut hits: u64 = 0;
    let mut total: u64 = 0;
    for rec in iter {
        let a = rec?;
        if matches!(policy.access(a.obj_id), Outcome::Hit) {
            hits += 1;
        }
        total += 1;
    }
    let miss_ratio = if total == 0 {
        0.0
    } else {
        1.0 - (hits as f64 / total as f64)
    };
    Ok(RunResult {
        policy: name.clone(),
        n_accesses: total,
        reuse_access_rate: 0.0,
        stack_distance_median: None,
        stack_distance_p90: None,
        footprint: 0,
        per_cache_size: vec![PerSizeResult {
            size: policy.capacity(),
            miss_ratio,
        }],
    })
}
