//! cachesim — LRU / ARC cache simulator for real and synthetic I/O traces.
//!
//! v1 scope:
//!   - Policies: LRU + ARC (classic Megiddo–Modha 2003).
//!   - Trace formats: oracleGeneral `.zst` (real) and synthetic CSV from
//!     `llgan/generate.py` and `altgan/generate.py`.
//!   - Output JSON schema: byte-identical to `llgan/long_rollout_eval.py`
//!     sidecar so existing reports keep working unmodified.
//!
//! Stores only `obj_id` set membership and recency metadata — no payload.

pub mod metrics;
pub mod policy;
pub mod report;
pub mod trace;

pub use policy::{Policy, PolicyKind};
pub use trace::{Access, Op, Trace};
