//! Public-API smoke tests for ARC. Internal-state assertions live in the
//! `#[cfg(test)] mod tests` block inside `src/policy/arc.rs`.

use cachesim::policy::{make, Outcome, PolicyKind};

fn run(policy: PolicyKind, cap: usize, trace: &[u64]) -> usize {
    let mut c = make(policy, cap);
    let mut hits = 0usize;
    for &x in trace {
        if matches!(c.access(x), Outcome::Hit) {
            hits += 1;
        }
    }
    hits
}

#[test]
fn arc_hits_on_repeats() {
    // 1,2,3,1,2,3,... cycles of size 3 in capacity-3 cache → ARC keeps them.
    let trace: Vec<u64> = (0..30).map(|i| (i % 3) as u64 + 1).collect();
    let hits = run(PolicyKind::Arc, 3, &trace);
    assert!(hits >= 24, "ARC should hit on repeated tight working set, got {hits}");
}

#[test]
fn arc_matches_lru_on_pure_cold_stream() {
    // No repeats: every access is a compulsory miss for both policies.
    let trace: Vec<u64> = (0..1000u64).collect();
    let lru_hits = run(PolicyKind::Lru, 64, &trace);
    let arc_hits = run(PolicyKind::Arc, 64, &trace);
    assert_eq!(lru_hits, 0);
    assert_eq!(arc_hits, 0);
}

#[test]
fn arc_scan_resistance_vs_lru() {
    // Working set of size W repeatedly accessed, interleaved with a longer
    // scan that pollutes LRU. ARC's frequent list T2 should retain the
    // working set; LRU thrashes.
    let cap = 16usize;
    let working: Vec<u64> = (0..8u64).collect();
    let scan: Vec<u64> = (1000..1000 + 64u64).collect();
    let mut trace = Vec::new();
    // Warm both caches on the working set so ARC promotes it to T2.
    for _ in 0..4 {
        trace.extend_from_slice(&working);
    }
    // Interleave scans with working-set re-access.
    for _ in 0..4 {
        trace.extend_from_slice(&scan);
        trace.extend_from_slice(&working);
    }
    let lru_hits = run(PolicyKind::Lru, cap, &trace);
    let arc_hits = run(PolicyKind::Arc, cap, &trace);
    assert!(
        arc_hits >= lru_hits,
        "ARC ({arc_hits}) should not be worse than LRU ({lru_hits}) on a scan+working-set trace"
    );
}
