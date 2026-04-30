//! Cross-policy correctness properties. These hold regardless of policy
//! and act as a tripwire if any implementation drifts from the canonical
//! semantics.

use cachesim::policy::{make, Outcome, PolicyKind};

const ALL: [PolicyKind; 6] = [
    PolicyKind::Fifo,
    PolicyKind::Lru,
    PolicyKind::Slru,
    PolicyKind::Arc,
    PolicyKind::Car,
    PolicyKind::Sieve,
];

fn count_hits(kind: PolicyKind, cap: usize, trace: &[u64]) -> usize {
    let mut c = make(kind, cap);
    trace
        .iter()
        .filter(|&&x| matches!(c.access(x), Outcome::Hit))
        .count()
}

#[test]
fn cold_only_trace_is_all_misses() {
    // No repeats → no policy can ever hit.
    let trace: Vec<u64> = (0..1_000).collect();
    for kind in ALL {
        assert_eq!(
            count_hits(kind, 64, &trace),
            0,
            "{:?} reported a hit on a unique-keys trace",
            kind
        );
    }
}

#[test]
fn capacity_strictly_respected_under_mixed_workload() {
    let cap = 32usize;
    let mut trace: Vec<u64> = Vec::new();
    for round in 0..20u64 {
        for x in 0..16u64 {
            trace.push(x);
        }
        for x in (1000 + round * 100)..(1000 + round * 100 + 50) {
            trace.push(x);
        }
        for x in 0..16u64 {
            trace.push(x);
        }
    }
    for kind in ALL {
        let mut c = make(kind, cap);
        for &x in &trace {
            c.access(x);
            assert!(
                c.len() <= cap,
                "{:?} resident set {} exceeded cap {}",
                kind,
                c.len(),
                cap
            );
        }
    }
}

#[test]
fn working_set_within_cap_eventually_all_hits() {
    // Working set of 10, cap = 50. Precondition for SLRU at the default
    // 80/20 split: working_set ≤ cap_probationary = cap / 5 = 10. ✓
    let working_set: Vec<u64> = (0..10).collect();
    let cap = 50;
    for kind in ALL {
        let mut c = make(kind, cap);
        for _ in 0..200 {
            for &x in &working_set {
                c.access(x);
            }
        }
        // After warmup, every working-set element must hit.
        let final_hits: usize = working_set
            .iter()
            .filter(|&&x| matches!(c.access(x), Outcome::Hit))
            .count();
        assert_eq!(
            final_hits,
            working_set.len(),
            "{:?} retained only {}/{} of a working set within cap",
            kind,
            final_hits,
            working_set.len()
        );
    }
}

#[test]
fn cap_ge_universe_misses_only_first_touch() {
    // 5 unique keys × 3 passes; cap = 50 (≥ everyone's effective insert
    // segment, including SLRU's default 20% probationary = 10 ≥ 5).
    // Total accesses = 15, expected hits = 10 (passes 2 and 3).
    let unique = 5u64;
    let trace: Vec<u64> = (0..unique)
        .chain(0..unique)
        .chain(0..unique)
        .collect();
    for kind in ALL {
        let hits = count_hits(kind, 50, &trace);
        assert_eq!(
            hits,
            (2 * unique) as usize,
            "{:?} got {} hits, expected {}",
            kind,
            hits,
            2 * unique
        );
    }
}

#[test]
fn lru_mattson_textbook_two_hits() {
    // Mattson 1970 worked example, cap=3, trace 1,2,3,4,1,2,5,1,2,3,4,5.
    // LRU yields exactly 2 hits (positions 8 and 9: 1 and 2).
    let trace: &[u64] = &[1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5];
    assert_eq!(count_hits(PolicyKind::Lru, 3, trace), 2);
}

#[test]
fn fifo_mattson_textbook_three_hits() {
    // Same trace, FIFO does not refresh on hit, so 1, 2 (positions 8, 9)
    // and 5 (position 12) stay long enough to hit. Three hits total.
    let trace: &[u64] = &[1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5];
    assert_eq!(count_hits(PolicyKind::Fifo, 3, trace), 3);
}
