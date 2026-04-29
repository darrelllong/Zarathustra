//! Mattson textbook example for LRU. Placeholder pending the real
//! single-pass stack-distance implementation; this just confirms the
//! basic LRU policy survives a known sequence.

use cachesim::policy::{make, Outcome, PolicyKind};

#[test]
fn lru_basic_sequence() {
    let mut c = make(PolicyKind::Lru, 3);
    let trace: &[u64] = &[1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5];
    let mut hits = 0;
    for &x in trace {
        if matches!(c.access(x), Outcome::Hit) {
            hits += 1;
        }
    }
    // Worked example: with capacity 3 the only hits are positions where the
    // object is still resident under LRU. Update once Mattson HRC validates.
    assert!(hits > 0, "LRU should produce at least one hit on this trace");
}
