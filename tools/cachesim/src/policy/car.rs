//! CAR (Clock with Adaptive Replacement), Bansal & Modha, FAST 2004.
//!
//! ARC's hit-rate guarantees with O(1) operations: T1 and T2 are CLOCK
//! lists (doubly-linked, one reference bit per resident); B1 and B2 are
//! LRU ghost lists. The adaptive `p` and four-case access logic mirror
//! ARC, but promotion is lazy — a hit just sets the reference bit.
//!
//! Naming follows the paper: front of each clock list is the head (next
//! page the hand will scan), back is the tail (most recently inserted).
//! Invariants maintained at end of every access:
//!   - 0 ≤ |T1| + |T2|             ≤ c
//!   - 0 ≤ |T1| + |B1|             ≤ c
//!   - 0 ≤ |T1| + |T2| + |B1| + |B2| ≤ 2c

use super::util::DList;
use super::{Outcome, Policy};

pub struct Car {
    capacity: usize,
    p: usize,
    t1: DList<bool>, // resident clock; payload = reference bit
    t2: DList<bool>,
    b1: DList<()>, // ghosts, LRU-ordered
    b2: DList<()>,
}

impl Car {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity,
            p: 0,
            t1: DList::new(),
            t2: DList::new(),
            b1: DList::new(),
            b2: DList::new(),
        }
    }

    /// REPLACE() from Bansal–Modha Fig. 2. Walks the T1/T2 clock heads
    /// until one is found with reference bit 0; that page is demoted to
    /// the corresponding ghost list. Reference-bit-1 pages get the bit
    /// cleared and either migrate (T1→T2) or rotate (T2→T2).
    fn replace(&mut self) {
        loop {
            if self.t1.len() >= self.p.max(1) {
                let (x, bit) = self.t1.pop_front().expect("t1 non-empty under |T1|≥max(1,p)");
                if bit {
                    self.t2.push_back(x, false);
                } else {
                    self.b1.push_front(x, ());
                    return;
                }
            } else {
                let Some((x, bit)) = self.t2.pop_front() else {
                    return; // both clocks empty (shouldn't happen if cache full)
                };
                if bit {
                    self.t2.push_back(x, false);
                } else {
                    self.b2.push_front(x, ());
                    return;
                }
            }
        }
    }
}

impl Policy for Car {
    fn name(&self) -> &'static str {
        "car"
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn len(&self) -> usize {
        self.t1.len() + self.t2.len()
    }

    fn access(&mut self, x: u64) -> Outcome {
        let c = self.capacity;

        // Case I: hit in T1 ∪ T2 → set reference bit; no list movement.
        if let Some(bit) = self.t1.payload_mut(x) {
            *bit = true;
            return Outcome::Hit;
        }
        if let Some(bit) = self.t2.payload_mut(x) {
            *bit = true;
            return Outcome::Hit;
        }

        // Miss path. If the resident set is full, free a slot then enforce
        // the ghost invariants for the slot the new entry will occupy.
        if self.t1.len() + self.t2.len() == c {
            self.replace();
            // After REPLACE: |T1|+|T2| = c-1. About to add x as a fresh
            // resident (case IV) or recycle a ghost into T2 (cases II/III).
            // Ghost-list trim only applies when x is genuinely cold.
            let cold = !self.b1.contains(x) && !self.b2.contains(x);
            if cold {
                if self.t1.len() + self.b1.len() == c {
                    self.b1.pop_back();
                } else if self.t1.len() + self.t2.len() + self.b1.len() + self.b2.len() == 2 * c {
                    self.b2.pop_back();
                }
            }
        }

        // Cases II, III, IV.
        if self.b1.contains(x) {
            let b1 = self.b1.len().max(1);
            let delta = (self.b2.len() / b1).max(1);
            self.p = (self.p + delta).min(c);
            self.b1.remove(x);
            self.t2.push_back(x, false);
        } else if self.b2.contains(x) {
            let b2 = self.b2.len().max(1);
            let delta = (self.b1.len() / b2).max(1);
            self.p = self.p.saturating_sub(delta);
            self.b2.remove(x);
            self.t2.push_back(x, false);
        } else {
            self.t1.push_back(x, false);
        }
        Outcome::Miss
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_then_hit_sets_bit() {
        let mut c = Car::with_capacity(4);
        assert_eq!(c.access(1), Outcome::Miss);
        assert_eq!(c.access(1), Outcome::Hit);
        assert_eq!(c.t1.payload_mut(1).copied(), Some(true));
    }

    #[test]
    fn invariants_hold_under_random_workload() {
        let c = 8;
        let mut car = Car::with_capacity(c);
        for x in 0u64..2000 {
            // Mix of cold and warm accesses.
            car.access(x);
            car.access(x % 11);
        }
        assert!(car.len() <= c, "|T1|+|T2| = {} > c = {}", car.len(), c);
        assert!(car.t1.len() + car.b1.len() <= c);
        assert!(car.t1.len() + car.t2.len() + car.b1.len() + car.b2.len() <= 2 * c);
    }

    #[test]
    fn hits_repeated_working_set() {
        let mut c = Car::with_capacity(3);
        let trace: Vec<u64> = (0..30).map(|i| (i % 3) as u64).collect();
        let mut hits = 0;
        for x in trace {
            if matches!(c.access(x), Outcome::Hit) {
                hits += 1;
            }
        }
        assert!(hits >= 24, "CAR hits on tight working set, got {hits}");
    }

    #[test]
    fn b1_hit_grows_p() {
        // CAR's ghost-trim runs whenever |T1|+|B1| reaches c on a cold
        // miss, so a tiny cache deletes its only B1 entry as fast as
        // REPLACE creates one. We need T2 to grow first (via pages whose
        // ref bits are set on a re-touch), which shrinks |T1| and lets
        // |B1| accumulate. The sequence below leaves B1 = {4, 3} before
        // the B1 hit on 3.
        let mut car = Car::with_capacity(4);
        for x in [1u64, 2, 1, 3, 2, 4, 5, 6] {
            car.access(x);
        }
        assert!(car.b1.contains(3), "test setup: 3 must be in B1");
        let p_before = car.p;
        assert_eq!(car.access(3), Outcome::Miss); // B1 hit (still a miss).
        assert!(car.p > p_before, "p should grow on B1 hit (was {p_before}, now {})", car.p);
        assert!(car.t2.contains(3));
        assert!(!car.b1.contains(3));
    }
}
