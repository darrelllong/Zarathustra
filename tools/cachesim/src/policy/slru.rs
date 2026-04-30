//! SLRU (Segmented LRU), Karedla, Love & Wherry, IEEE Computer 1994.
//!
//! Two LRU segments share one resident set:
//!   - Probationary (P): receives every cold miss.
//!   - Protected   (T): receives any object that hits while in P.
//!
//! Promotion: a hit in P moves the object to the MRU of T. If T was
//! already full, T's LRU is demoted to the MRU of P. The demoted item
//! takes the slot freed by the just-promoted x, so probationary's
//! cardinality is unchanged — no further eviction is needed.
//! Eviction: a miss with P full evicts P's LRU.
//!
//! Default split is 80% protected / 20% probationary, the value
//! recommended in the original SLRU paper. A tighter probationary helps
//! on workloads with rapid one-shot churn; a wider one helps when reuse
//! distance often exceeds 80% of the cache. `with_split` exposes the
//! ratio for sweeps.

use super::util::DList;
use super::{Outcome, Policy};

pub struct Slru {
    cap_protected: usize,
    cap_probationary: usize,
    protected: DList<()>,
    probationary: DList<()>,
}

impl Slru {
    /// Default 80/20 split per Karedla–Love–Wherry.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_split(capacity, 0.80)
    }

    /// `protected_fraction` ∈ (0, 1) determines |Protected| ÷ capacity.
    /// Each segment gets a floor of 1 so degenerate caches still work.
    pub fn with_split(capacity: usize, protected_fraction: f64) -> Self {
        assert!(
            protected_fraction > 0.0 && protected_fraction < 1.0,
            "protected_fraction must be in (0, 1), got {protected_fraction}"
        );
        let cap_protected = ((capacity as f64 * protected_fraction).round() as usize).max(1);
        let cap_probationary = capacity.saturating_sub(cap_protected).max(1);
        Self {
            cap_protected,
            cap_probationary,
            protected: DList::new(),
            probationary: DList::new(),
        }
    }
}

impl Policy for Slru {
    fn name(&self) -> &'static str {
        "slru"
    }

    fn capacity(&self) -> usize {
        self.cap_protected + self.cap_probationary
    }

    fn len(&self) -> usize {
        self.protected.len() + self.probationary.len()
    }

    fn access(&mut self, x: u64) -> Outcome {
        if self.protected.contains(x) {
            self.protected.move_to_front(x);
            return Outcome::Hit;
        }
        if self.probationary.contains(x) {
            self.probationary.remove(x);
            // Promote x to protected. If protected is full, demote its LRU
            // back to probationary (taking x's just-vacated slot, so no
            // eviction is needed in probationary).
            if self.protected.len() >= self.cap_protected {
                if let Some((demoted, _)) = self.protected.pop_back() {
                    self.probationary.push_front(demoted, ());
                }
            }
            self.protected.push_front(x, ());
            return Outcome::Hit;
        }
        // Cold miss → probationary; evict probationary's LRU if full.
        if self.probationary.len() >= self.cap_probationary {
            self.probationary.pop_back();
        }
        self.probationary.push_front(x, ());
        Outcome::Miss
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn miss_lands_in_probationary() {
        let mut s = Slru::with_capacity(10);
        s.access(1);
        assert!(s.probationary.contains(1));
        assert!(!s.protected.contains(1));
    }

    #[test]
    fn second_hit_promotes_to_protected() {
        let mut s = Slru::with_capacity(10);
        s.access(1);
        assert_eq!(s.access(1), Outcome::Hit);
        assert!(s.protected.contains(1));
        assert!(!s.probationary.contains(1));
    }

    #[test]
    fn protected_eviction_demotes_lru_to_probationary() {
        // 50/50 split keeps the test readable: cap=4 → 2 protected, 2 probationary.
        let mut s = Slru::with_split(4, 0.5);
        // Promote 1, 2 to protected.
        s.access(1);
        s.access(1);
        s.access(2);
        s.access(2);
        assert!(s.protected.contains(1) && s.protected.contains(2));
        // Now promote 3: protected was [2,1] (MRU first); LRU = 1, must demote to probationary.
        s.access(3);
        s.access(3);
        assert!(s.protected.contains(3));
        assert!(s.protected.contains(2));
        assert!(!s.protected.contains(1));
        assert!(s.probationary.contains(1), "1 should have been demoted to probationary");
    }

    #[test]
    fn protected_hit_refreshes_recency() {
        let mut s = Slru::with_split(4, 0.5);
        s.access(1);
        s.access(1); // 1 in protected
        s.access(2);
        s.access(2); // 2 in protected (MRU)
        s.access(1); // 1 → MRU of protected
        // Now promote 3, expect 2 (LRU of protected) demoted, not 1.
        s.access(3);
        s.access(3);
        assert!(s.protected.contains(1));
        assert!(s.protected.contains(3));
        assert!(s.probationary.contains(2));
    }

    #[test]
    fn capacity_strictly_respected() {
        let cap = 16;
        let mut s = Slru::with_capacity(cap);
        for x in 0u64..2_000 {
            s.access(x % 100);
            assert!(s.len() <= cap);
            assert!(s.protected.len() <= s.cap_protected);
            assert!(s.probationary.len() <= s.cap_probationary);
        }
    }
}
