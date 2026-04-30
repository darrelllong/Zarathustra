//! ARC (Adaptive Replacement Cache), Megiddo & Modha, FAST 2003.
//!
//! Textbook transcription of the four-case algorithm with the standard
//! REPLACE subroutine. Backed by `DList<()>` for each of T1/T2/B1/B2, so
//! all list operations (push/pop/remove/move) are O(1).

use super::util::DList;
use super::{Outcome, Policy};

pub struct Arc {
    capacity: usize,
    p: usize, // adaptive target for |T1|, in [0, capacity]
    t1: DList<()>,
    t2: DList<()>,
    b1: DList<()>,
    b2: DList<()>,
}

impl Arc {
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

    /// REPLACE(key, p) from Fig. 4 of Megiddo–Modha. `key_in_b2` flips the
    /// tie-break when |T1| == p toward evicting from T1.
    fn replace(&mut self, key_in_b2: bool) {
        let t1_len = self.t1.len();
        if t1_len > 0 && (t1_len > self.p || (key_in_b2 && t1_len == self.p)) {
            if let Some((victim, _)) = self.t1.pop_back() {
                self.b1.push_front(victim, ());
            }
        } else if let Some((victim, _)) = self.t2.pop_back() {
            self.b2.push_front(victim, ());
        }
    }
}

impl Policy for Arc {
    fn name(&self) -> &'static str {
        "arc"
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn len(&self) -> usize {
        self.t1.len() + self.t2.len()
    }

    fn access(&mut self, x: u64) -> Outcome {
        let c = self.capacity;

        // Case I: x ∈ T1 ∪ T2 (cache hit) → move to MRU of T2.
        if self.t1.contains(x) {
            self.t1.remove(x);
            self.t2.push_front(x, ());
            return Outcome::Hit;
        }
        if self.t2.contains(x) {
            self.t2.move_to_front(x);
            return Outcome::Hit;
        }

        // Case II: x ∈ B1 → grow p, REPLACE, move x to MRU of T2.
        if self.b1.contains(x) {
            let b1 = self.b1.len().max(1);
            let delta = (self.b2.len() / b1).max(1);
            self.p = (self.p + delta).min(c);
            self.replace(false);
            self.b1.remove(x);
            self.t2.push_front(x, ());
            return Outcome::Miss;
        }

        // Case III: x ∈ B2 → shrink p, REPLACE, move x to MRU of T2.
        if self.b2.contains(x) {
            let b2 = self.b2.len().max(1);
            let delta = (self.b1.len() / b2).max(1);
            self.p = self.p.saturating_sub(delta);
            self.replace(true);
            self.b2.remove(x);
            self.t2.push_front(x, ());
            return Outcome::Miss;
        }

        // Case IV: cold miss (x ∉ T1 ∪ T2 ∪ B1 ∪ B2).
        let l1 = self.t1.len() + self.b1.len();
        let l2 = self.t2.len() + self.b2.len();
        if l1 == c {
            if self.t1.len() < c {
                self.b1.pop_back();
                self.replace(false);
            } else {
                // |T1| = c, B1 empty: drop LRU of T1 directly.
                self.t1.pop_back();
            }
        } else if l1 < c && l1 + l2 >= c {
            if l1 + l2 == 2 * c {
                self.b2.pop_back();
            }
            self.replace(false);
        }
        self.t1.push_front(x, ());
        Outcome::Miss
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_then_hit_promotes_to_t2() {
        let mut a = Arc::with_capacity(4);
        assert_eq!(a.access(1), Outcome::Miss);
        assert!(a.t1.contains(1) && !a.t2.contains(1));
        assert_eq!(a.access(1), Outcome::Hit);
        assert!(!a.t1.contains(1) && a.t2.contains(1));
    }

    #[test]
    fn b1_hit_grows_p() {
        let mut a = Arc::with_capacity(2);
        a.access(1); // T1=[1]
        a.access(1); // hit → T2=[1], T1=[]
        a.access(2); // T1=[2], T2=[1]
        a.access(3); // cold; l1=1, l2=1, l1+l2=2=c → REPLACE then insert.
                     // |T1|=1>p=0 → evict LRU of T1 (=2) into B1. T1=[3], B1=[2].
        assert!(a.b1.contains(2));
        let p_before = a.p;
        a.access(2); // B1 hit → grow p, move 2 to T2.
        assert!(a.p > p_before);
        assert!(a.t2.contains(2));
        assert!(!a.b1.contains(2));
    }

    #[test]
    fn b2_hit_shrinks_p() {
        let mut a = Arc::with_capacity(2);
        a.access(1);
        a.access(1); // T2=[1]
        a.access(2);
        a.access(2); // T2=[2,1]
        a.access(3); // l1=0, l2=2, l1+l2=2=c → REPLACE.
                     // |T1|=0 → evict LRU of T2 (=1) to B2. T1=[3], T2=[2], B2=[1].
        assert!(a.b2.contains(1));
        a.access(4); // l1=1, l2=2 → REPLACE; |T1|=1>0=p → evict 3 to B1. T1=[4], B1=[3].
        a.access(3); // B1 hit → p grows.
        let p_before = a.p;
        assert!(p_before > 0);
        assert!(a.b2.contains(1));
        a.access(1); // B2 hit → p shrinks.
        assert!(a.p < p_before);
        assert!(a.t2.contains(1));
    }

    #[test]
    fn never_exceeds_capacity() {
        let mut a = Arc::with_capacity(8);
        for x in 0u64..1000 {
            a.access(x);
            assert!(a.len() <= 8);
            assert!(a.t1.len() + a.b1.len() <= 8);
            assert!(a.t1.len() + a.t2.len() + a.b1.len() + a.b2.len() <= 16);
        }
    }
}
