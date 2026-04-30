//! FIFO (First-In, First-Out). Insert at the head, evict the tail. No
//! hit-side reordering — a hit is silent. Baseline against which scan-
//! resistant policies (LRU, ARC, SIEVE) earn their complexity.

use super::util::DList;
use super::{Outcome, Policy};

pub struct Fifo {
    capacity: usize,
    list: DList<()>,
}

impl Fifo {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity,
            list: DList::new(),
        }
    }
}

impl Policy for Fifo {
    fn name(&self) -> &'static str {
        "fifo"
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn len(&self) -> usize {
        self.list.len()
    }

    fn access(&mut self, x: u64) -> Outcome {
        if self.list.contains(x) {
            return Outcome::Hit;
        }
        if self.list.len() >= self.capacity {
            self.list.pop_back();
        }
        self.list.push_front(x, ());
        Outcome::Miss
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_does_not_refresh_position() {
        // The defining FIFO property: hits don't reorder. Item 1 is the
        // oldest after access(1, 2); a subsequent hit on 1 must not save
        // it from eviction.
        let mut c = Fifo::with_capacity(2);
        assert_eq!(c.access(1), Outcome::Miss);
        assert_eq!(c.access(2), Outcome::Miss);
        assert_eq!(c.access(1), Outcome::Hit);
        assert_eq!(c.access(3), Outcome::Miss); // evicts 1 (oldest), keeps 2
        assert_eq!(c.access(2), Outcome::Hit);
        assert_eq!(c.access(1), Outcome::Miss);
    }

    #[test]
    fn capacity_strictly_respected() {
        let cap = 16;
        let mut c = Fifo::with_capacity(cap);
        for x in 0u64..1_000 {
            c.access(x);
            assert!(c.len() <= cap);
        }
    }
}
