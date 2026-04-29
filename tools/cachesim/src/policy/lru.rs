//! LRU.
//!
//! v1 stub: HashMap<obj_id, recency>. Correct semantics; not the production
//! data structure (real impl will be intrusive doubly-linked list + arena).
//! Replaced before v1 ships.

use std::collections::HashMap;

use super::{Outcome, Policy};

pub struct Lru {
    capacity: usize,
    counter: u64,
    map: HashMap<u64, u64>,
}

impl Lru {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity,
            counter: 0,
            map: HashMap::with_capacity(capacity.saturating_add(1)),
        }
    }

    fn evict_if_full(&mut self) {
        if self.map.len() <= self.capacity {
            return;
        }
        if let Some(victim) = self
            .map
            .iter()
            .min_by_key(|(_, &t)| t)
            .map(|(&k, _)| k)
        {
            self.map.remove(&victim);
        }
    }
}

impl Policy for Lru {
    fn name(&self) -> &'static str {
        "lru"
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn access(&mut self, obj_id: u64) -> Outcome {
        self.counter += 1;
        let t = self.counter;
        match self.map.get_mut(&obj_id) {
            Some(slot) => {
                *slot = t;
                Outcome::Hit
            }
            None => {
                self.map.insert(obj_id, t);
                self.evict_if_full();
                Outcome::Miss
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_then_hit() {
        let mut c = Lru::with_capacity(2);
        assert_eq!(c.access(1), Outcome::Miss);
        assert_eq!(c.access(1), Outcome::Hit);
        assert_eq!(c.access(2), Outcome::Miss);
        assert_eq!(c.access(3), Outcome::Miss); // evicts the older of {1, 2}
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn lru_evicts_least_recent() {
        let mut c = Lru::with_capacity(2);
        c.access(1);
        c.access(2);
        c.access(1); // refresh 1; resident {1,2}, 1 newer
        c.access(3); // evicts 2 (older); resident {1,3}
        assert_eq!(c.access(1), Outcome::Hit); // 1 still resident
        assert_eq!(c.access(2), Outcome::Miss); // 2 was evicted
    }
}
