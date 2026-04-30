//! LRU. Backed by `DList<()>`: O(1) hit (`move_to_front`), O(1) miss
//! insert (`push_front`), O(1) eviction (`pop_back`).

use super::util::DList;
use super::{Outcome, Policy};

pub struct Lru {
    capacity: usize,
    list: DList<()>,
}

impl Lru {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity,
            list: DList::new(),
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
        self.list.len()
    }

    fn access(&mut self, x: u64) -> Outcome {
        if self.list.contains(x) {
            self.list.move_to_front(x);
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
    fn cold_then_hit() {
        let mut c = Lru::with_capacity(2);
        assert_eq!(c.access(1), Outcome::Miss);
        assert_eq!(c.access(1), Outcome::Hit);
        assert_eq!(c.access(2), Outcome::Miss);
        assert_eq!(c.access(3), Outcome::Miss);
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn lru_evicts_least_recent() {
        let mut c = Lru::with_capacity(2);
        c.access(1);
        c.access(2);
        c.access(1);
        c.access(3);
        assert_eq!(c.access(1), Outcome::Hit);
        assert_eq!(c.access(2), Outcome::Miss);
    }
}
