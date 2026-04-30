//! SIEVE (Zhang, Yang, Yue, Vigfusson — NSDI 2024).
//!
//! FIFO-shaped queue with one "visited" bit per resident object plus a
//! single "hand" pointer. Hits are lazy (set visited=1, no reordering).
//! Eviction sweeps the hand from tail toward head: visited=1 → clear and
//! advance; visited=0 → evict here. New objects are inserted at the head
//! with visited=0 and are never moved on hit.
//!
//! Intuition: a CLOCK whose hand walks the wrong way and survives over
//! many evictions, giving scan-resistance close to LRU/ARC at FIFO cost.

use std::collections::HashMap;

use super::{Outcome, Policy};

/// Doubly-linked list arena. Index 0 is reserved as a sentinel-free invalid
/// slot? — no, we use `Option<usize>` for prev/next so any index is valid.
struct Node {
    obj: u64,
    visited: bool,
    prev: Option<usize>, // toward head (newer)
    next: Option<usize>, // toward tail (older)
}

pub struct Sieve {
    capacity: usize,
    nodes: Vec<Node>,
    free: Vec<usize>,
    by_obj: HashMap<u64, usize>,
    head: Option<usize>, // newest
    tail: Option<usize>, // oldest
    hand: Option<usize>, // eviction pointer; None ≡ start at tail
}

impl Sieve {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity,
            nodes: Vec::with_capacity(capacity),
            free: Vec::new(),
            by_obj: HashMap::with_capacity(capacity),
            head: None,
            tail: None,
            hand: None,
        }
    }

    fn alloc(&mut self, n: Node) -> usize {
        if let Some(slot) = self.free.pop() {
            self.nodes[slot] = n;
            slot
        } else {
            self.nodes.push(n);
            self.nodes.len() - 1
        }
    }

    fn unlink(&mut self, ix: usize) {
        let prev = self.nodes[ix].prev;
        let next = self.nodes[ix].next;
        match prev {
            Some(p) => self.nodes[p].next = next,
            None => self.head = next,
        }
        match next {
            Some(n) => self.nodes[n].prev = prev,
            None => self.tail = prev,
        }
    }

    fn push_head(&mut self, ix: usize) {
        self.nodes[ix].prev = None;
        self.nodes[ix].next = self.head;
        if let Some(h) = self.head {
            self.nodes[h].prev = Some(ix);
        }
        self.head = Some(ix);
        if self.tail.is_none() {
            self.tail = Some(ix);
        }
    }

    /// Hand sweep: starting at `hand` (or tail if None), walk toward the
    /// head, clearing visited bits. Stop at the first visited=0 node and
    /// evict it; the hand parks at its predecessor (toward the head).
    fn evict_one(&mut self) {
        let mut cur = self.hand.or(self.tail);
        loop {
            let ix = cur.expect("evict_one called on empty cache");
            if self.nodes[ix].visited {
                self.nodes[ix].visited = false;
                cur = self.nodes[ix].prev.or(self.tail); // wrap to tail when off the head
            } else {
                let next_hand = self.nodes[ix].prev.or(self.tail);
                let obj = self.nodes[ix].obj;
                self.unlink(ix);
                self.by_obj.remove(&obj);
                self.free.push(ix);
                self.hand = if next_hand == Some(ix) { None } else { next_hand };
                return;
            }
        }
    }
}

impl Policy for Sieve {
    fn name(&self) -> &'static str {
        "sieve"
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn len(&self) -> usize {
        self.by_obj.len()
    }

    fn access(&mut self, x: u64) -> Outcome {
        if let Some(&ix) = self.by_obj.get(&x) {
            self.nodes[ix].visited = true;
            return Outcome::Hit;
        }
        if self.by_obj.len() >= self.capacity {
            self.evict_one();
        }
        let ix = self.alloc(Node {
            obj: x,
            visited: false,
            prev: None,
            next: None,
        });
        self.push_head(ix);
        self.by_obj.insert(x, ix);
        Outcome::Miss
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_miss_then_hit_sets_visited() {
        let mut s = Sieve::with_capacity(4);
        assert_eq!(s.access(1), Outcome::Miss);
        let ix = *s.by_obj.get(&1).unwrap();
        assert!(!s.nodes[ix].visited, "fresh insert visited=0");
        assert_eq!(s.access(1), Outcome::Hit);
        let ix = *s.by_obj.get(&1).unwrap();
        assert!(s.nodes[ix].visited, "hit sets visited=1");
    }

    #[test]
    fn unvisited_evicted_first() {
        // Cap 3. Insert 1,2,3 (none visited). Touch 2 → visited. Insert 4:
        // hand sweeps from tail (1, oldest, visited=0) → evict 1.
        let mut s = Sieve::with_capacity(3);
        s.access(1);
        s.access(2);
        s.access(3);
        s.access(2); // visited bit on 2
        s.access(4); // should evict 1
        assert!(s.by_obj.contains_key(&2));
        assert!(s.by_obj.contains_key(&3));
        assert!(s.by_obj.contains_key(&4));
        assert!(!s.by_obj.contains_key(&1));
    }

    #[test]
    fn visited_bit_clears_during_sweep() {
        // Cap 2. Insert 1,2. Touch both (both visited=1). Insert 3: hand
        // clears 1 then 2, wraps, evicts 1 (now visited=0).
        let mut s = Sieve::with_capacity(2);
        s.access(1);
        s.access(2);
        s.access(1);
        s.access(2);
        s.access(3);
        assert_eq!(s.by_obj.len(), 2);
        assert!(s.by_obj.contains_key(&3));
    }

    #[test]
    fn never_exceeds_capacity() {
        let mut s = Sieve::with_capacity(8);
        for x in 0u64..1000 {
            s.access(x);
            assert!(s.len() <= 8);
        }
    }

    #[test]
    fn hand_persists_across_evictions() {
        // After evicting the tail, the hand parks at the position that
        // was the tail's prev (toward head). Subsequent eviction starts
        // there, not back at the new tail.
        let mut s = Sieve::with_capacity(4);
        for x in 1u64..=4 {
            s.access(x);
        }
        // List: [4, 3, 2, 1], head=4, tail=1, all bits 0.
        s.access(5); // miss → evict_one. Hand=tail=1, visited=0, evict 1.
        let idx_2 = *s.by_obj.get(&2).expect("2 should still be resident");
        assert_eq!(s.hand, Some(idx_2), "hand should park at 2 after evicting 1");
        // Next eviction continues from 2.
        s.access(6);
        assert!(!s.by_obj.contains_key(&2));
        assert!(s.by_obj.contains_key(&3) && s.by_obj.contains_key(&4));
        assert!(s.by_obj.contains_key(&5) && s.by_obj.contains_key(&6));
    }

    #[test]
    fn wraps_to_tail_when_all_bits_set() {
        // All-1 bits → hand walks tail→head clearing each, wraps from
        // head's prev=None to tail, finds the now-cleared first-scanned
        // node, evicts it. Net effect: oldest goes (FIFO fallback).
        let mut s = Sieve::with_capacity(3);
        for x in 1u64..=3 {
            s.access(x);
        }
        for x in 1u64..=3 {
            s.access(x); // sets all visited bits
        }
        s.access(4); // forces eviction sweep
        assert!(!s.by_obj.contains_key(&1), "FIFO oldest should be evicted under all-1 bits");
        assert!(s.by_obj.contains_key(&2));
        assert!(s.by_obj.contains_key(&3));
        assert!(s.by_obj.contains_key(&4));
    }

    #[test]
    fn capacity_one_terminates() {
        // Degenerate cache: every miss must evict the single resident,
        // and the eviction sweep must terminate.
        let mut s = Sieve::with_capacity(1);
        for x in 0u64..50 {
            s.access(x);
            assert_eq!(s.len(), 1);
        }
    }
}
