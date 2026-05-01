//! LFU (Least Frequently Used).
//!
//! Each resident object carries an access count. Hits increment the count;
//! evictions take the lowest-count object, tie-breaking on least-recent
//! insertion among that count's bucket.
//!
//! Implementation: O(1) hit/miss/evict via a doubly-linked list per
//! frequency bucket. Buckets are stored in a `BTreeMap<u64, FreqBucket>`
//! so the minimum frequency is the first map entry. The tie-breaker is
//! LRU within each bucket: a hit unlinks from old bucket, increments
//! freq, pushes to front of new bucket; eviction takes the BACK of the
//! min-freq bucket.

use std::collections::{BTreeMap, HashMap};

use super::{Outcome, Policy};

struct Node {
    obj: u64,
    freq: u64,
    prev: Option<usize>,
    next: Option<usize>,
}

struct FreqBucket {
    head: Option<usize>, // most-recent in this bucket
    tail: Option<usize>, // least-recent (eviction candidate)
    len: usize,
}

impl FreqBucket {
    fn new() -> Self {
        Self { head: None, tail: None, len: 0 }
    }
}

pub struct Lfu {
    capacity: usize,
    nodes: Vec<Node>,
    free: Vec<usize>,
    by_obj: HashMap<u64, usize>,
    buckets: BTreeMap<u64, FreqBucket>,
}

impl Lfu {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity,
            nodes: Vec::with_capacity(capacity),
            free: Vec::new(),
            by_obj: HashMap::with_capacity(capacity),
            buckets: BTreeMap::new(),
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

    /// Unlink node ix from its current frequency bucket.
    fn unlink(&mut self, ix: usize) {
        let freq = self.nodes[ix].freq;
        let prev = self.nodes[ix].prev;
        let next = self.nodes[ix].next;
        let bucket = self.buckets.get_mut(&freq).expect("bucket missing");
        match prev {
            Some(p) => self.nodes[p].next = next,
            None => bucket.head = next,
        }
        match next {
            Some(n) => self.nodes[n].prev = prev,
            None => bucket.tail = prev,
        }
        bucket.len -= 1;
        self.nodes[ix].prev = None;
        self.nodes[ix].next = None;
        if bucket.len == 0 {
            self.buckets.remove(&freq);
        }
    }

    /// Push node ix to the head (most-recent end) of its current bucket.
    fn push_head(&mut self, ix: usize) {
        let freq = self.nodes[ix].freq;
        let bucket = self.buckets.entry(freq).or_insert_with(FreqBucket::new);
        let old_head = bucket.head;
        self.nodes[ix].prev = None;
        self.nodes[ix].next = old_head;
        bucket.head = Some(ix);
        if let Some(h) = old_head {
            self.nodes[h].prev = Some(ix);
        }
        if bucket.tail.is_none() {
            bucket.tail = Some(ix);
        }
        bucket.len += 1;
    }

    /// Evict from the lowest-frequency bucket's tail.
    fn evict(&mut self) -> u64 {
        let (&min_freq, bucket) = self
            .buckets
            .iter_mut()
            .next()
            .expect("evict() called on empty cache");
        let victim = bucket.tail.expect("non-empty bucket missing tail");
        let _ = min_freq;
        let obj = self.nodes[victim].obj;
        self.unlink(victim);
        self.by_obj.remove(&obj);
        self.free.push(victim);
        obj
    }
}

impl Policy for Lfu {
    fn name(&self) -> &'static str {
        "lfu"
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn len(&self) -> usize {
        self.by_obj.len()
    }

    fn access(&mut self, obj: u64) -> Outcome {
        if let Some(&ix) = self.by_obj.get(&obj) {
            // Hit: increment freq, move to head of new bucket.
            self.unlink(ix);
            self.nodes[ix].freq += 1;
            self.push_head(ix);
            return Outcome::Hit;
        }
        // Miss: insert at freq=1.
        if self.by_obj.len() >= self.capacity {
            self.evict();
        }
        let ix = self.alloc(Node {
            obj,
            freq: 1,
            prev: None,
            next: None,
        });
        self.by_obj.insert(obj, ix);
        self.push_head(ix);
        Outcome::Miss
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_then_hit() {
        let mut c = Lfu::with_capacity(2);
        assert_eq!(c.access(1), Outcome::Miss);
        assert_eq!(c.access(1), Outcome::Hit);
        assert_eq!(c.access(2), Outcome::Miss);
        assert_eq!(c.access(3), Outcome::Miss);
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn lfu_evicts_least_frequent() {
        // 1 accessed twice, 2 accessed once → 3 evicts 2 (tie-break by LRU
        // within the freq=1 bucket: 2 is older than 3 once 3 enters; but
        // 3 is the new entry so 2 is the older single-access).
        let mut c = Lfu::with_capacity(2);
        c.access(1); // [1@1]
        c.access(1); // [1@2]
        c.access(2); // [1@2, 2@1]
        c.access(3); // 2 evicted (lowest freq, oldest in bucket): [1@2, 3@1]
        assert_eq!(c.access(1), Outcome::Hit);
        assert_eq!(c.access(3), Outcome::Hit);
        assert_eq!(c.access(2), Outcome::Miss);
    }

    #[test]
    fn lfu_tiebreaks_by_lru_within_bucket() {
        // LFU eviction picks the lowest-frequency bucket; within that
        // bucket the OLDEST entry (LRU tie-break) goes first.
        // Build state: cache=[1@2, 2@1] (1 was hit twice).
        let mut c = Lfu::with_capacity(2);
        c.access(1); // freq 1: [1]
        c.access(1); // freq 2: [1]
        c.access(2); // miss → freq 1: [2]; cache full, no eviction yet
        // Now miss 3: evict freq=1 bucket tail = 2 (only entry). Cache: [1@2, 3@1].
        c.access(3);
        assert_eq!(c.access(1), Outcome::Hit);  // 1 has highest freq, kept
        assert_eq!(c.access(3), Outcome::Hit);  // 3 still in freq=1 bucket
        assert_eq!(c.access(2), Outcome::Miss); // 2 was the LRU tie-break casualty
    }
}
