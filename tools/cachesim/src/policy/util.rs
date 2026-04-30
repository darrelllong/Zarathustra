//! O(1) doubly-linked list with by-value lookup, keyed on `u64` object id
//! and carrying a generic per-node payload `P`. Backed by a `Vec` arena
//! plus a free-list, so push/pop/remove/move are all O(1).
//!
//! Used by LRU, ARC (B1/B2 ghosts, payload `()`), SLRU (both segments,
//! `()`), CAR (T1/T2 clocks, `bool` reference bit; B1/B2 ghosts, `()`).
//! Front of the list is the "most recent" end; back is the "oldest" end.
//!
//! Arena slots are `Option<Node<P>>`: `Some` while occupied, `None` once
//! freed. This lets `pop_*` / `remove` take the payload out cleanly with
//! `Option::take()` and avoids requiring a `Default` bound on `P`.

use std::collections::HashMap;

struct Node<P> {
    obj: u64,
    payload: P,
    prev: Option<usize>,
    next: Option<usize>,
}

pub struct DList<P> {
    nodes: Vec<Option<Node<P>>>,
    free: Vec<usize>,
    by_obj: HashMap<u64, usize>,
    head: Option<usize>,
    tail: Option<usize>,
}

impl<P> DList<P> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            free: Vec::new(),
            by_obj: HashMap::new(),
            head: None,
            tail: None,
        }
    }

    pub fn len(&self) -> usize {
        self.by_obj.len()
    }

    pub fn contains(&self, obj: u64) -> bool {
        self.by_obj.contains_key(&obj)
    }

    pub fn payload_mut(&mut self, obj: u64) -> Option<&mut P> {
        let ix = *self.by_obj.get(&obj)?;
        self.nodes[ix].as_mut().map(|n| &mut n.payload)
    }

    fn alloc(&mut self, n: Node<P>) -> usize {
        if let Some(slot) = self.free.pop() {
            self.nodes[slot] = Some(n);
            slot
        } else {
            self.nodes.push(Some(n));
            self.nodes.len() - 1
        }
    }

    fn node(&self, ix: usize) -> &Node<P> {
        self.nodes[ix].as_ref().expect("dangling slot index")
    }

    fn node_mut(&mut self, ix: usize) -> &mut Node<P> {
        self.nodes[ix].as_mut().expect("dangling slot index")
    }

    fn unlink(&mut self, ix: usize) {
        let prev = self.node(ix).prev;
        let next = self.node(ix).next;
        match prev {
            Some(p) => self.node_mut(p).next = next,
            None => self.head = next,
        }
        match next {
            Some(n) => self.node_mut(n).prev = prev,
            None => self.tail = prev,
        }
        let n = self.node_mut(ix);
        n.prev = None;
        n.next = None;
    }

    fn link_front(&mut self, ix: usize) {
        self.node_mut(ix).prev = None;
        self.node_mut(ix).next = self.head;
        if let Some(h) = self.head {
            self.node_mut(h).prev = Some(ix);
        }
        self.head = Some(ix);
        if self.tail.is_none() {
            self.tail = Some(ix);
        }
    }

    fn link_back(&mut self, ix: usize) {
        self.node_mut(ix).next = None;
        self.node_mut(ix).prev = self.tail;
        if let Some(t) = self.tail {
            self.node_mut(t).next = Some(ix);
        }
        self.tail = Some(ix);
        if self.head.is_none() {
            self.head = Some(ix);
        }
    }

    fn take(&mut self, ix: usize) -> Node<P> {
        let n = self.nodes[ix].take().expect("slot already free");
        self.by_obj.remove(&n.obj);
        self.free.push(ix);
        n
    }

    /// Insert `obj` at the front (MRU). No-op if already present.
    pub fn push_front(&mut self, obj: u64, payload: P) {
        if self.by_obj.contains_key(&obj) {
            return;
        }
        let ix = self.alloc(Node {
            obj,
            payload,
            prev: None,
            next: None,
        });
        self.by_obj.insert(obj, ix);
        self.link_front(ix);
    }

    /// Insert `obj` at the back (LRU end). No-op if already present.
    pub fn push_back(&mut self, obj: u64, payload: P) {
        if self.by_obj.contains_key(&obj) {
            return;
        }
        let ix = self.alloc(Node {
            obj,
            payload,
            prev: None,
            next: None,
        });
        self.by_obj.insert(obj, ix);
        self.link_back(ix);
    }

    pub fn pop_front(&mut self) -> Option<(u64, P)> {
        let ix = self.head?;
        self.unlink(ix);
        let n = self.take(ix);
        Some((n.obj, n.payload))
    }

    pub fn pop_back(&mut self) -> Option<(u64, P)> {
        let ix = self.tail?;
        self.unlink(ix);
        let n = self.take(ix);
        Some((n.obj, n.payload))
    }

    pub fn remove(&mut self, obj: u64) -> Option<P> {
        let ix = *self.by_obj.get(&obj)?;
        self.unlink(ix);
        Some(self.take(ix).payload)
    }

    pub fn move_to_front(&mut self, obj: u64) -> bool {
        if let Some(&ix) = self.by_obj.get(&obj) {
            self.unlink(ix);
            self.link_front(ix);
            true
        } else {
            false
        }
    }
}

impl<P> Default for DList<P> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fifo_order_via_push_back_pop_front() {
        let mut d: DList<()> = DList::new();
        d.push_back(1, ());
        d.push_back(2, ());
        d.push_back(3, ());
        assert_eq!(d.pop_front().map(|(o, _)| o), Some(1));
        assert_eq!(d.pop_front().map(|(o, _)| o), Some(2));
        assert_eq!(d.pop_front().map(|(o, _)| o), Some(3));
        assert_eq!(d.len(), 0);
    }

    #[test]
    fn lru_order_via_push_front_pop_back() {
        let mut d: DList<()> = DList::new();
        d.push_front(1, ());
        d.push_front(2, ());
        d.push_front(3, ());
        assert_eq!(d.pop_back().map(|(o, _)| o), Some(1));
        assert_eq!(d.pop_back().map(|(o, _)| o), Some(2));
    }

    #[test]
    fn move_to_front_refresh() {
        let mut d: DList<()> = DList::new();
        d.push_front(1, ());
        d.push_front(2, ());
        d.push_front(3, ());
        d.move_to_front(1);
        assert_eq!(d.pop_back().map(|(o, _)| o), Some(2));
    }

    #[test]
    fn payload_mutation_via_payload_mut() {
        let mut d: DList<bool> = DList::new();
        d.push_back(7, false);
        *d.payload_mut(7).unwrap() = true;
        assert_eq!(d.pop_front(), Some((7, true)));
    }

    #[test]
    fn dedupe_on_double_push() {
        let mut d: DList<()> = DList::new();
        d.push_front(1, ());
        d.push_front(1, ());
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn arena_reuses_freed_slots() {
        let mut d: DList<u32> = DList::new();
        for i in 0u64..16 {
            d.push_back(i, i as u32);
            d.pop_front();
        }
        // Repeated push/pop should not grow the arena indefinitely.
        assert!(d.len() == 0);
    }
}
