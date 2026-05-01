//! LIRS (Low Inter-reference Recency Set) — Jiang & Zhang, SIGMETRICS 2002.
//!
//! Cache split into LIR (Low IRR) blocks and HIR (High IRR) blocks.
//! Default split: ~99% LIR / 1% HIR. Most workloads scan-resist by keeping
//! reuses in LIR and probing scans in HIR.
//!
//! Two structures:
//!   - LIRS stack S: contains all LIR blocks PLUS recently-accessed HIR
//!     blocks (resident or non-resident "ghost" entries). Bottom of S is
//!     stack-pruned: LIR at bottom is irreversibly LIR; non-LIR entries
//!     at bottom are ejected on prune.
//!   - HIR queue Q: resident HIR blocks only (FIFO order on misses).
//!
//! Access protocol (Jiang/Zhang Fig 3 / Table 2):
//!   On access(x):
//!     1. If x is a LIR block (resident in S as LIR):
//!          → Hit. Move x to top of S. If x was at bottom of S, prune S.
//!     2. If x is a HIR block resident in cache:
//!          → Hit. If x is also in S (as a HIR-in-S entry):
//!              promote x to LIR (move to top of S),
//!              demote bottom-LIR of S to HIR (move it to MRU of Q),
//!              prune S so its bottom is again LIR.
//!            Else (x not in S, only in Q):
//!              just move x to top of S as HIR-in-S, AND move x to MRU of Q.
//!     3. If x is non-resident (ghost in S, or unseen):
//!          → Miss. Allocate a HIR slot:
//!            evict LRU of Q if Q full;
//!            if x WAS in S (as ghost): promote-on-rehit treat as case (2)
//!              after the resident insert;
//!            otherwise insert as plain HIR resident at MRU of S and Q.
//!
//! Stack pruning: after any operation that may leave non-LIR entries at
//! the bottom of S, repeatedly remove bottom non-LIR entries until the
//! bottom is LIR (or S is empty).
//!
//! This implementation: textbook LIRS with default ratio 0.01 HIR,
//! minimum HIR=2 (so even at small cache the algorithm exists). LIR
//! count = capacity - HIR count. Stack S is unbounded in nodes (ghosts
//! can pile up); we cap S size at 4× capacity to keep memory bounded
//! without hurting hit ratio meaningfully (per Jiang/Zhang follow-ups).

use std::collections::HashMap;

use super::{Outcome, Policy};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    /// LIR block (always resident).
    Lir,
    /// HIR block, resident in cache (in queue Q).
    HirResident,
    /// HIR block, non-resident (ghost in stack S only).
    HirGhost,
}

struct Node {
    obj: u64,
    status: Status,
    s_prev: Option<usize>, // S list (stack)
    s_next: Option<usize>,
    q_prev: Option<usize>, // Q list (HIR resident queue)
    q_next: Option<usize>,
    in_s: bool,
    in_q: bool,
}

pub struct Lirs {
    capacity: usize,
    lir_capacity: usize,
    hir_capacity: usize,
    lir_count: usize,
    hir_resident_count: usize,
    s_max: usize, // soft cap on stack S size

    nodes: Vec<Option<Node>>,
    free: Vec<usize>,
    by_obj: HashMap<u64, usize>,

    s_head: Option<usize>, // top of S (most recent)
    s_tail: Option<usize>, // bottom of S (eviction-prune end)
    s_len: usize,
    q_head: Option<usize>, // MRU of Q
    q_tail: Option<usize>, // LRU of Q (eviction end)
    q_len: usize,
}

impl Lirs {
    pub fn with_capacity(capacity: usize) -> Self {
        // Standard LIRS HIR ratio is 1%, with a minimum of 2 HIR slots so
        // the algorithm can actually distinguish LIR from HIR at small cap.
        let hir_capacity = ((capacity as f64) * 0.01).ceil().max(2.0) as usize;
        let hir_capacity = hir_capacity.min(capacity.saturating_sub(1).max(1));
        let lir_capacity = capacity.saturating_sub(hir_capacity);
        let s_max = (capacity * 4).max(64);
        Self {
            capacity,
            lir_capacity,
            hir_capacity,
            lir_count: 0,
            hir_resident_count: 0,
            s_max,
            nodes: Vec::with_capacity(capacity * 2),
            free: Vec::new(),
            by_obj: HashMap::with_capacity(capacity * 2),
            s_head: None,
            s_tail: None,
            s_len: 0,
            q_head: None,
            q_tail: None,
            q_len: 0,
        }
    }

    fn node(&self, ix: usize) -> &Node {
        self.nodes[ix].as_ref().expect("dangling slot")
    }

    fn node_mut(&mut self, ix: usize) -> &mut Node {
        self.nodes[ix].as_mut().expect("dangling slot")
    }

    fn alloc(&mut self, obj: u64, status: Status) -> usize {
        let n = Node {
            obj,
            status,
            s_prev: None,
            s_next: None,
            q_prev: None,
            q_next: None,
            in_s: false,
            in_q: false,
        };
        let ix = if let Some(slot) = self.free.pop() {
            self.nodes[slot] = Some(n);
            slot
        } else {
            self.nodes.push(Some(n));
            self.nodes.len() - 1
        };
        self.by_obj.insert(obj, ix);
        ix
    }

    fn dealloc(&mut self, ix: usize) {
        let obj = self.node(ix).obj;
        self.by_obj.remove(&obj);
        self.nodes[ix] = None;
        self.free.push(ix);
    }

    // ---- S list ops ----
    fn s_unlink(&mut self, ix: usize) {
        if !self.node(ix).in_s { return; }
        let prev = self.node(ix).s_prev;
        let next = self.node(ix).s_next;
        match prev {
            Some(p) => self.node_mut(p).s_next = next,
            None => self.s_head = next,
        }
        match next {
            Some(n) => self.node_mut(n).s_prev = prev,
            None => self.s_tail = prev,
        }
        let n = self.node_mut(ix);
        n.s_prev = None;
        n.s_next = None;
        n.in_s = false;
        self.s_len -= 1;
    }

    fn s_push_head(&mut self, ix: usize) {
        let old = self.s_head;
        let n = self.node_mut(ix);
        n.s_prev = None;
        n.s_next = old;
        n.in_s = true;
        if let Some(h) = old {
            self.node_mut(h).s_prev = Some(ix);
        }
        self.s_head = Some(ix);
        if self.s_tail.is_none() {
            self.s_tail = Some(ix);
        }
        self.s_len += 1;
    }

    /// Move existing-in-S node to head of S.
    fn s_move_to_head(&mut self, ix: usize) {
        if self.s_head == Some(ix) { return; }
        if !self.node(ix).in_s { return self.s_push_head(ix); }
        self.s_unlink(ix);
        self.s_push_head(ix);
    }

    /// Stack pruning: remove non-LIR entries from bottom of S until
    /// bottom is LIR or S is empty. Drops HirGhost ghosts entirely;
    /// HirResident at the bottom is also removed from S but stays in Q.
    fn s_prune(&mut self) {
        while let Some(t) = self.s_tail {
            match self.node(t).status {
                Status::Lir => break,
                Status::HirResident => {
                    self.s_unlink(t);
                }
                Status::HirGhost => {
                    self.s_unlink(t);
                    if !self.node(t).in_q {
                        self.dealloc(t);
                    }
                }
            }
        }
        // Soft cap on S: drop stale ghost tail when too long. Keeps memory bounded.
        while self.s_len > self.s_max {
            if let Some(t) = self.s_tail {
                if matches!(self.node(t).status, Status::Lir) {
                    break;
                }
                self.s_unlink(t);
                if matches!(self.node(t).status, Status::HirGhost) && !self.node(t).in_q {
                    self.dealloc(t);
                }
            } else {
                break;
            }
        }
    }

    // ---- Q list ops ----
    fn q_unlink(&mut self, ix: usize) {
        if !self.node(ix).in_q { return; }
        let prev = self.node(ix).q_prev;
        let next = self.node(ix).q_next;
        match prev {
            Some(p) => self.node_mut(p).q_next = next,
            None => self.q_head = next,
        }
        match next {
            Some(n) => self.node_mut(n).q_prev = prev,
            None => self.q_tail = prev,
        }
        let n = self.node_mut(ix);
        n.q_prev = None;
        n.q_next = None;
        n.in_q = false;
        self.q_len -= 1;
    }

    fn q_push_head(&mut self, ix: usize) {
        let old = self.q_head;
        let n = self.node_mut(ix);
        n.q_prev = None;
        n.q_next = old;
        n.in_q = true;
        if let Some(h) = old {
            self.node_mut(h).q_prev = Some(ix);
        }
        self.q_head = Some(ix);
        if self.q_tail.is_none() {
            self.q_tail = Some(ix);
        }
        self.q_len += 1;
    }

    fn q_move_to_head(&mut self, ix: usize) {
        if self.q_head == Some(ix) { return; }
        self.q_unlink(ix);
        self.q_push_head(ix);
    }

    /// Evict LRU of Q (the HIR-resident with the longest IRR so far).
    /// Returns the slot index of the evicted node; the node remains as a
    /// HirGhost in S if it is still in S.
    fn q_evict_tail(&mut self) {
        let t = match self.q_tail { Some(t) => t, None => return };
        self.q_unlink(t);
        // Node may still be in S (as HIR-in-S). Convert resident → ghost.
        // If not in S either, dealloc.
        if self.node(t).in_s {
            self.node_mut(t).status = Status::HirGhost;
        } else {
            self.dealloc(t);
        }
        self.hir_resident_count -= 1;
    }

    /// Demote bottom-most LIR of S to HIR-resident, push to MRU of Q.
    fn demote_bottom_lir(&mut self) {
        // Find bottom-most LIR (s_prune has run, so s_tail IS LIR).
        let t = match self.s_tail { Some(t) => t, None => return };
        debug_assert!(matches!(self.node(t).status, Status::Lir));
        self.s_unlink(t);
        self.node_mut(t).status = Status::HirResident;
        self.q_push_head(t);
        self.lir_count -= 1;
        self.hir_resident_count += 1;
    }
}

impl Policy for Lirs {
    fn name(&self) -> &'static str {
        "lirs"
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn len(&self) -> usize {
        self.lir_count + self.hir_resident_count
    }

    fn access(&mut self, obj: u64) -> Outcome {
        if let Some(&ix) = self.by_obj.get(&obj) {
            match self.node(ix).status {
                Status::Lir => {
                    // Case 1: LIR hit. Move to top of S; if was at bottom, prune.
                    let was_tail = self.s_tail == Some(ix);
                    self.s_move_to_head(ix);
                    if was_tail {
                        self.s_prune();
                    }
                    Outcome::Hit
                }
                Status::HirResident => {
                    // Case 2: HIR resident hit.
                    let in_s = self.node(ix).in_s;
                    if in_s {
                        // Promote to LIR; demote bottom-LIR to HIR-in-Q.
                        self.s_unlink(ix);
                        self.q_unlink(ix);
                        self.node_mut(ix).status = Status::Lir;
                        self.s_push_head(ix);
                        self.lir_count += 1;
                        self.hir_resident_count -= 1;
                        if self.lir_count > self.lir_capacity {
                            self.demote_bottom_lir();
                        }
                        self.s_prune();
                    } else {
                        // Just move to S top (as HIR-in-S) and Q head.
                        self.s_push_head(ix);
                        self.q_move_to_head(ix);
                    }
                    Outcome::Hit
                }
                Status::HirGhost => {
                    // Case 3a: rehit on a non-resident ghost. This is a
                    // miss, but with promotion-on-rehit semantics: bring
                    // it back as LIR (because the IRR is small now —
                    // its previous reference is still in S).
                    // Make room in cache: if we're at capacity, evict
                    // LRU of Q.
                    let total = self.lir_count + self.hir_resident_count;
                    if total >= self.capacity {
                        self.q_evict_tail();
                    }
                    // Promote ghost → LIR resident; demote bottom-LIR.
                    self.s_unlink(ix);
                    self.node_mut(ix).status = Status::Lir;
                    self.s_push_head(ix);
                    self.lir_count += 1;
                    if self.lir_count > self.lir_capacity {
                        self.demote_bottom_lir();
                    }
                    self.s_prune();
                    Outcome::Miss
                }
            }
        } else {
            // Case 3b: cold miss (never seen).
            let total = self.lir_count + self.hir_resident_count;
            if self.lir_count < self.lir_capacity {
                // Fill LIR portion first.
                let ix = self.alloc(obj, Status::Lir);
                self.s_push_head(ix);
                self.lir_count += 1;
            } else {
                if total >= self.capacity {
                    self.q_evict_tail();
                }
                let ix = self.alloc(obj, Status::HirResident);
                self.s_push_head(ix); // also tracked in S for IRR detection
                self.q_push_head(ix);
                self.hir_resident_count += 1;
            }
            Outcome::Miss
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_misses_then_lir_hit() {
        let mut c = Lirs::with_capacity(4);
        // First 3 cold (capacity LIR=2, HIR=2, so 1 LIR + 2 HIR after fill)
        assert_eq!(c.access(1), Outcome::Miss);
        assert_eq!(c.access(2), Outcome::Miss);
        assert_eq!(c.access(3), Outcome::Miss);
        assert_eq!(c.access(4), Outcome::Miss);
        // Re-access one of the early ones — should hit.
        assert_eq!(c.access(1), Outcome::Hit);
    }

    #[test]
    fn hir_promotes_on_rehit() {
        // Sequence designed so a HIR block re-hits and promotes to LIR.
        let mut c = Lirs::with_capacity(3);
        c.access(1); c.access(2); c.access(3); // LIR=1 (1), HIR=2 (2,3)
        c.access(1); // hit on LIR
        c.access(2); // 2 is HIR-resident and in S → promote to LIR
        // 1 should still hit, 2 should still hit.
        assert_eq!(c.access(1), Outcome::Hit);
        assert_eq!(c.access(2), Outcome::Hit);
    }

    #[test]
    fn scan_resistance() {
        // Working set {100, 200, 300} repeatedly hit, then a long scan
        // over disjoint ids. LIRS at cap=12 (LIR cap = 10) should keep
        // the working set resident while the scan churns through HIR.
        let mut c = Lirs::with_capacity(12);
        // Establish working set as LIR (≥ 2 hits each)
        for _ in 0..3 {
            c.access(100); c.access(200); c.access(300);
        }
        // Inject a 50-item scan
        for x in 1..=50u64 {
            c.access(x);
        }
        // Working set should still hit
        assert_eq!(c.access(100), Outcome::Hit);
        assert_eq!(c.access(200), Outcome::Hit);
        assert_eq!(c.access(300), Outcome::Hit);
    }
}
