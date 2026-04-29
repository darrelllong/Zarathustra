//! ARC (Megiddo & Modha, FAST 2003).
//!
//! Skeleton only — full T1/T2/B1/B2 + adaptive `p` to land in v1. The trait
//! impl below is a placeholder so the binary links and the CLI wires through.

use super::{Outcome, Policy};

pub struct Arc {
    capacity: usize,
    // T1, T2: resident lists; B1, B2: ghost lists; p: adaptive target size for T1.
    // Real fields land with the implementation.
}

impl Arc {
    pub fn with_capacity(capacity: usize) -> Self {
        Self { capacity }
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
        0
    }

    fn access(&mut self, _obj_id: u64) -> Outcome {
        // STUB: every access reported as a miss until the real ARC lands.
        Outcome::Miss
    }
}
