//! Cache replacement policies.

pub mod arc;
pub mod lru;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Hit,
    Miss,
}

pub trait Policy {
    fn name(&self) -> &'static str;
    fn capacity(&self) -> usize;
    fn access(&mut self, obj_id: u64) -> Outcome;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyKind {
    Lru,
    Arc,
}

pub fn make(kind: PolicyKind, capacity: usize) -> Box<dyn Policy> {
    match kind {
        PolicyKind::Lru => Box::new(lru::Lru::with_capacity(capacity)),
        PolicyKind::Arc => Box::new(arc::Arc::with_capacity(capacity)),
    }
}
