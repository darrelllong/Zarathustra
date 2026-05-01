//! Cache replacement policies.

pub mod arc;
pub mod car;
pub mod fifo;
pub mod lfu;
pub mod lirs;
pub mod lru;
pub mod sieve;
pub mod slru;
pub mod util;

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
    Fifo,
    Sieve,
    Slru,
    Car,
    Lfu,
    Lirs,
}

pub fn make(kind: PolicyKind, capacity: usize) -> Box<dyn Policy> {
    match kind {
        PolicyKind::Lru => Box::new(lru::Lru::with_capacity(capacity)),
        PolicyKind::Arc => Box::new(arc::Arc::with_capacity(capacity)),
        PolicyKind::Fifo => Box::new(fifo::Fifo::with_capacity(capacity)),
        PolicyKind::Sieve => Box::new(sieve::Sieve::with_capacity(capacity)),
        PolicyKind::Slru => Box::new(slru::Slru::with_capacity(capacity)),
        PolicyKind::Car => Box::new(car::Car::with_capacity(capacity)),
        PolicyKind::Lfu => Box::new(lfu::Lfu::with_capacity(capacity)),
        PolicyKind::Lirs => Box::new(lirs::Lirs::with_capacity(capacity)),
    }
}
