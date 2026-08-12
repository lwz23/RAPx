#![allow(dead_code)]

pub trait Transform {
    fn apply(&self, value: usize) -> usize;
}

pub fn call_opaque(transform: &dyn Transform, value: usize) -> usize {
    transform.apply(value)
}
