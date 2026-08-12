#![allow(dead_code)]

pub fn call_opaque(callback: fn(usize) -> usize, value: usize) -> usize {
    callback(value)
}
