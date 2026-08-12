#![allow(dead_code)]

pub fn call_opaque<F: Fn(usize) -> usize>(callback: F, value: usize) -> usize {
    callback(value)
}
