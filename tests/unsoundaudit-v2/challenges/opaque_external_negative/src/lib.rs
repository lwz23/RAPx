#![allow(dead_code)]

extern "C" {
    fn external_transform(value: usize) -> usize;
}

pub fn call_opaque(value: usize) -> usize {
    unsafe { external_transform(value) }
}
