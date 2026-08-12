#![allow(dead_code)]

include!("generated_macro.rs");

pub fn read(slice: &[u8], index: usize) -> u8 {
    unchecked_byte!(slice, index)
}
