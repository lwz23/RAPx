#![allow(dead_code)]

#[inline(never)]
fn constant_index(_caller_index: usize) -> usize {
    0
}

pub fn read(index: usize) -> u8 {
    let bytes = [7_u8];
    let selected = constant_index(index);
    unsafe { *bytes.get_unchecked(selected) }
}
