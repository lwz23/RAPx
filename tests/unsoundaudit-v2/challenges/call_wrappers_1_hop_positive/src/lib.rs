#![allow(dead_code)]

#[inline(never)]
fn sink(slice: &[u8], index: usize) -> u8 {
    unsafe { *slice.get_unchecked(index) }
}

pub fn read(slice: &[u8], index: usize) -> u8 {
    sink(slice, index)
}
