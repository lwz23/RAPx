#![allow(dead_code)]

#[inline(never)]
fn descend(slice: &[u8], index: usize, depth: usize) -> u8 {
    if depth == 0 {
        unsafe { *slice.get_unchecked(index) }
    } else {
        descend(slice, index, depth - 1)
    }
}

pub fn read(slice: &[u8], index: usize, depth: usize) -> u8 {
    descend(slice, index, depth)
}
