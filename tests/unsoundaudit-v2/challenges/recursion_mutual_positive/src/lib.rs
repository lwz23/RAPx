#![allow(dead_code)]

#[inline(never)]
fn left(slice: &[u8], index: usize, depth: usize) -> u8 {
    if depth == 0 {
        unsafe { *slice.get_unchecked(index) }
    } else {
        right(slice, index, depth - 1)
    }
}

#[inline(never)]
fn right(slice: &[u8], index: usize, depth: usize) -> u8 {
    left(slice, index, depth)
}

pub fn read(slice: &[u8], index: usize, depth: usize) -> u8 {
    right(slice, index, depth)
}
