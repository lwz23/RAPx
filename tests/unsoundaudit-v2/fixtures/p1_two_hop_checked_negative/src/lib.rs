#![allow(dead_code)]

#[inline(never)]
fn sink(slice: &[u8], index: usize) -> u8 {
    unsafe { *slice.get_unchecked(index) }
}

#[inline(never)]
fn forward(slice: &[u8], index: usize) -> u8 {
    sink(slice, index)
}

pub fn entry(slice: &[u8], index: usize) -> Option<u8> {
    if index >= slice.len() {
        return None;
    }
    Some(forward(slice, index))
}
