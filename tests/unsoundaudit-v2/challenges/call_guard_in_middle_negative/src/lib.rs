#![allow(dead_code)]

#[inline(never)]
fn sink(slice: &[u8], index: usize) -> u8 {
    unsafe { *slice.get_unchecked(index) }
}

#[inline(never)]
fn guard_then_sink(slice: &[u8], index: usize) -> Option<u8> {
    if index >= slice.len() {
        return None;
    }
    Some(sink(slice, index))
}

pub fn read(slice: &[u8], index: usize) -> Option<u8> {
    guard_then_sink(slice, index)
}
