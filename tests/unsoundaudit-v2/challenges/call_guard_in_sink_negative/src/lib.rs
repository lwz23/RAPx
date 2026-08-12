#![allow(dead_code)]

#[inline(never)]
fn guarded_sink(slice: &[u8], index: usize) -> Option<u8> {
    if index >= slice.len() {
        return None;
    }
    Some(unsafe { *slice.get_unchecked(index) })
}

pub fn read(slice: &[u8], index: usize) -> Option<u8> {
    guarded_sink(slice, index)
}
