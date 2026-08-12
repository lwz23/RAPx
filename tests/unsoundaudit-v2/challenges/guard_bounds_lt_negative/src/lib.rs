#![allow(dead_code)]

pub fn read(slice: &[u8], index: usize) -> Option<u8> {
    if index < slice.len() {
        Some(unsafe { *slice.get_unchecked(index) })
    } else {
        None
    }
}
