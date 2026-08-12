#![allow(dead_code)]

pub fn read(slice: &[u8], index: usize) -> u8 {
    unsafe { *slice.get_unchecked(index) }
}
