#![allow(dead_code)]

pub fn read(slice: &[u8], mut index: usize, rewrite: bool) -> u8 {
    if index >= slice.len() {
        return 0;
    }
    if rewrite {
        index = slice.len();
    }
    unsafe { *slice.get_unchecked(index) }
}
