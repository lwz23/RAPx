#![allow(dead_code)]

pub fn read(slice: &[u8], index: usize, rounds: usize) -> u8 {
    if index >= slice.len() {
        return 0;
    }
    let mut remaining = rounds;
    while remaining != 0 {
        remaining -= 1;
    }
    unsafe { *slice.get_unchecked(index) }
}
