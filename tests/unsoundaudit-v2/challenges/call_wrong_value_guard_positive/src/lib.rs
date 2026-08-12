#![allow(dead_code)]

#[inline(never)]
fn sink(slice: &[u8], index: usize) -> u8 {
    unsafe { *slice.get_unchecked(index) }
}

#[inline(never)]
fn guard_other_value(slice: &[u8], index: usize, checked: usize) -> u8 {
    if checked >= slice.len() {
        return 0;
    }
    sink(slice, index)
}

pub fn read(slice: &[u8], index: usize, checked: usize) -> u8 {
    guard_other_value(slice, index, checked)
}
