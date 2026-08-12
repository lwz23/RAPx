#![allow(dead_code)]

#[inline(never)]
fn set(out: &mut usize, value: usize) {
    *out = value;
}

pub fn read(slice: &[u8], index: usize) -> u8 {
    let mut selected = 0;
    set(&mut selected, index);
    unsafe { *slice.get_unchecked(selected) }
}
