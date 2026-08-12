#![allow(dead_code)]

pub fn read(slice: &[u8], index: usize, first_path: bool) -> u8 {
    if first_path {
        if index >= slice.len() {
            return 0;
        }
    } else {
        core::hint::spin_loop();
    }
    unsafe { *slice.get_unchecked(index) }
}
