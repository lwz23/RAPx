#![allow(dead_code)]

pub fn read_from_raw(pointer: *const u8) -> u8 {
    // Intentionally unsound test fixture: the public caller supplies the raw pointer.
    unsafe { *pointer }
}
