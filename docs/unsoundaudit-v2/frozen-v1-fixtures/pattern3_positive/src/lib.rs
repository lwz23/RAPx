#![allow(dead_code)]

#[inline(never)]
fn private_read(pointer: *const u8) -> u8 {
    unsafe { *pointer }
}

#[inline(never)]
pub fn public_forward(pointer: *const u8) -> u8 {
    private_read(pointer)
}
