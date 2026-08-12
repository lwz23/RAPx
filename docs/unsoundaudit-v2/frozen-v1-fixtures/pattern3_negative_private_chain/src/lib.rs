#![allow(dead_code)]

#[inline(never)]
fn private_read(pointer: *const u8) -> u8 {
    unsafe { *pointer }
}

#[inline(never)]
fn private_forward(pointer: *const u8) -> u8 {
    private_read(pointer)
}

pub fn unrelated_public(value: u8) -> u8 {
    value
}
