#![allow(dead_code)]

use core::mem::MaybeUninit;

#[inline(never)]
fn produce() -> bool {
    unsafe { MaybeUninit::<bool>::uninit().assume_init() }
}

#[inline(never)]
fn forward() -> bool {
    produce()
}

pub fn expose() -> bool {
    forward()
}
