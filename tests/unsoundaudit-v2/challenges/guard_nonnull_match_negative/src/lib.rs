#![allow(dead_code)]

use core::ptr::NonNull;

extern "C" {
    fn acquire(out: *mut *mut u8) -> i32;
}

pub fn acquire_pointer() -> Option<NonNull<u8>> {
    let mut out = core::ptr::null_mut();
    unsafe { acquire(&mut out) };
    match out.is_null() {
        true => None,
        false => Some(unsafe { NonNull::new_unchecked(out) }),
    }
}
