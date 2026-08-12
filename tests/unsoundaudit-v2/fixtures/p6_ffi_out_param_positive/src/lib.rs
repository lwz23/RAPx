#![allow(dead_code)]

use core::ptr::NonNull;

extern "C" {
    fn acquire(out: *mut *mut u8) -> i32;
}

pub fn acquire_pointer() -> Option<NonNull<u8>> {
    let mut out = core::ptr::null_mut();
    let status = unsafe { acquire(&mut out) };
    if status != 0 {
        return None;
    }
    Some(unsafe { NonNull::new_unchecked(out) })
}
