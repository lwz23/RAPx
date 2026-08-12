#![allow(dead_code)]

use core::mem::MaybeUninit;
use core::ptr::NonNull;

pub struct Checked {
    pub index: usize,
    data: [u8; 8],
}

pub trait IndexSource {
    fn index(&self) -> usize;
}

pub trait Provider {
    type Results: AsRef<[u64]> + Default;
}

extern "C" {
    fn acquire(out: *mut *mut u8) -> i32;
}

fn checked_helper(data: &[u8], index: usize) -> Option<u8> {
    data.get(index).copied()
}

impl Checked {
    pub fn new(index: usize) -> Self {
        Self { index, data: [0; 8] }
    }

    pub fn field(&self) -> Option<u8> {
        if self.index >= self.data.len() {
            return None;
        }
        Some(unsafe { *self.data.get_unchecked(self.index) })
    }
}

pub fn initialized_bool() -> bool {
    unsafe { MaybeUninit::new(false).assume_init() }
}

pub fn generic_first<P: Provider>() -> Option<u64> {
    let results = P::Results::default();
    let slice = results.as_ref();
    if slice.is_empty() { None } else { Some(unsafe { *slice.get_unchecked(0) }) }
}

pub fn ffi_pointer() -> Option<NonNull<u8>> {
    let mut out = core::ptr::null_mut();
    let status = unsafe { acquire(&mut out) };
    if status != 0 || out.is_null() { None } else { Some(unsafe { NonNull::new_unchecked(out) }) }
}

pub fn behavior<S: IndexSource>(source: &S, data: &[u8]) -> Option<u8> {
    checked_helper(data, source.index())
}
