#![allow(dead_code)]

pub fn direct(pointer: *const u8) -> u8 {
    unsafe { *pointer }
}

#[inline(never)]
fn sink(pointer: *const u8) -> u8 {
    unsafe { *pointer }
}

#[inline(never)]
fn hop5(pointer: *const u8) -> u8 {
    sink(pointer)
}
#[inline(never)]
fn hop4(pointer: *const u8) -> u8 {
    hop5(pointer)
}
#[inline(never)]
fn hop3(pointer: *const u8) -> u8 {
    hop4(pointer)
}
#[inline(never)]
fn hop2(pointer: *const u8) -> u8 {
    hop3(pointer)
}
#[inline(never)]
fn hop1(pointer: *const u8) -> u8 {
    hop2(pointer)
}

pub fn five_hop(pointer: *const u8) -> u8 {
    hop1(pointer)
}

pub fn guarded(slice: &[u8], index: usize) -> Option<u8> {
    if index >= slice.len() {
        return None;
    }
    Some(unsafe { *slice.get_unchecked(index) })
}

extern "C" {
    fn opaque_pointer_sink(pointer: *const u8);
}

pub fn opaque_public_input(pointer: *const u8) {
    unsafe { opaque_pointer_sink(pointer) }
}

pub fn vec_index(values: &Vec<u8>, index: usize) -> u8 {
    unsafe { *values.get_unchecked(index) }
}

pub fn string_from_bytes(bytes: Vec<u8>) -> String {
    unsafe { String::from_utf8_unchecked(bytes) }
}

pub fn slice_from_ptr(pointer: *const u8, len: usize) -> &'static [u8] {
    unsafe { core::slice::from_raw_parts(pointer, len) }
}

pub fn cstr_from_ptr(pointer: *const core::ffi::c_char) -> &'static core::ffi::CStr {
    unsafe { core::ffi::CStr::from_ptr(pointer) }
}

pub fn ptr_add(pointer: *const u8, count: usize) -> *const u8 {
    unsafe { pointer.add(count) }
}

pub fn ptr_copy(src: *const u8, dst: *mut u8, len: usize) {
    unsafe { core::ptr::copy(src, dst, len) }
}

pub fn vec_from_raw(pointer: *mut u8, length: usize, capacity: usize) -> Vec<u8> {
    unsafe { Vec::from_raw_parts(pointer, length, capacity) }
}

pub fn guarded_vec(values: &Vec<u8>, index: usize) -> Option<u8> {
    let slice = values.as_slice();
    if index >= slice.len() {
        return None;
    }
    Some(unsafe { *slice.get_unchecked(index) })
}

pub unsafe fn already_unsafe_from_raw(pointer: *const u8, len: usize) -> &'static [u8] {
    core::slice::from_raw_parts(pointer, len)
}

