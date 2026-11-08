#![allow(dead_code)]

pub struct Holder {
    ptr: *const u8,
}

impl Holder {
    pub fn read_unchecked(&self) -> u8 {
        unsafe { core::ptr::read(self.ptr) }
    }

    pub fn read_null_checked(&self) -> Option<u8> {
        let pointer = self.ptr;
        if pointer.is_null() {
            None
        } else {
            Some(unsafe { core::ptr::read(pointer) })
        }
    }
}

pub fn public_ptr_read(pointer: *const u8) -> u8 {
    unsafe { core::ptr::read(pointer) }
}

pub struct Indexed {
    data: Vec<u8>,
    i: usize,
}

impl Indexed {
    pub fn get_private_index(&self) -> u8 {
        unsafe { *self.data.get_unchecked(self.i) }
    }

    pub fn get_private_index_checked(&self) -> Option<u8> {
        let index = self.i;
        if index < self.data.len() {
            Some(unsafe { *self.data.get_unchecked(index) })
        } else {
            None
        }
    }
}

pub fn public_index(slice: &[u8], index: usize) -> u8 {
    unsafe { *slice.get_unchecked(index) }
}
