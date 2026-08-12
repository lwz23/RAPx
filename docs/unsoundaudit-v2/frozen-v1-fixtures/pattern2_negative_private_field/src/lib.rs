#![allow(dead_code)]

pub struct PrivateRaw {
    pointer: *const u8,
}

impl PrivateRaw {
    pub fn new(pointer: *const u8) -> Self {
        Self { pointer }
    }

    pub fn read_private_field(&self) -> u8 {
        let pointer = self.pointer;
        // Unsafe is present, but the field is deliberately not public.
        unsafe { *pointer }
    }
}
