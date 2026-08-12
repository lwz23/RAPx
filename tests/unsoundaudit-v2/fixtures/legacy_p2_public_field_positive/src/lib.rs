#![allow(dead_code)]

pub struct PublicRaw {
    pub pointer: *const u8,
}

impl PublicRaw {
    pub fn read_public_field(&self) -> u8 {
        let pointer = self.pointer;
        // Intentionally unsound test fixture: a public field feeds the unsafe read.
        unsafe { *pointer }
    }
}
