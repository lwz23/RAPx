#![allow(dead_code)]

pub struct Packet {
    bytes: [u8; 8],
}

impl Packet {
    pub fn read_word(&self, offset: usize) -> Option<u32> {
        let width = core::mem::size_of::<u32>();
        let end = offset.checked_add(width)?;
        if end > self.bytes.len() {
            return None;
        }
        let pointer = self.bytes.as_ptr().wrapping_add(offset).cast::<u32>();
        Some(unsafe { pointer.read_unaligned() })
    }
}
