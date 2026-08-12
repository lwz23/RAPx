#![allow(dead_code)]

pub struct Packet {
    bytes: [u8; 8],
    used: usize,
}

impl Packet {
    pub fn new(used: usize) -> Option<Self> {
        if !(1..=8).contains(&used) {
            return None;
        }
        Some(Self { bytes: [0; 8], used })
    }

    pub fn trailing_word(&self) -> u32 {
        let offset = self.used.saturating_sub(1);
        let pointer = self.bytes.as_ptr().wrapping_add(offset).cast::<u32>();
        unsafe { pointer.read_unaligned() }
    }
}
