#![allow(dead_code)]

pub struct Arena {
    data: [u8; 8],
    len: usize,
}

#[inline(never)]
fn read(slice: &[u8], index: usize) -> u8 {
    unsafe { *slice.get_unchecked(index) }
}

impl Arena {
    pub fn new(len: usize) -> Option<Self> {
        if !(1..=8).contains(&len) {
            return None;
        }
        Some(Self { data: [0; 8], len })
    }

    pub fn last_prefix_byte(&self) -> Option<u8> {
        let shortened_len = self.len.saturating_sub(1);
        let shortened = &self.data[..shortened_len];
        if shortened_len >= shortened.len() {
            return None;
        }
        Some(read(shortened, shortened_len))
    }
}
