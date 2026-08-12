#![allow(dead_code)]

pub struct PublicIndex {
    pub index: usize,
    data: [u8; 4],
}

#[inline(never)]
fn read(data: &[u8], index: usize) -> u8 {
    unsafe { *data.get_unchecked(index) }
}

impl PublicIndex {
    pub fn new(index: usize) -> Self {
        Self { index, data: [0; 4] }
    }

    pub fn read(&self) -> Option<u8> {
        if self.index >= self.data.len() {
            return None;
        }
        Some(read(&self.data, self.index))
    }
}
