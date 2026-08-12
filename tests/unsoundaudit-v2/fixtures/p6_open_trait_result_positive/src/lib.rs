#![allow(dead_code)]

pub trait IndexSource {
    fn index(&self) -> usize;
}

pub fn read_with<S: IndexSource>(source: &S, data: &[u8]) -> u8 {
    let index = source.index();
    unsafe { *data.get_unchecked(index) }
}
