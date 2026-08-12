#![allow(dead_code)]

pub trait IndexSource {
    fn index(&self) -> usize;
}

pub fn read_with<S: IndexSource>(source: &S, data: &[u8]) -> Option<u8> {
    let index = source.index();
    if index >= data.len() {
        return None;
    }
    Some(unsafe { *data.get_unchecked(index) })
}
