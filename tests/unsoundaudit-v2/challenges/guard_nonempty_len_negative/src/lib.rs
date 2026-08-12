#![allow(dead_code)]

pub fn first(slice: &[u8]) -> Option<u8> {
    if slice.len() == 0 {
        return None;
    }
    Some(unsafe { *slice.get_unchecked(0) })
}
