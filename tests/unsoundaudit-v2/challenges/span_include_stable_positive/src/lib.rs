#![allow(dead_code)]

include!("included_sink.rs");

pub fn read(slice: &[u8], index: usize) -> u8 {
    included_sink(slice, index)
}
