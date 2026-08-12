#![allow(dead_code)]

#[inline(never)]
fn sink(slice: &[u8], index: usize) -> u8 {
    unsafe { *slice.get_unchecked(index) }
}
#[inline(never)]
fn hop5(slice: &[u8], index: usize) -> u8 { sink(slice, index) }
#[inline(never)]
fn hop4(slice: &[u8], index: usize) -> u8 { hop5(slice, index) }
#[inline(never)]
fn hop3(slice: &[u8], index: usize) -> u8 { hop4(slice, index) }
#[inline(never)]
fn hop2(slice: &[u8], index: usize) -> u8 { hop3(slice, index) }
#[inline(never)]
fn hop1(slice: &[u8], index: usize) -> u8 { hop2(slice, index) }

pub fn read(slice: &[u8], index: usize) -> u8 {
    hop1(slice, index)
}
