#[inline(never)]
fn included_sink(slice: &[u8], index: usize) -> u8 {
    unsafe { *slice.get_unchecked(index) }
}
