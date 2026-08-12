macro_rules! unchecked_byte {
    ($slice:expr, $index:expr) => {{
        unsafe { *$slice.get_unchecked($index) }
    }};
}
