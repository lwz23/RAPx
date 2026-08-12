#![allow(dead_code)]

pub fn read_local_pointer() -> u8 {
    let value = std::hint::black_box(17_u8);
    let pointer = &value as *const u8;
    // The unsafe operand originates locally, not from a public parameter.
    unsafe { *pointer }
}
