#![feature(rustc_attrs, target_feature_11)]
#![allow(dead_code, internal_features, unsafe_op_in_unsafe_fn)]

use core::arch::asm;
use core::cell::UnsafeCell;

include!("included.rs");

#[rustc_layout_scalar_valid_range_start(1)]
#[repr(transparent)]
pub struct Restricted(pub u8);

#[rustc_layout_scalar_valid_range_start(1)]
#[repr(transparent)]
pub struct RestrictedCell(pub UnsafeCell<u8>);

pub union Number {
    pub integer: u32,
    pub bytes: [u8; 4],
}

static mut MUTABLE: u8 = 7;

extern "C" {
    static EXTERNAL: u8;
}

pub unsafe fn direct_target(pointer: *const u8) -> u8 {
    *pointer
}

#[target_feature(enable = "neon")]
pub fn neon_target() -> u8 {
    1
}

pub unsafe fn census_all(pointer: *const u8, mut restricted: RestrictedCell, number: Number) -> u8 {
    let direct = direct_target(pointer);
    let indirect: unsafe fn(*const u8) -> u8 = direct_target;
    let via_pointer = indirect(pointer);
    let raw = *pointer;
    let mutable = MUTABLE;
    let external = EXTERNAL;
    let union = number.integer as u8;
    let _restricted = Restricted(1);
    restricted.0 = UnsafeCell::new(2);
    let _shared = &restricted.0;
    let _borrowed = &mut restricted.0;
    let target_feature = neon_target();
    asm!("nop");
    direct ^ via_pointer ^ raw ^ mutable ^ external ^ union ^ target_feature
}

pub unsafe fn nested_owners(pointer: *const u8) -> (u8, u8, u8) {
    let closure = || *pointer;
    let from_closure = closure();
    let from_const = const { *(&9_u8 as *const u8) };
    let from_include = included_call(pointer);
    (from_closure, from_const, from_include)
}

#[macro_export]
macro_rules! raw_read {
    ($pointer:expr) => {{ *$pointer }};
}

pub unsafe fn macro_site(pointer: *const u8) -> u8 {
    raw_read!(pointer)
}

pub fn nested_explicit(pointer: *const u8) -> u8 {
    unsafe { unsafe { *pointer } }
}

pub fn safe_control(value: &u8) -> u8 {
    let _ = unsafe {};
    *value
}
