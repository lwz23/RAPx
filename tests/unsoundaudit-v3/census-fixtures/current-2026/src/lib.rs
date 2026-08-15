#![feature(unsafe_binders, unsafe_fields)]
#![allow(dead_code, incomplete_features, unsafe_op_in_unsafe_fn)]

use core::arch::asm;
use core::unsafe_binder::{unwrap_binder, wrap_binder};

include!("included.rs");

pub struct UnsafeState {
    pub unsafe value: u8,
}

pub enum UnsafeVariant {
    Item { unsafe value: u8 },
}

pub union Number {
    pub integer: u32,
    pub bytes: [u8; 4],
}

pub struct Droppable;

impl Drop for Droppable {
    fn drop(&mut self) {}
}

static mut MUTABLE: u8 = 7;

unsafe extern "C" {
    static EXTERNAL: u8;
}

pub unsafe fn direct_target(pointer: *const u8) -> u8 {
    *pointer
}

#[target_feature(enable = "neon")]
pub fn neon_target() -> u8 {
    1
}

pub unsafe fn census_all(pointer: *const u8, state: UnsafeState, number: Number) -> u8 {
    let direct = direct_target(pointer);
    let indirect: unsafe fn(*const u8) -> u8 = direct_target;
    let via_pointer = indirect(pointer);
    let raw = *pointer;
    let mutable = MUTABLE;
    let external = EXTERNAL;
    let union = number.integer as u8;
    let unsafe_field = state.value;
    let _constructed = UnsafeState { value: 1 };
    let _variant = UnsafeVariant::Item { value: 2 };
    let target_feature = neon_target();
    asm!("nop");
    direct ^ via_pointer ^ raw ^ mutable ^ external ^ union ^ unsafe_field ^ target_feature
}

pub unsafe fn binder(value: u8) -> u8 {
    let wrapped: unsafe<'a> &'a u8 = wrap_binder!(&value);
    *unwrap_binder!(wrapped)
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

pub fn raw_borrow_deref(pointer: *const u8) -> *const u8 {
    &raw const *pointer
}

pub fn raw_borrow_union(number: &Number) -> *const u32 {
    &raw const number.integer
}

pub fn safe_control(value: &u8) -> u8 {
    let _ = unsafe {};
    *value
}
