#![allow(dead_code)]
#![feature(auto_traits)]

pub trait ArbitraryOffsetProvider {
    fn index(&self) -> usize;
}

pub fn open_trait(source: &impl ArbitraryOffsetProvider, data: &[u8]) -> u8 {
    let index = source.index();
    unsafe { *data.get_unchecked(index) }
}

pub fn dyn_open_trait(source: &dyn ArbitraryOffsetProvider, data: &[u8]) -> u8 {
    let index = source.index();
    unsafe { *data.get_unchecked(index) }
}

pub fn callback_result(callback: fn() -> *const u8) -> u8 {
    let pointer = callback();
    unsafe { *pointer }
}

pub fn overwritten_callback(mut callback: fn() -> *const u8, data: &[u8]) -> Option<u8> {
    if data.is_empty() {
        return None;
    }
    callback = || core::ptr::null();
    let _ = callback();
    Some(unsafe { *data.get_unchecked(0) })
}

pub fn validated_trait(source: &impl ArbitraryOffsetProvider, data: &[u8]) -> Option<u8> {
    let index = source.index();
    if index >= data.len() {
        return None;
    }
    Some(unsafe { *data.get_unchecked(index) })
}

mod private_behavior {
    pub(crate) trait Offset {
        fn index(&self) -> usize;
    }

    pub(crate) struct Zero;

    impl Offset for Zero {
        fn index(&self) -> usize {
            0
        }
    }

    pub(crate) fn read<T: Offset>(source: &T, data: &[u8]) -> u8 {
        let index = source.index();
        unsafe { *data.get_unchecked(index) }
    }
}

pub fn private_trait_behavior(data: &[u8]) -> Option<u8> {
    if data.is_empty() {
        return None;
    }
    Some(private_behavior::read(&private_behavior::Zero, data))
}

mod sealed {
    pub(crate) trait Sealed {}
}

pub trait SealedOffsetProvider: sealed::Sealed {
    fn index(&self) -> usize;
}

pub struct SealedZero;

impl sealed::Sealed for SealedZero {}

impl SealedOffsetProvider for SealedZero {
    fn index(&self) -> usize {
        0
    }
}

fn read_from_sealed<T: SealedOffsetProvider>(source: &T, data: &[u8]) -> u8 {
    let index = source.index();
    unsafe { *data.get_unchecked(index) }
}

pub fn sealed_trait_behavior(data: &[u8]) -> Option<u8> {
    if data.is_empty() {
        return None;
    }
    Some(read_from_sealed(&SealedZero, data))
}

mod auto_supertrait {
    pub(crate) auto trait AutomaticallyImplemented {}
}

pub trait AutoSupertraitOffsetProvider: auto_supertrait::AutomaticallyImplemented {
    fn index(&self) -> usize;
}

pub fn auto_supertrait_open(source: &impl AutoSupertraitOffsetProvider, data: &[u8]) -> u8 {
    let index = source.index();
    unsafe { *data.get_unchecked(index) }
}

pub fn concrete_trait_dispatch(index: usize, data: &[u8]) -> Option<u8> {
    if data.is_empty() {
        return None;
    }
    let upper = data.len() - 1;
    let bounded = core::cmp::Ord::min(index, upper);
    Some(unsafe { *data.get_unchecked(bounded) })
}

pub struct ConcreteZero;

impl ArbitraryOffsetProvider for ConcreteZero {
    fn index(&self) -> usize {
        0
    }
}

pub fn concrete_open_trait(data: &[u8]) -> Option<u8> {
    if data.is_empty() {
        return None;
    }
    let index = ArbitraryOffsetProvider::index(&ConcreteZero);
    Some(unsafe { *data.get_unchecked(index) })
}

fn identity_pointer(pointer: *const u8) -> *const u8 {
    pointer
}

pub fn internal_function_pointer(data: &[u8]) -> u8 {
    let callback: fn(*const u8) -> *const u8 = identity_pointer;
    let pointer = callback(data.as_ptr());
    unsafe { *pointer }
}

pub fn callback_body_unsafe(data: &[u8], index: usize) -> u8 {
    let callback: fn(&[u8], usize) -> u8 = |bytes, offset| unsafe { *bytes.get_unchecked(offset) };
    callback(data, index)
}

fn discard<T>(_value: T) {}

pub fn discarded_closure(data: &[u8], index: usize) {
    let callback = || unsafe { *data.get_unchecked(index) };
    discard(callback);
}

extern "C" {
    fn write_pointer(output: *mut *mut u8) -> i32;
    fn foreign_pointer() -> *const u8;
    fn foreign_flag() -> bool;
    fn foreign_tuple() -> (usize,);
    fn opaque_pointer_sink(pointer: *const u8);
}

pub fn direct_ffi_return() -> u8 {
    let pointer = unsafe { foreign_pointer() };
    unsafe { *pointer }
}

#[inline(never)]
fn ffi_return_inner() -> *const u8 {
    unsafe { foreign_pointer() }
}

#[inline(never)]
fn ffi_return_outer() -> *const u8 {
    ffi_return_inner()
}

pub fn wrapped_ffi_return() -> u8 {
    let pointer = ffi_return_outer();
    unsafe { *pointer }
}

#[no_mangle]
pub extern "C" fn ffi_callback_input(pointer: *const u8) -> u8 {
    unsafe { *pointer }
}

pub fn sibling_ffi_callsite_does_not_splice(data: &[u8]) -> Option<u8> {
    let _unused = unsafe { foreign_pointer() };
    if data.is_empty() {
        return None;
    }
    Some(unsafe { *data.get_unchecked(0) })
}

pub fn sibling_ffi_branch_does_not_splice(flag: bool, data: &[u8]) -> Option<u8> {
    if flag {
        let _unused = unsafe { foreign_pointer() };
        return None;
    }
    if data.is_empty() {
        return None;
    }
    Some(unsafe { *data.get_unchecked(0) })
}

pub fn opaque_callback_sink(callback: fn() -> *const u8) {
    let pointer = callback();
    unsafe { opaque_pointer_sink(pointer) }
}

pub fn ffi_boolean_not_an_index(data: &[u8]) -> Option<u8> {
    if data.is_empty() {
        return None;
    }
    let flag = unsafe { foreign_flag() };
    let index = flag as usize;
    Some(unsafe { *data.get_unchecked(index) })
}

pub fn ffi_aggregate_not_an_index(data: &[u8]) -> u8 {
    let aggregate = unsafe { foreign_tuple() };
    unsafe { *data.get_unchecked(aggregate.0) }
}

pub fn indirect_ffi_function_pointer() -> u8 {
    let callback: unsafe extern "C" fn() -> *const u8 = foreign_pointer;
    let pointer = unsafe { callback() };
    unsafe { *pointer }
}

pub fn ffi_out_through_closure() -> Option<core::ptr::NonNull<u8>> {
    let mut pointer = core::ptr::null_mut();
    let status = unsafe { write_pointer(&mut pointer) };
    if status == 0 {
        Some(unsafe { core::ptr::NonNull::new_unchecked(pointer) })
    } else {
        None
    }
}
