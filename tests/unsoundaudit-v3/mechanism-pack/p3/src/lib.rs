#![allow(dead_code, invalid_value)]

use core::mem::MaybeUninit;

#[repr(u8)]
pub enum Tiny {
    Zero = 0,
    One = 1,
}

pub fn expose_bool() -> bool {
    unsafe { MaybeUninit::<bool>::uninit().assume_init() }
}

pub fn expose_enum() -> Tiny {
    unsafe { core::mem::transmute::<u8, Tiny>(3) }
}

pub fn not_exposed() -> u8 {
    let _invalid = unsafe { MaybeUninit::<bool>::uninit().assume_init() };
    0
}

pub fn precondition_only(slice: &[u8], index: usize) -> u8 {
    unsafe { *slice.get_unchecked(index) }
}

pub fn scalar_to_bytes(value: u32) -> [u8; 4] {
    unsafe { core::mem::transmute(value) }
}

pub fn pointer_to_address(pointer: *const u8) -> usize {
    unsafe { core::mem::transmute(pointer) }
}

pub fn initialized_then_exposed() -> bool {
    let mut value = MaybeUninit::<bool>::uninit();
    value.write(true);
    unsafe { value.assume_init() }
}

pub fn ptr_write_then_exposed() -> bool {
    let mut value = MaybeUninit::<bool>::uninit();
    unsafe {
        core::ptr::write(value.as_mut_ptr(), true);
        value.assume_init()
    }
}

pub fn ptr_write_via_copied_pointer() -> bool {
    let mut value = MaybeUninit::<bool>::uninit();
    let pointer = value.as_mut_ptr();
    let copied_pointer = pointer;
    unsafe {
        core::ptr::write(copied_pointer, true);
        value.assume_init()
    }
}

pub fn ptr_copy_initialized_then_exposed() -> bool {
    let source = MaybeUninit::new(true);
    let mut destination = MaybeUninit::<bool>::uninit();
    unsafe {
        core::ptr::copy_nonoverlapping(source.as_ptr(), destination.as_mut_ptr(), 1);
        destination.assume_init()
    }
}

pub fn ptr_copy_same_source_then_exposed() -> bool {
    let mut value = MaybeUninit::new(true);
    let pointer = value.as_mut_ptr();
    unsafe {
        core::ptr::copy(pointer, pointer, 1);
        value.assume_init()
    }
}

pub fn ptr_copy_uninitialized_then_exposed() -> bool {
    let source = MaybeUninit::<bool>::uninit();
    let mut destination = MaybeUninit::<bool>::uninit();
    unsafe {
        core::ptr::copy_nonoverlapping(source.as_ptr(), destination.as_mut_ptr(), 1);
        destination.assume_init()
    }
}

pub fn ptr_copy_zero_then_exposed() -> bool {
    let source = MaybeUninit::new(true);
    let mut destination = MaybeUninit::<bool>::uninit();
    unsafe {
        core::ptr::copy(source.as_ptr(), destination.as_mut_ptr(), 0);
        destination.assume_init()
    }
}

pub fn ptr_copy_branch_then_exposed(initialize: bool) -> bool {
    let source = MaybeUninit::new(true);
    let mut destination = MaybeUninit::<bool>::uninit();
    if initialize {
        unsafe {
            core::ptr::copy_nonoverlapping(source.as_ptr(), destination.as_mut_ptr(), 1);
        }
    }
    unsafe { destination.assume_init() }
}

pub fn ptr_copy_unknown_count_then_exposed(count: usize) -> bool {
    let source = [MaybeUninit::new(true), MaybeUninit::new(false)];
    let mut destination = [MaybeUninit::<bool>::uninit(), MaybeUninit::<bool>::uninit()];
    unsafe {
        core::ptr::copy_nonoverlapping(source[0].as_ptr(), destination[0].as_mut_ptr(), count);
        destination[0].assume_init()
    }
}

pub fn ptr_copy_two_then_exposed() -> bool {
    let source = [MaybeUninit::new(true), MaybeUninit::new(false)];
    let mut destination = [MaybeUninit::<bool>::uninit(), MaybeUninit::<bool>::uninit()];
    unsafe {
        core::ptr::copy_nonoverlapping(source[0].as_ptr(), destination[0].as_mut_ptr(), 2);
        destination[0].assume_init()
    }
}

pub fn ptr_write_ambiguous_then_exposed(select_first: bool) -> bool {
    let mut first = MaybeUninit::<bool>::uninit();
    let mut second = MaybeUninit::<bool>::uninit();
    let destination = if select_first {
        first.as_mut_ptr()
    } else {
        second.as_mut_ptr()
    };
    unsafe {
        core::ptr::write(destination, true);
        first.assume_init()
    }
}

pub fn pointer_method_write_then_exposed() -> bool {
    let mut value = MaybeUninit::<bool>::uninit();
    unsafe {
        value.as_mut_ptr().write(true);
        value.assume_init()
    }
}

pub fn write_bytes_then_exposed() -> bool {
    let mut value = MaybeUninit::<bool>::uninit();
    unsafe {
        core::ptr::write_bytes(value.as_mut_ptr(), 0, 1);
        value.assume_init()
    }
}

pub fn ptr_copy_joined_external_then_exposed(external: *mut bool, use_external: bool) -> bool {
    let source = MaybeUninit::new(true);
    let mut destination = MaybeUninit::<bool>::uninit();
    let target = if use_external {
        external
    } else {
        destination.as_mut_ptr()
    };
    unsafe {
        core::ptr::copy_nonoverlapping(source.as_ptr(), target, 1);
        destination.assume_init()
    }
}

pub fn ptr_copy_uninitialized_via_wrapping_add_then_exposed(offset: usize) -> bool {
    let source = MaybeUninit::<bool>::uninit();
    let mut destination = MaybeUninit::new(true);
    let target = destination.as_mut_ptr().wrapping_add(offset);
    unsafe {
        core::ptr::copy_nonoverlapping(source.as_ptr(), target, 1);
        destination.assume_init()
    }
}

#[repr(transparent)]
pub struct OpaqueByte(u8);

pub fn expose_unknown_type(value: u8) -> OpaqueByte {
    unsafe { core::mem::transmute(value) }
}

#[no_mangle]
pub extern "C" fn ffi_public_out(output: *mut bool) {
    let value = unsafe { MaybeUninit::<bool>::uninit().assume_init() };
    unsafe { *output = value }
}

extern "C" {
    fn ffi_initialize_bool(output: *mut bool);
}

pub fn ffi_initialized_then_exposed() -> bool {
    let mut value = MaybeUninit::<bool>::uninit();
    unsafe {
        ffi_initialize_bool(value.as_mut_ptr());
        value.assume_init()
    }
}
