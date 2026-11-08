#![allow(dead_code)]

pub struct Arena {
    data: [u8; 8],
    len: usize,
}

impl Arena {
    pub fn new(len: usize) -> Option<Self> {
        if !(1..=8).contains(&len) {
            return None;
        }
        Some(Self { data: [0; 8], len })
    }

    pub fn internal_bounds(&self) -> u8 {
        let shortened_len = self.len.saturating_sub(1);
        let shortened = &self.data[..shortened_len];
        unsafe { *shortened.get_unchecked(shortened_len) }
    }

    pub fn internal_width(&self) -> u32 {
        let offset = self.len.saturating_sub(1);
        let pointer = self.data.as_ptr().wrapping_add(offset).cast::<u32>();
        unsafe { pointer.read_unaligned() }
    }

    pub fn checked(&self) -> Option<u8> {
        let index = self.len.saturating_sub(1);
        if index >= self.data.len() {
            return None;
        }
        Some(unsafe { *self.data.get_unchecked(index) })
    }
}

pub fn bypass_stays_p1(slice: &[u8], index: usize, bump: bool) -> u8 {
    let selected = if bump { index.saturating_add(1) } else { index };
    unsafe { *slice.get_unchecked(selected) }
}

pub fn transparent_aggregate_stays_p1(pointer: *const u8) -> u8 {
    let wrapper = (pointer,);
    unsafe { *wrapper.0 }
}

pub fn validity_weakening_view_stays_p1(pointer: *const u8) -> i8 {
    let pointer = pointer.cast::<i8>();
    unsafe { *pointer }
}

pub fn restricted_target_is_p4(pointer: *const u8) -> bool {
    let pointer = pointer.cast::<bool>();
    unsafe { *pointer }
}

pub fn wider_target_is_p4(pointer: *const u8) -> u32 {
    let pointer = pointer.cast::<u32>();
    unsafe { *pointer }
}

pub fn assert_public_bool_is_p1(condition: bool) {
    unsafe { core::hint::assert_unchecked(condition) }
}

pub fn assert_internal_comparison_is_p4(left: usize, right: usize) {
    let condition = left < right;
    unsafe { core::hint::assert_unchecked(condition) }
}

pub fn assert_guarded_comparison_is_discharged(left: usize, right: usize) {
    let condition = left < right;
    if !condition {
        return;
    }
    unsafe { core::hint::assert_unchecked(condition) }
}

pub fn assert_stale_comparison_stays_p4(mut left: usize, right: usize) {
    let condition = left < right;
    if !condition {
        return;
    }
    left = left.wrapping_add(1);
    core::hint::black_box(left);
    unsafe { core::hint::assert_unchecked(condition) }
}

pub fn assert_diamond_without_must_proof_stays_p4(left: usize, right: usize, check: bool) {
    let condition = left < right;
    if check && !condition {
        return;
    }
    unsafe { core::hint::assert_unchecked(condition) }
}

pub fn ordinary_assert_is_not_an_unsafe_sink(condition: bool) {
    assert!(condition);
}

#[repr(transparent)]
pub struct SafeFlag(u8);

impl SafeFlag {
    pub fn new(value: bool) -> Self {
        Self(value as u8)
    }

    // The caller supplies a valid typed reference, not arbitrary validity
    // bits. Without a modeled private-invariant proof this is Unknown rather
    // than a concrete counterexample.
    pub fn as_bool_mut(&mut self) -> &mut bool {
        unsafe { &mut *(&mut self.0 as *mut u8).cast::<bool>() }
    }
}
