#![allow(dead_code)]

pub fn checked_utf8(bytes: &[u8]) -> Option<&str> {
    if core::str::from_utf8(bytes).is_err() {
        return None;
    }
    // The preceding check establishes the precondition for this fixture.
    Some(unsafe { core::str::from_utf8_unchecked(bytes) })
}

