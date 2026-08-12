#![allow(dead_code)]

pub fn extend_local_lifetime() -> &'static str {
    let owned = String::from("fixture");
    // Intentionally unsound: the returned reference outlives `owned`.
    unsafe { core::mem::transmute::<&str, &'static str>(owned.as_str()) }
}

