#![allow(dead_code)]

pub trait Provider {
    type Results: AsRef<[u64]> + Default;
}

pub fn first_result<P: Provider>() -> u64 {
    let results = P::Results::default();
    unsafe { *results.as_ref().get_unchecked(0) }
}

pub struct EmptyProvider;
impl Provider for EmptyProvider {
    type Results = [u64; 0];
}
