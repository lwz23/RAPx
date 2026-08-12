#![allow(dead_code)]

pub trait Provider {
    type Results: AsRef<[u64]> + Default;
}

pub fn first_result<P: Provider>() -> Option<u64> {
    let results = P::Results::default();
    let slice = results.as_ref();
    if slice.is_empty() {
        return None;
    }
    Some(unsafe { *slice.get_unchecked(0) })
}

pub struct EmptyProvider;
impl Provider for EmptyProvider {
    type Results = [u64; 0];
}
