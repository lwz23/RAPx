#![allow(dead_code)]

pub trait Provider {
    type Results: AsRef<[u64]> + Default;
}

pub fn generic_nonempty<P: Provider>() -> u64 {
    let results = P::Results::default();
    unsafe { *results.as_ref().get_unchecked(0) }
}

pub struct EmptyProvider;
impl Provider for EmptyProvider {
    type Results = [u64; 0];
}

pub trait ReprPair {
    type Source: Default;
    type Target;
}

pub fn generic_layout<P: ReprPair>() -> P::Target {
    let source = core::mem::ManuallyDrop::new(P::Source::default());
    let pointer = (&*source as *const P::Source).cast::<P::Target>();
    unsafe { pointer.read() }
}

fn layout_probe<S: Default, T>() -> T {
    let source = core::mem::ManuallyDrop::new(S::default());
    let pointer = (&*source as *const S).cast::<T>();
    unsafe { pointer.read() }
}

pub fn forwarded_generic_layout<P: ReprPair>() -> P::Target {
    layout_probe::<P::Source, P::Target>()
}

pub fn concrete_layout_only() -> u64 {
    layout_probe::<u64, u64>()
}

pub fn parametric_and_concrete<P: ReprPair>(choose: bool) -> P::Target {
    if choose {
        let _ = layout_probe::<u64, u64>();
    }
    layout_probe::<P::Source, P::Target>()
}

pub fn concrete_layout_violation() -> u64 {
    layout_probe::<u8, u64>()
}

fn pointer_probe<S, T>(source: &S) -> *const T {
    (source as *const S).cast::<T>()
}

pub fn cross_owner_generic_layout<P: ReprPair>() -> P::Target {
    let source = core::mem::ManuallyDrop::new(P::Source::default());
    let pointer = pointer_probe::<P::Source, P::Target>(&source);
    unsafe { pointer.read() }
}

pub fn cross_owner_unmatched_sibling<P: ReprPair>() -> u64 {
    let source = core::mem::ManuallyDrop::new(1_u64);
    let used = pointer_probe::<u64, u64>(&source);
    let unrelated = core::mem::ManuallyDrop::new(P::Source::default());
    let _unused = pointer_probe::<P::Source, P::Target>(&unrelated);
    unsafe { used.read() }
}

// The associated-type equality leaves no independent layout assignment for
// the direct helper parameters. A bounded transport certificate must not turn
// this into the `Source = u8, Target = u64` witness used by the positive case.
pub fn cross_owner_equal_projection<P, T: Default>() -> P::Target
where
    P: ReprPair<Source = T, Target = T>,
{
    let source = core::mem::ManuallyDrop::new(P::Source::default());
    let pointer = pointer_probe::<P::Source, P::Target>(&source);
    unsafe { pointer.read() }
}

// This caller strengthens the associated source projection. The bounded
// certificate accepts only one plain `P: ReprPair` predicate, so the extra
// bound remains Unknown rather than being silently normalized away.
pub fn cross_owner_strengthened_projection<P>() -> P::Target
where
    P: ReprPair,
    P::Source: Copy,
{
    let source = core::mem::ManuallyDrop::new(P::Source::default());
    let pointer = pointer_probe::<P::Source, P::Target>(&source);
    unsafe { pointer.read() }
}

pub fn fixed_nonempty() -> u64 {
    let values = [1_u64];
    unsafe { *values.get_unchecked(0) }
}

pub fn same_pointee_view<T: Copy>(pointer: *mut T) -> T {
    let read_only = pointer as *const T;
    unsafe { read_only.read() }
}

pub fn slice_element_view<T: Copy>(values: &[T]) -> T {
    let element = values as *const [T] as *const T;
    unsafe { element.read() }
}

pub fn maybe_uninit_view<T: Copy>(value: &core::mem::MaybeUninit<T>) -> T {
    let pointer = value.as_ptr();
    unsafe { pointer.read() }
}

#[inline(never)]
fn const_u32_value<const N: u32>() -> u32 {
    N
}

pub fn non_usize_const_dispatch() -> u32 {
    const_u32_value::<7>()
}
