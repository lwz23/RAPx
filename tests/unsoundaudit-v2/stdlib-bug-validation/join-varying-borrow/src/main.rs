use join_varying_borrow_poc::{
    concat_growing_first, join_growing_first, join_growing_first_nonempty_separator,
    join_growing_later, join_shrinking_first, stable_control,
};

fn main() {
    assert_eq!(
        std::env::var("UNSOUNDAUDIT_STDLIB_POC_ACK").as_deref(),
        Ok("1"),
        "refusing to run a potential UB PoC without explicit acknowledgement",
    );
    match std::env::args().nth(1).as_deref() {
        Some("stable-control") => stable_control(),
        Some("join-growing-first") => join_growing_first(),
        Some("concat-growing-first") => concat_growing_first(),
        Some("join-growing-later") => join_growing_later(),
        Some("join-growing-first-nonempty-separator") => join_growing_first_nonempty_separator(),
        Some("join-shrinking-first") => join_shrinking_first(),
        other => panic!("unknown validation case: {other:?}"),
    }
}
