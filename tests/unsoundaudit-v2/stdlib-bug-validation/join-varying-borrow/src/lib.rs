use std::borrow::Borrow;
use std::cell::Cell;
use std::hint::black_box;

pub struct ChangingBorrow {
    calls: Cell<usize>,
    first: &'static str,
    later: &'static str,
}

impl ChangingBorrow {
    pub const fn new(first: &'static str, later: &'static str) -> Self {
        Self {
            calls: Cell::new(0),
            first,
            later,
        }
    }
}

impl Borrow<str> for ChangingBorrow {
    fn borrow(&self) -> &str {
        let calls = self.calls.get();
        self.calls.set(calls + 1);
        if calls == 0 {
            self.first
        } else {
            self.later
        }
    }
}

pub fn join_growing_first() {
    let values = [ChangingBorrow::new("", "0123456789abcdef")];
    black_box(values.join(""));
}

pub fn concat_growing_first() {
    let values = [ChangingBorrow::new("", "0123456789abcdef")];
    black_box(values.concat());
}

pub fn join_growing_later() {
    let values = [
        ChangingBorrow::new("a", "a"),
        ChangingBorrow::new("", "0123456789abcdef"),
    ];
    black_box(values.join(""));
}

pub fn join_growing_first_nonempty_separator() {
    let values = [
        ChangingBorrow::new("", "0123456789abcdef"),
        ChangingBorrow::new("x", "x"),
    ];
    black_box(values.join("-"));
}

pub fn join_shrinking_first() {
    let values = [ChangingBorrow::new("0123456789abcdef", "")];
    black_box(values.join(""));
}

pub fn stable_control() {
    let values = [ChangingBorrow::new("abc", "abc")];
    assert_eq!(values.join(""), "abc");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn changing_borrow_is_short_then_long() {
        let value = ChangingBorrow::new("", "long");
        assert_eq!(Borrow::<str>::borrow(&value), "");
        assert_eq!(Borrow::<str>::borrow(&value), "long");
    }

    #[test]
    fn stable_control_is_safe_to_execute() {
        stable_control();
    }

    #[test]
    fn stable_join_and_concat_borrow_each_element_twice() {
        let joined = [ChangingBorrow::new("abc", "abc")];
        assert_eq!(joined.join(""), "abc");
        assert_eq!(joined[0].calls.get(), 2);

        let concatenated = [ChangingBorrow::new("abc", "abc")];
        assert_eq!(concatenated.concat(), "abc");
        assert_eq!(concatenated[0].calls.get(), 2);
    }
}
