#![allow(dead_code)]

struct Droppable;

impl Drop for Droppable {
    fn drop(&mut self) {}
}

fn explicit_drop(mut value: Droppable) {
    Drop::drop(&mut value);
}
