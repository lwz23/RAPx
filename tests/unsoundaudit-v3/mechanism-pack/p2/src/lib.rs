#![allow(dead_code, unused_assignments)]

static ZERO: u8 = 0;
static ONE: u8 = 1;

extern "C" {
    fn opaque_consume(pointer: *const u8);
}

pub struct PublicRaw {
    pub pointer: *const u8,
    private_pointer: *const u8,
}

mod hidden {
    pub struct HiddenRaw {
        pub pointer: *const u8,
    }
}

// A public field is not an externally controllable P2 state surface when its
// parent ADT has no caller-nameable exported path.
pub fn read_unnameable_public_field(value: &hidden::HiddenRaw) -> u8 {
    unsafe { *value.pointer }
}

impl PublicRaw {
    pub fn read_public(&self) -> u8 {
        unsafe { *self.pointer }
    }

    pub fn read_private(&self) -> u8 {
        unsafe { *self.private_pointer }
    }

    pub fn opaque_sink(&self) {
        unsafe { opaque_consume(self.pointer) }
    }

    pub fn overwritten(&self) -> u8 {
        let mut pointer = self.pointer;
        pointer = &ZERO;
        unsafe { *pointer }
    }

    pub fn field_overwritten(mut self) -> u8 {
        self.pointer = &ZERO;
        unsafe { *self.pointer }
    }

    pub fn field_overwritten_on_all_paths(mut self, choose: bool) -> u8 {
        if choose {
            self.pointer = &ZERO;
        } else {
            self.pointer = &ONE;
        }
        unsafe { *self.pointer }
    }

    pub fn field_overwritten_on_one_path(mut self, choose: bool) -> u8 {
        if choose {
            self.pointer = &ZERO;
        }
        unsafe { *self.pointer }
    }

    pub fn base_overwritten(mut self) -> u8 {
        self = Self {
            pointer: &ZERO,
            private_pointer: &ZERO,
        };
        unsafe { *self.pointer }
    }

    pub fn base_overwritten_on_one_path(mut self, choose: bool) -> u8 {
        if choose {
            self = Self {
                pointer: &ZERO,
                private_pointer: &ZERO,
            };
        }
        unsafe { *self.pointer }
    }
}

pub struct NestedRaw {
    pub inner: PublicRaw,
}

impl NestedRaw {
    pub fn ancestor_overwritten(mut self) -> u8 {
        self.inner = PublicRaw {
            pointer: &ZERO,
            private_pointer: &ZERO,
        };
        unsafe { *self.inner.pointer }
    }

    pub fn ancestor_overwritten_on_one_path(mut self, choose: bool) -> u8 {
        if choose {
            self.inner = PublicRaw {
                pointer: &ZERO,
                private_pointer: &ZERO,
            };
        }
        unsafe { *self.inner.pointer }
    }
}

pub enum PublicChoice {
    Pointer(*const u8),
    Empty,
}

impl PublicChoice {
    pub fn read_payload(&self) -> Option<u8> {
        match self {
            Self::Pointer(pointer) => Some(unsafe { **pointer }),
            Self::Empty => None,
        }
    }
}

pub struct LocalAggregate {
    pub pointer: *const u8,
}

pub fn read_local_aggregate(pointer: *const u8) -> u8 {
    let value = LocalAggregate { pointer };
    unsafe { *value.pointer }
}

#[inline(never)]
fn shared_sink(pointer: *const u8) -> u8 {
    unsafe { *pointer }
}

pub fn read_public_through_helper(value: &PublicRaw) -> u8 {
    shared_sink(value.pointer)
}

pub fn unrelated_constant_through_helper() -> u8 {
    shared_sink(&ZERO)
}

fn zero_pointer() -> *const u8 {
    &ZERO
}

pub struct PublicCallback {
    pub callback: fn() -> *const u8,
}

impl PublicCallback {
    pub fn overwritten(mut self) -> u8 {
        self.callback = zero_pointer;
        let pointer = (self.callback)();
        unsafe { *pointer }
    }

    pub fn base_overwritten(mut self) -> u8 {
        self = Self {
            callback: zero_pointer,
        };
        let pointer = (self.callback)();
        unsafe { *pointer }
    }
}

pub struct PrivateCallback {
    callback: fn() -> *const u8,
}

impl PrivateCallback {
    pub fn new() -> Self {
        Self {
            callback: zero_pointer,
        }
    }

    pub fn invoke(&self) -> u8 {
        let pointer = (self.callback)();
        unsafe { *pointer }
    }
}

pub struct LocalCallback {
    pub callback: fn() -> *const u8,
}

pub fn local_internal_callback() -> u8 {
    let value = LocalCallback {
        callback: zero_pointer,
    };
    let pointer = (value.callback)();
    unsafe { *pointer }
}

pub enum CallbackChoice {
    Call(fn() -> *const u8),
    Empty,
}

impl CallbackChoice {
    pub fn invoke(self) -> Option<u8> {
        match self {
            Self::Call(callback) => {
                let pointer = callback();
                Some(unsafe { *pointer })
            }
            Self::Empty => None,
        }
    }
}
