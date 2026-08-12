use core::fmt::{self, Write};

pub struct WriteValue {
    pub value: u32,
}

impl WriteValue {
    pub fn write_public_value(&self, output: &mut String) -> fmt::Result {
        write!(output, "{}", self.value)
    }
}

