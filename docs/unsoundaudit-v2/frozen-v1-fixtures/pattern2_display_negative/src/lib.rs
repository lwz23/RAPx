use core::fmt;

pub struct DisplayValue {
    pub value: String,
}

impl fmt::Display for DisplayValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.value)
    }
}

