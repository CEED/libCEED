use crate::prelude::*;
use std::mem;

pub struct QFunction<'a> {
    ceed: &'a crate::Ceed,
    pub ptr: bind_ceed::CeedQFunction,
}
