use crate::prelude::*;
use std::mem;

pub struct Operator<'a> {
  ceed : &'a crate::Ceed,
  ptr : bind_ceed::CeedOperator,
}
impl<'a> Operator<'a> {
  pub fn create(ceed: &'a crate::Ceed, qf: &crate::qfunction::QFunction,
  dqf: &crate::qfunction::QFunction, dqfT: &crate::qfunction::QFunction)
  -> Self {
    let mut ptr = unsafe {libc::malloc(mem::size_of::<bind_ceed::CeedOperator>()) as bind_ceed::CeedOperator};
    pub use crate::qfunction::QFunction;
    unsafe { bind_ceed::CeedOperatorCreate(ceed.ptr,qf.ptr,dqf.ptr,dqfT.ptr, &mut ptr) };
    Self {ceed, ptr}
  }
}