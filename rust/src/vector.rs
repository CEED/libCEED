use crate::prelude::*;
use std::mem;

/// CeedVector context wrapper
pub struct Vector<'a> {
  ceed_reference : &'a crate::Ceed,
  ceed_vec_ptr : bind_ceed::CeedVector,
}
impl<'a> Vector<'a> {
  pub fn create(ceed_reference: &'a crate::Ceed, n: i32) -> Self {
    let mut ceed_vec_ptr = unsafe { libc::malloc(mem::size_of::<bind_ceed::CeedVector>()) as bind_ceed::CeedVector };
    unsafe { bind_ceed::CeedVectorCreate(ceed_reference.ceed_ptr, n, &mut ceed_vec_ptr) };
    Self { ceed_reference , ceed_vec_ptr}
  }

  pub fn new(ceed_reference: &'a crate::Ceed, ceed_vec_ptr: bind_ceed::CeedVector) -> Self {
    Self { ceed_reference , ceed_vec_ptr}
  }
}

/// Destructor
impl<'a> Drop for Vector<'a> {
  fn drop(&mut self) {
    unsafe {
      bind_ceed::CeedVectorDestroy(&mut self.ceed_vec_ptr);
    }
  }
}
