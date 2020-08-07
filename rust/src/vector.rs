use crate::prelude::*;
use std::mem;

/// CeedVector context wrapper
pub struct Vector<'a> {
  ceed : &'a crate::Ceed,
  ptr : bind_ceed::CeedVector,
}
pub enum NormType {

}
impl<'a> Vector<'a> {
  pub fn create(ceed: &'a crate::Ceed, n: i32) -> Self {
    let mut ptr = unsafe { libc::malloc(mem::size_of::<bind_ceed::CeedVector>()) as bind_ceed::CeedVector };
    unsafe { bind_ceed::CeedVectorCreate(ceed.ptr, n, &mut ptr) };
    Self { ceed , ptr}
  }

  pub fn new(ceed: &'a crate::Ceed, ptr: bind_ceed::CeedVector) -> Self {
    Self { ceed , ptr}
  }

  pub fn length(&self) -> i32 {
    let mut n = 0;
    unsafe { bind_ceed::CeedVectorGetLength(self.ptr, &mut n) };
    n
  }

  pub fn set_value(&mut self, value : f64) {
    unsafe { bind_ceed::CeedVectorSetValue(self.ptr, value) };
  }

  // pub fn sync(&self, mtype: crate::MemType) {
  //   unsafe { bind_ceed::CeedVectorSyncArray(self.ptr, mtype) };
  // }

  // pub fn norm(&self, type: NormType)  -> f64 {
  //   let res :f64 = 0.0;
  //   unsafe{ bind_ceed::CeedVectorNorm(self.ptr, type, &mut res) };
  //   res
  // }
}

/// Destructor
impl<'a> Drop for Vector<'a> {
  fn drop(&mut self) {
    unsafe {
      bind_ceed::CeedVectorDestroy(&mut self.ptr);
    }
  }
}
