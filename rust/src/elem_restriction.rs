use crate::prelude::*;
use std::mem;

pub struct ElemRestriction<'a> {
  ceed : &'a crate::Ceed,
  ptr : bind_ceed::CeedElemRestriction,
}
impl<'a> ElemRestriction<'a> {
  pub fn create(ceed: &'a crate::Ceed, nelem : i32, elemsize : i32, ncomp : i32, 
    compstride : i32, lsize : i32, mtype : crate::MemType, cmode : crate::CopyMode,
    offsets : &Vec<i32>) -> Self {
    let mut ptr = unsafe {libc::malloc(mem::size_of::<bind_ceed::CeedElemRestriction>()) as bind_ceed::CeedElemRestriction};
    // unsafe { bind_ceed::CeedElemRestrictionCreate(ceed.ptr, nelem, elemsize, ncomp, compstride, lsize, mtype, cmode, offsets, &mut ptr) };
    Self { ceed, ptr }
  }
}