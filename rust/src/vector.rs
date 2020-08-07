use crate::prelude::*;
use std::mem;

/// CeedVector context wrapper
pub struct Vector<'a> {
    ceed: &'a crate::Ceed,
    ptr: bind_ceed::CeedVector,
}
impl<'a> Vector<'a> {
    pub fn create(ceed: &'a crate::Ceed, n: i32) -> Self {
        let mut ptr = unsafe {
            libc::malloc(mem::size_of::<bind_ceed::CeedVector>()) as bind_ceed::CeedVector
        };
        unsafe { bind_ceed::CeedVectorCreate(ceed.ptr, n, &mut ptr) };
        Self { ceed, ptr }
    }

    pub fn new(ceed: &'a crate::Ceed, ptr: bind_ceed::CeedVector) -> Self {
        Self { ceed, ptr }
    }
}

/// Destructor
impl<'a> Drop for Vector<'a> {
    fn drop(&mut self) {
        unsafe {
            bind_ceed::CeedVectorDestroy(&mut self.ptr);
        }
    }
}
