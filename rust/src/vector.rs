use crate::prelude::*;

/// CeedVector context wrapper
pub struct Vector<'a> {
    ceed: &'a crate::Ceed,
    ptr: bind_ceed::CeedVector,
}
pub enum NormType {
    One,
    Two,
    Max,
}
impl<'a> Vector<'a> {
    pub fn create(ceed: &'a crate::Ceed, n: i32) -> Self {
        let mut ptr = std::ptr::null_mut();
        unsafe { bind_ceed::CeedVectorCreate(ceed.ptr, n, &mut ptr) };
        Self { ceed, ptr }
    }
    
    pub fn new(ceed: &'a crate::Ceed, ptr: bind_ceed::CeedVector) -> Self {
        Self { ceed , ptr}
    }
    
    pub fn length(&self) -> i32 {
        let mut n = 0;
        unsafe { bind_ceed::CeedVectorGetLength(self.ptr, &mut n) };
        n
    }
    
    pub fn set_value(&mut self, value: f64) {
        unsafe { bind_ceed::CeedVectorSetValue(self.ptr, value) };
    }
    
    pub fn sync(&self, mtype: crate::MemType) {
      unsafe { bind_ceed::CeedVectorSyncArray(self.ptr, mtype as bind_ceed::CeedMemType) };
    }
    
    pub fn norm(&self, ntype: NormType)  -> f64 {
      let mut res :f64 = 0.0;
      unsafe{ bind_ceed::CeedVectorNorm(self.ptr, ntype as bind_ceed::CeedNormType, &mut res) };
      res
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
