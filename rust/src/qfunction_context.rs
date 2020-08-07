use crate::prelude::*;

/// CeedQFunctionContext context wrapper
pub struct QFunctionContext<'a> {
    ceed: &'a crate::Ceed,
    pub ptr: bind_ceed::CeedQFunctionContext,
}

impl<'a> QFunctionContext<'a> {
    /// Constructor
    pub fn create(ceed: &'a crate::Ceed) -> Self {
        let mut ptr = std::ptr::null_mut();
        unsafe { bind_ceed::CeedQFunctionContextCreate(ceed.ptr, &mut ptr) };
        Self { ceed, ptr }
    }
}

/// Destructor
impl<'a> Drop for QFunctionContext<'a> {
    fn drop(&mut self) {
        unsafe {
            bind_ceed::CeedQFunctionContextDestroy(&mut self.ptr);
        }
    }
}
