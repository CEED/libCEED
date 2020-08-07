use crate::prelude::*;

/// CeedQFunction context wrapper
pub struct QFunction<'a> {
    ceed: &'a crate::Ceed,
    pub ptr: bind_ceed::CeedQFunction,
}

/// Destructor
impl<'a> Drop for QFunction<'a> {
    fn drop(&mut self) {
        unsafe {
            bind_ceed::CeedQFunctionDestroy(&mut self.ptr);
        }
    }
}
