use crate::prelude::*;

/// CeedOperator context wrapper
pub struct Operator<'a> {
    ceed: &'a crate::Ceed,
    ptr: bind_ceed::CeedOperator,
}

impl<'a> Operator<'a> {
    /// Constructor
    pub fn create(
        ceed: &'a crate::Ceed,
        qf: &crate::qfunction::QFunction,
        dqf: &crate::qfunction::QFunction,
        dqfT: &crate::qfunction::QFunction,
    ) -> Self {
        let mut ptr = std::ptr::null_mut();
        unsafe { bind_ceed::CeedOperatorCreate(ceed.ptr, qf.ptr, dqf.ptr, dqfT.ptr, &mut ptr) };
        Self { ceed, ptr }
    }
}

/// Destructor
impl<'a> Drop for Operator<'a> {
    fn drop(&mut self) {
        unsafe {
            bind_ceed::CeedOperatorDestroy(&mut self.ptr);
        }
    }
}
