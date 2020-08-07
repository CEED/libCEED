use crate::prelude::*;

pub struct Operator<'a> {
    ceed: &'a crate::Ceed,
    ptr: bind_ceed::CeedOperator,
}
impl<'a> Operator<'a> {
    pub fn create(
        ceed: &'a crate::Ceed,
        qf: &crate::qfunction::QFunction,
        dqf: &crate::qfunction::QFunction,
        dqfT: &crate::qfunction::QFunction,
    ) -> Self {
        let mut ptr = std::ptr::null_mut();
        pub use crate::qfunction::QFunction;
        unsafe { bind_ceed::CeedOperatorCreate(ceed.ptr, qf.ptr, dqf.ptr, dqfT.ptr, &mut ptr) };
        Self { ceed, ptr }
    }
}
