use crate::prelude::*;

pub struct ElemRestriction<'a> {
    ceed: &'a crate::Ceed,
    ptr: bind_ceed::CeedElemRestriction,
}
impl<'a> ElemRestriction<'a> {
    pub fn create(
        ceed: &'a crate::Ceed,
        nelem: i32,
        elemsize: i32,
        ncomp: i32,
        compstride: i32,
        lsize: i32,
        mtype: crate::MemType,
        cmode: crate::CopyMode,
        offsets: &Vec<i32>,
    ) -> Self {
        let mut ptr = std::ptr::null_mut();
        unsafe {
            bind_ceed::CeedElemRestrictionCreate(
                ceed.ptr,
                nelem,
                elemsize,
                ncomp,
                compstride,
                lsize,
                mtype as bind_ceed::CeedMemType,
                cmode as bind_ceed::CeedCopyMode,
                offsets.as_ptr(),
                &mut ptr,
            )
        };
        Self { ceed, ptr }
    }
}
