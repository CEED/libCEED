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
            &mut ptr)
        };
        Self { ceed, ptr }
    }

    pub fn create_strided(
        ceed: &'a crate::Ceed,
        nelem: i32,
        elemsize: i32,
        ncomp: i32,
        lsize: i32,
        strides: [i32; 3],
    ) -> Self {
        let mut ptr = std::ptr::null_mut();
        unsafe {
            bind_ceed::CeedElemRestrictionCreateStrided(
                ceed.ptr,
                nelem,
                elemsize,
                ncomp,
                lsize,
                strides.as_ptr(),
                &mut ptr)
        };
        Self { ceed, ptr}
    }

    pub fn create_blocked(
        ceed: &'a crate::Ceed,
        nelem: i32,
        elemsize: i32,
        blksize: i32,
        ncomp: i32,
        compstride: i32,
        lsize: i32,
        mtype: crate::MemType,
        cmode: crate::CopyMode,
        offsets: &Vec<i32>,
    ) -> Self {
        let mut ptr = std::ptr::null_mut();
        unsafe {
            bind_ceed::CeedElemRestrictionCreateBlocked(
                ceed.ptr,
                nelem,
                elemsize,
                blksize,
                ncomp,
                compstride,
                lsize,
                mtype as bind_ceed::CeedMemType,
                cmode as bind_ceed::CeedCopyMode,
                offsets.as_ptr(),
                &mut ptr)
        };
        Self { ceed, ptr }
    }

    pub fn create_blocked_strided(
        ceed: &'a crate::Ceed,
        nelem: i32,
        elemsize: i32,
        blksize: i32,
        ncomp: i32,
        lsize: i32,
        strides: [i32; 3],
    ) -> Self {
        let mut ptr = std::ptr::null_mut();
        unsafe {
            bind_ceed::CeedElemRestrictionCreateBlockedStrided(
                ceed.ptr,
                nelem,
                elemsize,
                blksize,
                ncomp,
                lsize,
                strides.as_ptr(),
                &mut ptr,
            )
        };
        Self { ceed,ptr }
    }

    pub fn apply(
        &self,
        tmode: crate::TransposeMode,
        u: &crate::vector::Vector,
        ru: &mut crate::vector::Vector,
    ) {
        unsafe {
            bind_ceed::CeedElemRestrictionApply(
                self.ptr,
                tmode as bind_ceed::CeedTransposeMode,
                u.ptr,
                ru.ptr,
                bind_ceed::CEED_REQUEST_IMMEDIATE,
            )
        };
    }
}
