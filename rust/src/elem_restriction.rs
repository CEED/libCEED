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

    pub fn apply_block(
        &self,
        block: i32,
        tmode: crate::TransposeMode,
        u: &crate::vector::Vector,
        ru: &mut crate::vector::Vector,
    ) {
        unsafe {
            bind_ceed::CeedElemRestrictionApplyBlock(
                self.ptr,
                block,
                tmode as bind_ceed::CeedTransposeMode,
                u.ptr,
                ru.ptr,
                bind_ceed::CEED_REQUEST_IMMEDIATE,
            )
        };
    }

    pub fn get_comp_stride(
        &self,
    ) -> i32 {
        let mut compstride = 0;
        unsafe {
            bind_ceed::CeedElemRestrictionGetCompStride(
                self.ptr,
                &mut compstride,
            )
        };
        compstride
    }

    pub fn get_num_elements(
        &self,
    ) -> i32 {
        let mut numelem = 0;
        unsafe {
            bind_ceed::CeedElemRestrictionGetNumElements(
                self.ptr,
                &mut numelem,
            )
        };
        numelem
    }

    pub fn get_elem_size(
        &self,
    ) -> i32 {
        let mut elemsize = 0;
        unsafe {
            bind_ceed::CeedElemRestrictionGetElementSize(
                self.ptr,
                &mut elemsize,
            )
        };
        elemsize
    }

    pub fn get_Lvector_size(
        &self,
    ) -> i32 {
        let mut lsize = 0;
        unsafe {
            bind_ceed::CeedElemRestrictionGetLVectorSize(
                self.ptr,
                &mut lsize,
            )
        };
        lsize
    }

    pub fn get_num_components(
        &self,
    ) -> i32 {
        let mut numcomp = 0;
        unsafe {
            bind_ceed::CeedElemRestrictionGetNumComponents(
                self.ptr,
                &mut numcomp,
            )
        };
        numcomp
    }

    pub fn get_num_blocks(
        &self,
    ) -> i32 {
        let mut numblock = 0;
        unsafe {
            bind_ceed::CeedElemRestrictionGetNumBlocks(
                self.ptr,
                &mut numblock,
            )
        };
        numblock
    }

    pub fn get_block_size(
        &self,
    ) -> i32 {
        let mut blksize = 0;
        unsafe {
            bind_ceed::CeedElemRestrictionGetBlockSize(
                self.ptr,
                &mut blksize,
            )
        };
        blksize
    }

    pub fn get_multiplicity(
        &self,
        mult: &mut crate::vector::Vector,
    ) {
        unsafe {
            bind_ceed::CeedElemRestrictionGetMultiplicity(
                self.ptr,
                mult.ptr,
            )
        };
    }
}

/// Destructor
impl<'a> Drop for ElemRestriction<'a> {
    fn drop(&mut self) {
        unsafe {
            bind_ceed::CeedElemRestrictionDestroy(&mut self.ptr);
        }
    }
}
