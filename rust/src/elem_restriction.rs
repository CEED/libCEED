use crate::prelude::*;

/// CeedElemRestriction context wrapper
pub struct ElemRestriction<'a> {
    ceed: &'a crate::Ceed,
    pub ptr: bind_ceed::CeedElemRestriction,
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
                &mut ptr,
            )
        };
        Self { ceed, ptr }
    }

    /// I think these should be excluded for now - they're intended for backneds
    /*
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
                &mut ptr,
            )
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
        Self { ceed, ptr }
    }
    */

    /// Restrict an L-vector to an E-vector or apply its transpose
    ///
    /// # arguments
    ///
    /// * 'tmode' - Apply restriction or transpose
    /// * 'u'     - Input vector (of size lsize when tmode=NoTranspose)
    /// * 'ru'    - Output vector (of shape [nelem * elemsize] when
    ///               tmode=NoTranspose). Ordering of the e-vector is decided
    ///               by the backend.
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let ne = 3;
    /// let mut ind : Vec<i32> = vec![0; 2*ne];
    /// for i in 0..ne {
    ///   ind[2*i+0] = i as i32;
    ///   ind[2*i+1] = (i+1) as i32;
    /// }
    /// let r = ceed.elem_restriction(ne as i32, 2, 1, 1, (ne+1) as i32, ceed::MemType::Host,
    ///                               ceed::CopyMode::CopyValues, &ind);
    ///
    /// let x = ceed.vector_from_slice(&[0., 1., 2., 3.]);
    /// let mut y = ceed.vector(ne*2);
    /// y.set_value(0.0);
    ///
    /// r.apply(ceed::TransposeMode::NoTranspose, &x, &mut y);
    ///
    /// let array = y.get_array_read(ceed::MemType::Host);
    /// for i in 0..(ne*2) {
    ///   assert_eq!(array[i], ((i+1)/2) as f64);
    /// }
    /// y.restore_array_read(array);
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

    /// See above
    /*
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
    */

    /// Returns the L-vector component stride
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let ne = 3;
    /// let mut ind : Vec<i32> = vec![0; 2*ne];
    /// for i in 0..ne {
    ///   ind[2*i+0] = i as i32;
    ///   ind[2*i+1] = (i+1) as i32;
    /// }
    /// let r = ceed.elem_restriction(ne as i32, 2, 1, 1, (ne+1) as i32, ceed::MemType::Host,
    ///                               ceed::CopyMode::CopyValues, &ind);
    ///
    /// let compstride = r.get_comp_stride();
    /// assert_eq!(compstride, 1);
    /// ```
    pub fn get_comp_stride(&self) -> i32 {
        let mut compstride = 0;
        unsafe { bind_ceed::CeedElemRestrictionGetCompStride(self.ptr, &mut compstride) };
        compstride
    }

    /// Returns the total number of elements in the range of a ElemRestriction
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let ne = 3;
    /// let mut ind : Vec<i32> = vec![0; 2*ne];
    /// for i in 0..ne {
    ///   ind[2*i+0] = i as i32;
    ///   ind[2*i+1] = (i+1) as i32;
    /// }
    /// let r = ceed.elem_restriction(ne as i32, 2, 1, 1, (ne+1) as i32, ceed::MemType::Host,
    ///                               ceed::CopyMode::CopyValues, &ind);
    ///
    /// let nelem = r.get_num_elements();
    /// assert_eq!(nelem, ne as i32);
    /// ```
    pub fn get_num_elements(&self) -> i32 {
        let mut numelem = 0;
        unsafe { bind_ceed::CeedElemRestrictionGetNumElements(self.ptr, &mut numelem) };
        numelem
    }

    /// Returns the size of elements in the ElemRestriction
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let ne = 3;
    /// let mut ind : Vec<i32> = vec![0; 2*ne];
    /// for i in 0..ne {
    ///   ind[2*i+0] = i as i32;
    ///   ind[2*i+1] = (i+1) as i32;
    /// }
    /// let r = ceed.elem_restriction(ne as i32, 2, 1, 1, (ne+1) as i32, ceed::MemType::Host,
    ///                               ceed::CopyMode::CopyValues, &ind);
    ///
    /// let esize = r.get_elem_size();
    /// assert_eq!(esize, 2);
    /// ```
    pub fn get_elem_size(&self) -> i32 {
        let mut elemsize = 0;
        unsafe { bind_ceed::CeedElemRestrictionGetElementSize(self.ptr, &mut elemsize) };
        elemsize
    }

    /// Returns the size of the l-vector for an ElemRestriction
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let ne = 3;
    /// let mut ind : Vec<i32> = vec![0; 2*ne];
    /// for i in 0..ne {
    ///   ind[2*i+0] = i as i32;
    ///   ind[2*i+1] = (i+1) as i32;
    /// }
    /// let r = ceed.elem_restriction(ne as i32, 2, 1, 1, (ne+1) as i32, ceed::MemType::Host,
    ///                               ceed::CopyMode::CopyValues, &ind);
    ///
    /// let lsize = r.get_Lvector_size();
    /// assert_eq!(lsize, (ne+1) as i32);
    /// ```
    pub fn get_Lvector_size(&self) -> i32 {
        let mut lsize = 0;
        unsafe { bind_ceed::CeedElemRestrictionGetLVectorSize(self.ptr, &mut lsize) };
        lsize
    }

    /// Returns the number of components in the elements of an ElemRestriction
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let ne = 3;
    /// let mut ind : Vec<i32> = vec![0; 2*ne];
    /// for i in 0..ne {
    ///   ind[2*i+0] = i as i32;
    ///   ind[2*i+1] = (i+1) as i32;
    /// }
    /// let r = ceed.elem_restriction(ne as i32, 2, 42, 1, 42*(ne+1) as i32, ceed::MemType::Host,
    ///                               ceed::CopyMode::CopyValues, &ind);
    ///
    /// let ncomp = r.get_num_components();
    /// assert_eq!(ncomp, 42);
    /// ```
    pub fn get_num_components(&self) -> i32 {
        let mut numcomp = 0;
        unsafe { bind_ceed::CeedElemRestrictionGetNumComponents(self.ptr, &mut numcomp) };
        numcomp
    }

    /// See above
    /*
    pub fn get_num_blocks(&self) -> i32 {
        let mut numblock = 0;
        unsafe { bind_ceed::CeedElemRestrictionGetNumBlocks(self.ptr, &mut numblock) };
        numblock
    }

    pub fn get_block_size(&self) -> i32 {
        let mut blksize = 0;
        unsafe { bind_ceed::CeedElemRestrictionGetBlockSize(self.ptr, &mut blksize) };
        blksize
    }
    */

    /// Returns the multiplicity oof nodes in an ElemRestriction
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let ne = 3;
    /// let mut ind : Vec<i32> = vec![0; 2*ne];
    /// for i in 0..ne {
    ///   ind[2*i+0] = i as i32;
    ///   ind[2*i+1] = (i+1) as i32;
    /// }
    /// let r = ceed.elem_restriction(ne as i32, 2, 1, 1, (ne+1) as i32, ceed::MemType::Host,
    ///                               ceed::CopyMode::CopyValues, &ind);
    ///
    /// let mut mult = ceed.vector(ne+1);
    /// mult.set_value(0.0);
    ///
    /// r.get_multiplicity(&mut mult);
    ///
    /// let array = mult.get_array_read(ceed::MemType::Host);
    /// for i in 0..(ne+1) {
    ///   assert_eq!(if (i == 0 || i == ne) { 1. } else { 2. }, array[i]);
    /// }
    /// mult.restore_array_read(array);
    /// ```
    pub fn get_multiplicity(&self, mult: &mut crate::vector::Vector) {
        unsafe { bind_ceed::CeedElemRestrictionGetMultiplicity(self.ptr, mult.ptr) };
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
