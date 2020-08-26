// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative
use crate::prelude::*;
use std::ffi::CString;
use std::fmt;

// -----------------------------------------------------------------------------
// CeedElemRestriction context wrapper
// -----------------------------------------------------------------------------
pub struct ElemRestriction<'a> {
    ceed: &'a crate::Ceed,
    pub ptr: bind_ceed::CeedElemRestriction,
}

// -----------------------------------------------------------------------------
// Destructor
// -----------------------------------------------------------------------------
impl<'a> Drop for ElemRestriction<'a> {
    fn drop(&mut self) {
        unsafe {
            bind_ceed::CeedElemRestrictionDestroy(&mut self.ptr);
        }
    }
}

// -----------------------------------------------------------------------------
// Display
// -----------------------------------------------------------------------------
impl<'a> fmt::Display for ElemRestriction<'a> {
    /// View a Basis
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let b = ceed.basis_tensor_H1_Lagrange(1, 2, 3, 4, ceed::QuadMode::Gauss);
    /// println!("{}", b);
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut ptr = std::ptr::null_mut();
        let mut sizeloc = 202020;
        unsafe {
            let file = bind_ceed::open_memstream(&mut ptr, &mut sizeloc);
            bind_ceed::CeedElemRestrictionView(self.ptr, file);
            bind_ceed::fclose(file);
            let cstring = CString::from_raw(ptr);
            let s = cstring.to_string_lossy().into_owned();
            write!(f, "{}", s)
        }
    }
}

// -----------------------------------------------------------------------------
// Implementations
// -----------------------------------------------------------------------------
impl<'a> ElemRestriction<'a> {
    // Constructors
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

// -----------------------------------------------------------------------------
