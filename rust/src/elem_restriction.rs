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
    pub(crate) ceed: &'a crate::Ceed,
    pub(crate) ptr: bind_ceed::CeedElemRestriction,
}

// -----------------------------------------------------------------------------
// Destructor
// -----------------------------------------------------------------------------
impl<'a> Drop for ElemRestriction<'a> {
    fn drop(&mut self) {
        unsafe {
            if self.ptr != bind_ceed::CEED_ELEMRESTRICTION_NONE {
                bind_ceed::CeedElemRestrictionDestroy(&mut self.ptr);
            }
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
    /// let nelem = 3;
    /// let mut ind: Vec<i32> = vec![0; 2*nelem];
    /// for i in 0..nelem {
    ///   ind[2*i+0] = i as i32;
    ///   ind[2*i+1] = (i+1) as i32;
    /// }
    /// let r = ceed.elem_restriction(nelem, 2, 1, 1, nelem+1, ceed::MemType::Host,
    ///                               ceed::CopyMode::CopyValues, &ind);
    /// println!("{}", r);
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut ptr = std::ptr::null_mut();
        let mut sizeloc = crate::max_buffer_length;
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
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        compstride: usize,
        lsize: usize,
        mtype: crate::MemType,
        cmode: crate::CopyMode,
        offsets: &Vec<i32>,
    ) -> Self {
        let mut ptr = std::ptr::null_mut();
        unsafe {
            bind_ceed::CeedElemRestrictionCreate(
                ceed.ptr,
                nelem as i32,
                elemsize as i32,
                ncomp as i32,
                compstride as i32,
                lsize as i32,
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
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        lsize: usize,
        strides: [i32; 3],
    ) -> Self {
        let mut ptr = std::ptr::null_mut();
        unsafe {
            bind_ceed::CeedElemRestrictionCreateStrided(
                ceed.ptr,
                nelem as i32,
                elemsize as i32,
                ncomp as i32,
                lsize as i32,
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
    /// let nelem = 3;
    /// let mut ind: Vec<i32> = vec![0; 2*nelem];
    /// for i in 0..nelem {
    ///   ind[2*i+0] = i as i32;
    ///   ind[2*i+1] = (i+1) as i32;
    /// }
    /// let r = ceed.elem_restriction(nelem, 2, 1, 1, nelem+1, ceed::MemType::Host,
    ///                               ceed::CopyMode::CopyValues, &ind);
    ///
    /// let x = ceed.vector_from_slice(&[0., 1., 2., 3.]);
    /// let mut y = ceed.vector(nelem*2);
    /// y.set_value(0.0);
    ///
    /// r.apply(ceed::TransposeMode::NoTranspose, &x, &mut y);
    ///
    /// let array = y.view();
    /// for i in 0..(nelem*2) {
    ///   assert_eq!(array[i], ((i+1)/2) as f64, "Incorrect value in restricted vector");
    /// }
    /// ```
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
    /// let nelem = 3;
    /// let compstride = 1;
    /// let mut ind: Vec<i32> = vec![0; 2*nelem];
    /// for i in 0..nelem {
    ///   ind[2*i+0] = i as i32;
    ///   ind[2*i+1] = (i+1) as i32;
    /// }
    /// let r = ceed.elem_restriction(nelem, 2, 1, compstride, nelem+1, ceed::MemType::Host,
    ///                               ceed::CopyMode::CopyValues, &ind);
    ///
    /// let c = r.get_comp_stride();
    /// assert_eq!(c, compstride as i32, "Incorrect component stride");
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
    /// let nelem = 3;
    /// let mut ind: Vec<i32> = vec![0; 2*nelem];
    /// for i in 0..nelem {
    ///   ind[2*i+0] = i as i32;
    ///   ind[2*i+1] = (i+1) as i32;
    /// }
    /// let r = ceed.elem_restriction(nelem, 2, 1, 1, nelem+1, ceed::MemType::Host,
    ///                               ceed::CopyMode::CopyValues, &ind);
    ///
    /// let n = r.get_num_elements();
    /// assert_eq!(n, nelem as i32, "Incorrect number of elements");
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
    /// let nelem = 3;
    /// let elem_size = 2;
    /// let mut ind: Vec<i32> = vec![0; 2*nelem];
    /// for i in 0..nelem {
    ///   ind[2*i+0] = i as i32;
    ///   ind[2*i+1] = (i+1) as i32;
    /// }
    /// let r = ceed.elem_restriction(nelem, elem_size, 1, 1, nelem+1, ceed::MemType::Host,
    ///                               ceed::CopyMode::CopyValues, &ind);
    ///
    /// let e = r.get_elem_size();
    /// assert_eq!(e, elem_size as i32, "Incorrect element size");
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
    /// let nelem = 3;
    /// let mut ind : Vec<i32> = vec![0; 2*nelem];
    /// for i in 0..nelem {
    ///   ind[2*i+0] = i as i32;
    ///   ind[2*i+1] = (i+1) as i32;
    /// }
    /// let r = ceed.elem_restriction(nelem, 2, 1, 1, nelem+1, ceed::MemType::Host,
    ///                               ceed::CopyMode::CopyValues, &ind);
    ///
    /// let lsize = r.get_Lvector_size();
    /// assert_eq!(lsize, (nelem+1) as i32);
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
    /// let nelem = 3;
    /// let ncomp = 42;
    /// let mut ind: Vec<i32> = vec![0; 2*nelem];
    /// for i in 0..nelem {
    ///   ind[2*i+0] = i as i32;
    ///   ind[2*i+1] = (i+1) as i32;
    /// }
    /// let r = ceed.elem_restriction(nelem, 2, 42, 1, ncomp*(nelem+1), ceed::MemType::Host,
    ///                               ceed::CopyMode::CopyValues, &ind);
    ///
    /// let n = r.get_num_components();
    /// assert_eq!(n, ncomp as i32, "Incorrect number of components");
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
    /// let nelem = 3;
    /// let mut ind: Vec<i32> = vec![0; 2*nelem];
    /// for i in 0..nelem {
    ///   ind[2*i+0] = i as i32;
    ///   ind[2*i+1] = (i+1) as i32;
    /// }
    /// let r = ceed.elem_restriction(nelem, 2, 1, 1, nelem+1, ceed::MemType::Host,
    ///                               ceed::CopyMode::CopyValues, &ind);
    ///
    /// let mut mult = ceed.vector(nelem+1);
    /// mult.set_value(0.0);
    ///
    /// r.get_multiplicity(&mut mult);
    ///
    /// let array = mult.view();
    /// for i in 0..(nelem+1) {
    ///   assert_eq!(if (i == 0 || i == nelem) { 1. } else { 2. }, array[i], "Incorrect multiplicity array");
    /// }
    /// ```
    pub fn get_multiplicity(&self, mult: &mut crate::vector::Vector) {
        unsafe { bind_ceed::CeedElemRestrictionGetMultiplicity(self.ptr, mult.ptr) };
    }
}

// -----------------------------------------------------------------------------
