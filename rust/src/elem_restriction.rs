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

//! A Ceed ElemRestriction decomposes elements and groups the degrees of freedom
//! (dofs) according to the different elements they belong to.

use crate::prelude::*;

// -----------------------------------------------------------------------------
// CeedElemRestriction option
// -----------------------------------------------------------------------------
#[derive(Clone, Copy)]
pub enum ElemRestrictionOpt<'a> {
    Some(&'a ElemRestriction<'a>),
    None,
}
/// Construct a ElemRestrictionOpt reference from a ElemRestriction reference
impl<'a> From<&'a ElemRestriction<'_>> for ElemRestrictionOpt<'a> {
    fn from(restr: &'a ElemRestriction) -> Self {
        debug_assert!(restr.ptr != unsafe { bind_ceed::CEED_ELEMRESTRICTION_NONE });
        Self::Some(restr)
    }
}
impl<'a> ElemRestrictionOpt<'a> {
    /// Transform a Rust libCEED ElemRestrictionOpt into C libCEED
    /// CeedElemRestriction
    pub(crate) fn to_raw(self) -> bind_ceed::CeedElemRestriction {
        match self {
            Self::Some(restr) => restr.ptr,
            Self::None => unsafe { bind_ceed::CEED_ELEMRESTRICTION_NONE },
        }
    }
}

// -----------------------------------------------------------------------------
// CeedElemRestriction context wrapper
// -----------------------------------------------------------------------------
pub struct ElemRestriction<'a> {
    ceed: &'a crate::Ceed,
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
    /// View an ElemRestriction
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let nelem = 3;
    /// let mut ind: Vec<i32> = vec![0; 2 * nelem];
    /// for i in 0..nelem {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed
    ///     .elem_restriction(nelem, 2, 1, 1, nelem + 1, MemType::Host, &ind)
    ///     .unwrap();
    /// println!("{}", r);
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut ptr = std::ptr::null_mut();
        let mut sizeloc = crate::MAX_BUFFER_LENGTH;
        let cstring = unsafe {
            let file = bind_ceed::open_memstream(&mut ptr, &mut sizeloc);
            bind_ceed::CeedElemRestrictionView(self.ptr, file);
            bind_ceed::fclose(file);
            CString::from_raw(ptr)
        };
        cstring.to_string_lossy().fmt(f)
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
        offsets: &[i32],
    ) -> crate::Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let (nelem, elemsize, ncomp, compstride, lsize, mtype) = (
            i32::try_from(nelem).unwrap(),
            i32::try_from(elemsize).unwrap(),
            i32::try_from(ncomp).unwrap(),
            i32::try_from(compstride).unwrap(),
            i32::try_from(lsize).unwrap(),
            mtype as bind_ceed::CeedMemType,
        );
        let ierr = unsafe {
            bind_ceed::CeedElemRestrictionCreate(
                ceed.ptr,
                nelem,
                elemsize,
                ncomp,
                compstride,
                lsize,
                mtype,
                crate::CopyMode::CopyValues as bind_ceed::CeedCopyMode,
                offsets.as_ptr(),
                &mut ptr,
            )
        };
        ceed.check_error(ierr)?;
        Ok(Self { ceed, ptr })
    }

    pub fn create_strided(
        ceed: &'a crate::Ceed,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        lsize: usize,
        strides: [i32; 3],
    ) -> crate::Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let (nelem, elemsize, ncomp, lsize) = (
            i32::try_from(nelem).unwrap(),
            i32::try_from(elemsize).unwrap(),
            i32::try_from(ncomp).unwrap(),
            i32::try_from(lsize).unwrap(),
        );
        let ierr = unsafe {
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
        ceed.check_error(ierr)?;
        Ok(Self { ceed, ptr })
    }

    /// Create an Lvector for an ElemRestriction
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let nelem = 3;
    /// let mut ind: Vec<i32> = vec![0; 2 * nelem];
    /// for i in 0..nelem {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed
    ///     .elem_restriction(nelem, 2, 1, 1, nelem + 1, MemType::Host, &ind)
    ///     .unwrap();
    ///
    /// let lvector = r.create_lvector().unwrap();
    ///
    /// assert_eq!(lvector.length(), nelem + 1, "Incorrect Lvector size");
    /// ```
    pub fn create_lvector(&self) -> crate::Result<Vector> {
        let mut ptr_lvector = std::ptr::null_mut();
        let null = std::ptr::null_mut() as *mut _;
        let ierr =
            unsafe { bind_ceed::CeedElemRestrictionCreateVector(self.ptr, &mut ptr_lvector, null) };
        self.ceed.check_error(ierr)?;
        Vector::from_raw(self.ceed, ptr_lvector)
    }

    /// Create an Evector for an ElemRestriction
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let nelem = 3;
    /// let mut ind: Vec<i32> = vec![0; 2 * nelem];
    /// for i in 0..nelem {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed
    ///     .elem_restriction(nelem, 2, 1, 1, nelem + 1, MemType::Host, &ind)
    ///     .unwrap();
    ///
    /// let evector = r.create_evector().unwrap();
    ///
    /// assert_eq!(evector.length(), nelem * 2, "Incorrect Evector size");
    /// ```
    pub fn create_evector(&self) -> crate::Result<Vector> {
        let mut ptr_evector = std::ptr::null_mut();
        let null = std::ptr::null_mut() as *mut _;
        let ierr =
            unsafe { bind_ceed::CeedElemRestrictionCreateVector(self.ptr, null, &mut ptr_evector) };
        self.ceed.check_error(ierr)?;
        Vector::from_raw(self.ceed, ptr_evector)
    }

    /// Create Vectors for an ElemRestriction
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let nelem = 3;
    /// let mut ind: Vec<i32> = vec![0; 2 * nelem];
    /// for i in 0..nelem {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed
    ///     .elem_restriction(nelem, 2, 1, 1, nelem + 1, MemType::Host, &ind)
    ///     .unwrap();
    ///
    /// let (lvector, evector) = r.create_vectors().unwrap();
    ///
    /// assert_eq!(lvector.length(), nelem + 1, "Incorrect Lvector size");
    /// assert_eq!(evector.length(), nelem * 2, "Incorrect Evector size");
    /// ```
    pub fn create_vectors(&self) -> crate::Result<(Vector, Vector)> {
        let mut ptr_lvector = std::ptr::null_mut();
        let mut ptr_evector = std::ptr::null_mut();
        let ierr = unsafe {
            bind_ceed::CeedElemRestrictionCreateVector(self.ptr, &mut ptr_lvector, &mut ptr_evector)
        };
        self.ceed.check_error(ierr)?;
        let lvector = Vector::from_raw(self.ceed, ptr_lvector)?;
        let evector = Vector::from_raw(self.ceed, ptr_evector)?;
        Ok((lvector, evector))
    }

    /// Restrict an Lvector to an Evector or apply its transpose
    ///
    /// # arguments
    ///
    /// * `tmode` - Apply restriction or transpose
    /// * `u`     - Input vector (of size `lsize` when `TransposeMode::NoTranspose`)
    /// * `ru`    - Output vector (of shape `[nelem * elemsize]` when
    ///               `TransposeMode::NoTranspose`). Ordering of the Evector is
    ///               decided by the backend.
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let nelem = 3;
    /// let mut ind: Vec<i32> = vec![0; 2 * nelem];
    /// for i in 0..nelem {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed
    ///     .elem_restriction(nelem, 2, 1, 1, nelem + 1, MemType::Host, &ind)
    ///     .unwrap();
    ///
    /// let x = ceed.vector_from_slice(&[0., 1., 2., 3.]).unwrap();
    /// let mut y = ceed.vector(nelem * 2).unwrap();
    /// y.set_value(0.0);
    ///
    /// r.apply(TransposeMode::NoTranspose, &x, &mut y).unwrap();
    ///
    /// y.view().iter().enumerate().for_each(|(i, arr)| {
    ///     assert_eq!(
    ///         *arr,
    ///         ((i + 1) / 2) as f64,
    ///         "Incorrect value in restricted vector"
    ///     );
    /// });
    /// ```
    pub fn apply(&self, tmode: TransposeMode, u: &Vector, ru: &mut Vector) -> crate::Result<i32> {
        let tmode = tmode as bind_ceed::CeedTransposeMode;
        let ierr = unsafe {
            bind_ceed::CeedElemRestrictionApply(
                self.ptr,
                tmode,
                u.ptr,
                ru.ptr,
                bind_ceed::CEED_REQUEST_IMMEDIATE,
            )
        };
        self.ceed.check_error(ierr)
    }

    /// Returns the Lvector component stride
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let nelem = 3;
    /// let compstride = 1;
    /// let mut ind: Vec<i32> = vec![0; 2 * nelem];
    /// for i in 0..nelem {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed
    ///     .elem_restriction(nelem, 2, 1, compstride, nelem + 1, MemType::Host, &ind)
    ///     .unwrap();
    ///
    /// let c = r.comp_stride();
    /// assert_eq!(c, compstride, "Incorrect component stride");
    /// ```
    pub fn comp_stride(&self) -> usize {
        let mut compstride = 0;
        unsafe { bind_ceed::CeedElemRestrictionGetCompStride(self.ptr, &mut compstride) };
        usize::try_from(compstride).unwrap()
    }

    /// Returns the total number of elements in the range of a ElemRestriction
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let nelem = 3;
    /// let mut ind: Vec<i32> = vec![0; 2 * nelem];
    /// for i in 0..nelem {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed
    ///     .elem_restriction(nelem, 2, 1, 1, nelem + 1, MemType::Host, &ind)
    ///     .unwrap();
    ///
    /// let n = r.num_elements();
    /// assert_eq!(n, nelem, "Incorrect number of elements");
    /// ```
    pub fn num_elements(&self) -> usize {
        let mut numelem = 0;
        unsafe { bind_ceed::CeedElemRestrictionGetNumElements(self.ptr, &mut numelem) };
        usize::try_from(numelem).unwrap()
    }

    /// Returns the size of elements in the ElemRestriction
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let nelem = 3;
    /// let elem_size = 2;
    /// let mut ind: Vec<i32> = vec![0; 2 * nelem];
    /// for i in 0..nelem {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed
    ///     .elem_restriction(nelem, elem_size, 1, 1, nelem + 1, MemType::Host, &ind)
    ///     .unwrap();
    ///
    /// let e = r.elem_size();
    /// assert_eq!(e, elem_size, "Incorrect element size");
    /// ```
    pub fn elem_size(&self) -> usize {
        let mut elemsize = 0;
        unsafe { bind_ceed::CeedElemRestrictionGetElementSize(self.ptr, &mut elemsize) };
        usize::try_from(elemsize).unwrap()
    }

    /// Returns the size of the Lvector for an ElemRestriction
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let nelem = 3;
    /// let mut ind: Vec<i32> = vec![0; 2 * nelem];
    /// for i in 0..nelem {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed
    ///     .elem_restriction(nelem, 2, 1, 1, nelem + 1, MemType::Host, &ind)
    ///     .unwrap();
    ///
    /// let lsize = r.lvector_size();
    /// assert_eq!(lsize, nelem + 1);
    /// ```
    pub fn lvector_size(&self) -> usize {
        let mut lsize = 0;
        unsafe { bind_ceed::CeedElemRestrictionGetLVectorSize(self.ptr, &mut lsize) };
        usize::try_from(lsize).unwrap()
    }

    /// Returns the number of components in the elements of an ElemRestriction
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let nelem = 3;
    /// let ncomp = 42;
    /// let mut ind: Vec<i32> = vec![0; 2 * nelem];
    /// for i in 0..nelem {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed
    ///     .elem_restriction(nelem, 2, 42, 1, ncomp * (nelem + 1), MemType::Host, &ind)
    ///     .unwrap();
    ///
    /// let n = r.num_components();
    /// assert_eq!(n, ncomp, "Incorrect number of components");
    /// ```
    pub fn num_components(&self) -> usize {
        let mut ncomp = 0;
        unsafe { bind_ceed::CeedElemRestrictionGetNumComponents(self.ptr, &mut ncomp) };
        usize::try_from(ncomp).unwrap()
    }

    /// Returns the multiplicity of nodes in an ElemRestriction
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let nelem = 3;
    /// let mut ind: Vec<i32> = vec![0; 2 * nelem];
    /// for i in 0..nelem {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed
    ///     .elem_restriction(nelem, 2, 1, 1, nelem + 1, MemType::Host, &ind)
    ///     .unwrap();
    ///
    /// let mut mult = ceed.vector(nelem + 1).unwrap();
    /// mult.set_value(0.0);
    ///
    /// r.multiplicity(&mut mult).unwrap();
    ///
    /// mult.view().iter().enumerate().for_each(|(i, arr)| {
    ///     assert_eq!(
    ///         if (i == 0 || i == nelem) { 1. } else { 2. },
    ///         *arr,
    ///         "Incorrect multiplicity array"
    ///     );
    /// });
    /// ```
    pub fn multiplicity(&self, mult: &mut Vector) -> crate::Result<i32> {
        let ierr = unsafe { bind_ceed::CeedElemRestrictionGetMultiplicity(self.ptr, mult.ptr) };
        self.ceed.check_error(ierr)
    }
}

// -----------------------------------------------------------------------------
