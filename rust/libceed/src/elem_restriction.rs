// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

//! A Ceed ElemRestriction decomposes elements and groups the degrees of freedom
//! (dofs) according to the different elements they belong to.

use crate::{prelude::*, vector::Vector, TransposeMode};

// -----------------------------------------------------------------------------
// ElemRestriction option
// -----------------------------------------------------------------------------
#[derive(Debug)]
pub enum ElemRestrictionOpt<'a> {
    Some(&'a ElemRestriction<'a>),
    None,
}
/// Construct a ElemRestrictionOpt reference from a ElemRestriction reference
impl<'a> From<&'a ElemRestriction<'_>> for ElemRestrictionOpt<'a> {
    fn from(rstr: &'a ElemRestriction) -> Self {
        debug_assert!(rstr.ptr != unsafe { bind_ceed::CEED_ELEMRESTRICTION_NONE });
        Self::Some(rstr)
    }
}
impl<'a> ElemRestrictionOpt<'a> {
    /// Transform a Rust libCEED ElemRestrictionOpt into C libCEED
    /// CeedElemRestriction
    pub(crate) fn to_raw(&self) -> bind_ceed::CeedElemRestriction {
        match self {
            Self::Some(rstr) => rstr.ptr,
            Self::None => unsafe { bind_ceed::CEED_ELEMRESTRICTION_NONE },
        }
    }

    /// Check if an ElemRestrictionOpt is Some
    ///
    /// ```
    /// # use libceed::{prelude::*, ElemRestrictionOpt, MemType};
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let nelem = 3;
    /// let mut ind: Vec<i32> = vec![0; 2 * nelem];
    /// for i in 0..nelem {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed.elem_restriction(nelem, 2, 1, 1, nelem + 1, MemType::Host, &ind)?;
    /// let r_opt = ElemRestrictionOpt::from(&r);
    /// assert!(r_opt.is_some(), "Incorrect ElemRestrictionOpt");
    ///
    /// let r_opt = ElemRestrictionOpt::None;
    /// assert!(!r_opt.is_some(), "Incorrect ElemRestrictionOpt");
    /// # Ok(())
    /// # }
    /// ```
    pub fn is_some(&self) -> bool {
        match self {
            Self::Some(_) => true,
            Self::None => false,
        }
    }

    /// Check if an ElemRestrictionOpt is None
    ///
    /// ```
    /// # use libceed::{prelude::*, ElemRestrictionOpt, MemType};
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let nelem = 3;
    /// let mut ind: Vec<i32> = vec![0; 2 * nelem];
    /// for i in 0..nelem {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed.elem_restriction(nelem, 2, 1, 1, nelem + 1, MemType::Host, &ind)?;
    /// let r_opt = ElemRestrictionOpt::from(&r);
    /// assert!(!r_opt.is_none(), "Incorrect ElemRestrictionOpt");
    ///
    /// let r_opt = ElemRestrictionOpt::None;
    /// assert!(r_opt.is_none(), "Incorrect ElemRestrictionOpt");
    /// # Ok(())
    /// # }
    /// ```
    pub fn is_none(&self) -> bool {
        match self {
            Self::Some(_) => false,
            Self::None => true,
        }
    }
}

// -----------------------------------------------------------------------------
// ElemRestriction context wrapper
// -----------------------------------------------------------------------------
#[derive(Debug)]
pub struct ElemRestriction<'a> {
    pub(crate) ptr: bind_ceed::CeedElemRestriction,
    _lifeline: PhantomData<&'a ()>,
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
    /// # use libceed::{prelude::*, MemType};
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let nelem = 3;
    /// let mut ind: Vec<i32> = vec![0; 2 * nelem];
    /// for i in 0..nelem {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed.elem_restriction(nelem, 2, 1, 1, nelem + 1, MemType::Host, &ind)?;
    /// println!("{}", r);
    /// # Ok(())
    /// # }
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
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        ceed: &crate::Ceed,
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
            isize::try_from(lsize).unwrap(),
            mtype as bind_ceed::CeedMemType,
        );
        ceed.check_error(unsafe {
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
        })?;
        Ok(Self {
            ptr,
            _lifeline: PhantomData,
        })
    }

    pub(crate) unsafe fn from_raw(ptr: bind_ceed::CeedElemRestriction) -> crate::Result<Self> {
        Ok(Self {
            ptr,
            _lifeline: PhantomData,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_oriented(
        ceed: &crate::Ceed,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        compstride: usize,
        lsize: usize,
        mtype: crate::MemType,
        offsets: &[i32],
        orients: &[bool],
    ) -> crate::Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let (nelem, elemsize, ncomp, compstride, lsize, mtype) = (
            i32::try_from(nelem).unwrap(),
            i32::try_from(elemsize).unwrap(),
            i32::try_from(ncomp).unwrap(),
            i32::try_from(compstride).unwrap(),
            isize::try_from(lsize).unwrap(),
            mtype as bind_ceed::CeedMemType,
        );
        ceed.check_error(unsafe {
            bind_ceed::CeedElemRestrictionCreateOriented(
                ceed.ptr,
                nelem,
                elemsize,
                ncomp,
                compstride,
                lsize,
                mtype,
                crate::CopyMode::CopyValues as bind_ceed::CeedCopyMode,
                offsets.as_ptr(),
                orients.as_ptr(),
                &mut ptr,
            )
        })?;
        Ok(Self {
            ptr,
            _lifeline: PhantomData,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_curl_oriented(
        ceed: &crate::Ceed,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        compstride: usize,
        lsize: usize,
        mtype: crate::MemType,
        offsets: &[i32],
        curlorients: &[i8],
    ) -> crate::Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let (nelem, elemsize, ncomp, compstride, lsize, mtype) = (
            i32::try_from(nelem).unwrap(),
            i32::try_from(elemsize).unwrap(),
            i32::try_from(ncomp).unwrap(),
            i32::try_from(compstride).unwrap(),
            isize::try_from(lsize).unwrap(),
            mtype as bind_ceed::CeedMemType,
        );
        ceed.check_error(unsafe {
            bind_ceed::CeedElemRestrictionCreateCurlOriented(
                ceed.ptr,
                nelem,
                elemsize,
                ncomp,
                compstride,
                lsize,
                mtype,
                crate::CopyMode::CopyValues as bind_ceed::CeedCopyMode,
                offsets.as_ptr(),
                curlorients.as_ptr(),
                &mut ptr,
            )
        })?;
        Ok(Self {
            ptr,
            _lifeline: PhantomData,
        })
    }

    pub fn create_strided(
        ceed: &crate::Ceed,
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
            isize::try_from(lsize).unwrap(),
        );
        ceed.check_error(unsafe {
            bind_ceed::CeedElemRestrictionCreateStrided(
                ceed.ptr,
                nelem,
                elemsize,
                ncomp,
                lsize,
                strides.as_ptr(),
                &mut ptr,
            )
        })?;
        Ok(Self {
            ptr,
            _lifeline: PhantomData,
        })
    }

    // Raw Ceed for error handling
    #[doc(hidden)]
    fn ceed(&self) -> bind_ceed::Ceed {
        unsafe { bind_ceed::CeedElemRestrictionReturnCeed(self.ptr) }
    }

    // Error handling
    #[doc(hidden)]
    fn check_error(&self, ierr: i32) -> crate::Result<i32> {
        crate::check_error(|| self.ceed(), ierr)
    }

    /// Create an Lvector for an ElemRestriction
    ///
    /// ```
    /// # use libceed::{prelude::*, MemType};
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let nelem = 3;
    /// let mut ind: Vec<i32> = vec![0; 2 * nelem];
    /// for i in 0..nelem {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed.elem_restriction(nelem, 2, 1, 1, nelem + 1, MemType::Host, &ind)?;
    ///
    /// let lvector = r.create_lvector()?;
    ///
    /// assert_eq!(lvector.length(), nelem + 1, "Incorrect Lvector size");
    /// # Ok(())
    /// # }
    /// ```
    pub fn create_lvector<'b>(&self) -> crate::Result<Vector<'b>> {
        let mut ptr_lvector = std::ptr::null_mut();
        let null = std::ptr::null_mut() as *mut _;
        self.check_error(unsafe {
            bind_ceed::CeedElemRestrictionCreateVector(self.ptr, &mut ptr_lvector, null)
        })?;
        unsafe { Vector::from_raw(ptr_lvector) }
    }

    /// Create an Evector for an ElemRestriction
    ///
    /// ```
    /// # use libceed::{prelude::*, MemType};
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let nelem = 3;
    /// let mut ind: Vec<i32> = vec![0; 2 * nelem];
    /// for i in 0..nelem {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed.elem_restriction(nelem, 2, 1, 1, nelem + 1, MemType::Host, &ind)?;
    ///
    /// let evector = r.create_evector()?;
    ///
    /// assert_eq!(evector.length(), nelem * 2, "Incorrect Evector size");
    /// # Ok(())
    /// # }
    /// ```
    pub fn create_evector<'b>(&self) -> crate::Result<Vector<'b>> {
        let mut ptr_evector = std::ptr::null_mut();
        let null = std::ptr::null_mut() as *mut _;
        self.check_error(unsafe {
            bind_ceed::CeedElemRestrictionCreateVector(self.ptr, null, &mut ptr_evector)
        })?;
        unsafe { Vector::from_raw(ptr_evector) }
    }

    /// Create Vectors for an ElemRestriction
    ///
    /// ```
    /// # use libceed::{prelude::*, MemType};
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let nelem = 3;
    /// let mut ind: Vec<i32> = vec![0; 2 * nelem];
    /// for i in 0..nelem {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed.elem_restriction(nelem, 2, 1, 1, nelem + 1, MemType::Host, &ind)?;
    ///
    /// let (lvector, evector) = r.create_vectors()?;
    ///
    /// assert_eq!(lvector.length(), nelem + 1, "Incorrect Lvector size");
    /// assert_eq!(evector.length(), nelem * 2, "Incorrect Evector size");
    /// # Ok(())
    /// # }
    /// ```
    pub fn create_vectors<'b, 'c>(&self) -> crate::Result<(Vector<'b>, Vector<'c>)> {
        let mut ptr_lvector = std::ptr::null_mut();
        let mut ptr_evector = std::ptr::null_mut();
        self.check_error(unsafe {
            bind_ceed::CeedElemRestrictionCreateVector(self.ptr, &mut ptr_lvector, &mut ptr_evector)
        })?;
        let lvector = unsafe { Vector::from_raw(ptr_lvector)? };
        let evector = unsafe { Vector::from_raw(ptr_evector)? };
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
    /// # use libceed::{prelude::*, MemType, Scalar, TransposeMode};
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let nelem = 3;
    /// let mut ind: Vec<i32> = vec![0; 2 * nelem];
    /// for i in 0..nelem {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed.elem_restriction(nelem, 2, 1, 1, nelem + 1, MemType::Host, &ind)?;
    ///
    /// let x = ceed.vector_from_slice(&[0., 1., 2., 3.])?;
    /// let mut y = ceed.vector(nelem * 2)?;
    /// y.set_value(0.0);
    ///
    /// r.apply(TransposeMode::NoTranspose, &x, &mut y)?;
    ///
    /// for (i, y) in y.view()?.iter().enumerate() {
    ///     assert_eq!(
    ///         *y,
    ///         ((i + 1) / 2) as Scalar,
    ///         "Incorrect value in restricted vector"
    ///     );
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn apply(&self, tmode: TransposeMode, u: &Vector, ru: &mut Vector) -> crate::Result<i32> {
        let tmode = tmode as bind_ceed::CeedTransposeMode;
        self.check_error(unsafe {
            bind_ceed::CeedElemRestrictionApply(
                self.ptr,
                tmode,
                u.ptr,
                ru.ptr,
                bind_ceed::CEED_REQUEST_IMMEDIATE,
            )
        })
    }

    /// Returns the Lvector component stride
    ///
    /// ```
    /// # use libceed::{prelude::*, MemType};
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let nelem = 3;
    /// let compstride = 1;
    /// let mut ind: Vec<i32> = vec![0; 2 * nelem];
    /// for i in 0..nelem {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed.elem_restriction(nelem, 2, 1, compstride, nelem + 1, MemType::Host, &ind)?;
    ///
    /// let c = r.comp_stride();
    /// assert_eq!(c, compstride, "Incorrect component stride");
    /// # Ok(())
    /// # }
    /// ```
    pub fn comp_stride(&self) -> usize {
        let mut compstride = 0;
        unsafe { bind_ceed::CeedElemRestrictionGetCompStride(self.ptr, &mut compstride) };
        usize::try_from(compstride).unwrap()
    }

    /// Returns the total number of elements in the range of a ElemRestriction
    ///
    /// ```
    /// # use libceed::{prelude::*, MemType};
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let nelem = 3;
    /// let mut ind: Vec<i32> = vec![0; 2 * nelem];
    /// for i in 0..nelem {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed.elem_restriction(nelem, 2, 1, 1, nelem + 1, MemType::Host, &ind)?;
    ///
    /// let n = r.num_elements();
    /// assert_eq!(n, nelem, "Incorrect number of elements");
    /// # Ok(())
    /// # }
    /// ```
    pub fn num_elements(&self) -> usize {
        let mut numelem = 0;
        unsafe { bind_ceed::CeedElemRestrictionGetNumElements(self.ptr, &mut numelem) };
        usize::try_from(numelem).unwrap()
    }

    /// Returns the size of elements in the ElemRestriction
    ///
    /// ```
    /// # use libceed::{prelude::*, MemType};
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let nelem = 3;
    /// let elem_size = 2;
    /// let mut ind: Vec<i32> = vec![0; 2 * nelem];
    /// for i in 0..nelem {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed.elem_restriction(nelem, elem_size, 1, 1, nelem + 1, MemType::Host, &ind)?;
    ///
    /// let e = r.elem_size();
    /// assert_eq!(e, elem_size, "Incorrect element size");
    /// # Ok(())
    /// # }
    /// ```
    pub fn elem_size(&self) -> usize {
        let mut elemsize = 0;
        unsafe { bind_ceed::CeedElemRestrictionGetElementSize(self.ptr, &mut elemsize) };
        usize::try_from(elemsize).unwrap()
    }

    /// Returns the size of the Lvector for an ElemRestriction
    ///
    /// ```
    /// # use libceed::{prelude::*, MemType};
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let nelem = 3;
    /// let mut ind: Vec<i32> = vec![0; 2 * nelem];
    /// for i in 0..nelem {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed.elem_restriction(nelem, 2, 1, 1, nelem + 1, MemType::Host, &ind)?;
    ///
    /// let lsize = r.lvector_size();
    /// assert_eq!(lsize, nelem + 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn lvector_size(&self) -> usize {
        let mut lsize = 0;
        unsafe { bind_ceed::CeedElemRestrictionGetLVectorSize(self.ptr, &mut lsize) };
        usize::try_from(lsize).unwrap()
    }

    /// Returns the number of components in the elements of an ElemRestriction
    ///
    /// ```
    /// # use libceed::{prelude::*, MemType};
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let nelem = 3;
    /// let ncomp = 42;
    /// let mut ind: Vec<i32> = vec![0; 2 * nelem];
    /// for i in 0..nelem {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed.elem_restriction(nelem, 2, 42, 1, ncomp * (nelem + 1), MemType::Host, &ind)?;
    ///
    /// let n = r.num_components();
    /// assert_eq!(n, ncomp, "Incorrect number of components");
    /// # Ok(())
    /// # }
    /// ```
    pub fn num_components(&self) -> usize {
        let mut ncomp = 0;
        unsafe { bind_ceed::CeedElemRestrictionGetNumComponents(self.ptr, &mut ncomp) };
        usize::try_from(ncomp).unwrap()
    }

    /// Returns the multiplicity of nodes in an ElemRestriction
    ///
    /// ```
    /// # use libceed::{prelude::*, MemType};
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let nelem = 3;
    /// let mut ind: Vec<i32> = vec![0; 2 * nelem];
    /// for i in 0..nelem {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed.elem_restriction(nelem, 2, 1, 1, nelem + 1, MemType::Host, &ind)?;
    ///
    /// let mut mult = ceed.vector(nelem + 1)?;
    /// mult.set_value(0.0);
    ///
    /// r.multiplicity(&mut mult)?;
    ///
    /// for (i, m) in mult.view()?.iter().enumerate() {
    ///     assert_eq!(
    ///         *m,
    ///         if (i == 0 || i == nelem) { 1. } else { 2. },
    ///         "Incorrect multiplicity value"
    ///     );
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn multiplicity(&self, mult: &mut Vector) -> crate::Result<i32> {
        self.check_error(unsafe {
            bind_ceed::CeedElemRestrictionGetMultiplicity(self.ptr, mult.ptr)
        })
    }
}

// -----------------------------------------------------------------------------
