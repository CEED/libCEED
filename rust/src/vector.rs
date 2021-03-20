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

//! A Ceed Vector constitutes the main data structure and serves as input/output
//! for Ceed Operators.

use std::{
    ops::{Deref, DerefMut},
    os::raw::c_char,
};

use crate::prelude::*;

// -----------------------------------------------------------------------------
// CeedVector option
// -----------------------------------------------------------------------------
#[derive(Debug, Clone, Copy)]
pub enum VectorOpt<'a> {
    Some(&'a Vector<'a>),
    Active,
    None,
}
/// Construct a VectorOpt reference from a Vector reference
impl<'a> From<&'a Vector<'_>> for VectorOpt<'a> {
    fn from(vec: &'a Vector) -> Self {
        debug_assert!(vec.ptr != unsafe { bind_ceed::CEED_VECTOR_NONE });
        debug_assert!(vec.ptr != unsafe { bind_ceed::CEED_VECTOR_ACTIVE });
        Self::Some(vec)
    }
}
impl<'a> VectorOpt<'a> {
    /// Transform a Rust libCEED VectorOpt into C libCEED CeedVector
    pub(crate) fn to_raw(self) -> bind_ceed::CeedVector {
        match self {
            Self::Some(vec) => vec.ptr,
            Self::None => unsafe { bind_ceed::CEED_VECTOR_NONE },
            Self::Active => unsafe { bind_ceed::CEED_VECTOR_ACTIVE },
        }
    }
}

// -----------------------------------------------------------------------------
// CeedVector context wrapper
// -----------------------------------------------------------------------------
#[derive(Debug)]
pub struct Vector<'a> {
    ceed: &'a crate::Ceed,
    pub(crate) ptr: bind_ceed::CeedVector,
}
impl From<&'_ Vector<'_>> for bind_ceed::CeedVector {
    fn from(vec: &Vector) -> Self {
        vec.ptr
    }
}

// -----------------------------------------------------------------------------
// Destructor
// -----------------------------------------------------------------------------
impl<'a> Drop for Vector<'a> {
    fn drop(&mut self) {
        let not_none_and_active = self.ptr != unsafe { bind_ceed::CEED_VECTOR_NONE }
            && self.ptr != unsafe { bind_ceed::CEED_VECTOR_ACTIVE };

        if not_none_and_active {
            unsafe { bind_ceed::CeedVectorDestroy(&mut self.ptr) };
        }
    }
}

// -----------------------------------------------------------------------------
// Display
// -----------------------------------------------------------------------------
impl<'a> fmt::Display for Vector<'a> {
    /// View a Vector
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let vec = libceed::vector::Vector::from_slice(&ceed, &[1., 2., 3.]).unwrap();
    /// assert_eq!(
    ///     vec.to_string(),
    ///     "CeedVector length 3
    ///     1.00000000
    ///     2.00000000
    ///     3.00000000
    /// "
    /// )
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut ptr = std::ptr::null_mut();
        let mut sizeloc = crate::MAX_BUFFER_LENGTH;
        let format = CString::new("%12.8f").expect("CString::new failed");
        let format_c: *const c_char = format.into_raw();
        let cstring = unsafe {
            let file = bind_ceed::open_memstream(&mut ptr, &mut sizeloc);
            bind_ceed::CeedVectorView(self.ptr, format_c, file);
            bind_ceed::fclose(file);
            CString::from_raw(ptr)
        };
        cstring.to_string_lossy().fmt(f)
    }
}

// -----------------------------------------------------------------------------
// Implementations
// -----------------------------------------------------------------------------
impl<'a> Vector<'a> {
    // Constructors
    pub fn create(ceed: &'a crate::Ceed, n: usize) -> crate::Result<Self> {
        let n = i32::try_from(n).unwrap();
        let mut ptr = std::ptr::null_mut();
        let ierr = unsafe { bind_ceed::CeedVectorCreate(ceed.ptr, n, &mut ptr) };
        ceed.check_error(ierr)?;
        Ok(Self { ceed, ptr })
    }

    pub(crate) fn from_raw(
        ceed: &'a crate::Ceed,
        ptr: bind_ceed::CeedVector,
    ) -> crate::Result<Self> {
        Ok(Self { ceed, ptr })
    }

    /// Create a Vector from a slice
    ///
    /// # arguments
    ///
    /// * `slice` - values to initialize vector with
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let vec = vector::Vector::from_slice(&ceed, &[1., 2., 3.]).unwrap();
    /// assert_eq!(vec.length(), 3, "Incorrect length from slice");
    /// ```
    pub fn from_slice(ceed: &'a crate::Ceed, v: &[f64]) -> crate::Result<Self> {
        let mut x = Self::create(ceed, v.len())?;
        x.set_slice(v)?;
        Ok(x)
    }

    /// Create a Vector from a mutable array reference
    ///
    /// # arguments
    ///
    /// * `slice` - values to initialize vector with
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let mut rust_vec = vec![1., 2., 3.];
    /// let vec = libceed::vector::Vector::from_array(&ceed, &mut rust_vec).unwrap();
    ///
    /// assert_eq!(vec.length(), 3, "Incorrect length from slice");
    /// ```
    pub fn from_array(ceed: &'a crate::Ceed, v: &mut [f64]) -> crate::Result<Self> {
        let x = Self::create(ceed, v.len())?;
        let (host, user_pointer) = (
            crate::MemType::Host as bind_ceed::CeedMemType,
            crate::CopyMode::UsePointer as bind_ceed::CeedCopyMode,
        );
        let v = v.as_ptr() as *mut f64;
        let ierr = unsafe { bind_ceed::CeedVectorSetArray(x.ptr, host, user_pointer, v) };
        ceed.check_error(ierr)?;
        Ok(x)
    }

    /// Returns the length of a CeedVector
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let vec = ceed.vector(10).unwrap();
    ///
    /// let n = vec.length();
    /// assert_eq!(n, 10, "Incorrect length");
    /// ```
    pub fn length(&self) -> usize {
        let mut n = 0;
        unsafe { bind_ceed::CeedVectorGetLength(self.ptr, &mut n) };
        usize::try_from(n).unwrap()
    }

    /// Returns the length of a CeedVector
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let vec = ceed.vector(10).unwrap();
    /// assert_eq!(vec.len(), 10, "Incorrect length");
    /// ```
    pub fn len(&self) -> usize {
        self.length()
    }

    /// Set the CeedVector to a constant value
    ///
    /// # arguments
    ///
    /// * `val` - Value to be used
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let len = 10;
    /// let mut vec = ceed.vector(len).unwrap();
    ///
    /// let val = 42.0;
    /// vec.set_value(val).unwrap();
    ///
    /// vec.view().iter().for_each(|v| {
    ///     assert_eq!(*v, val, "Value not set correctly");
    /// });
    /// ```
    pub fn set_value(&mut self, value: f64) -> crate::Result<i32> {
        let ierr = unsafe { bind_ceed::CeedVectorSetValue(self.ptr, value) };
        self.ceed.check_error(ierr)
    }

    /// Set values from a slice of the same length
    ///
    /// # arguments
    ///
    /// * `slice` - values to into self; length must match
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let mut vec = ceed.vector(4).unwrap();
    /// vec.set_slice(&[10., 11., 12., 13.]).unwrap();
    ///
    /// vec.view().iter().enumerate().for_each(|(i, v)| {
    ///     assert_eq!(*v, 10. + i as f64, "Slice not set correctly");
    /// });
    /// ```
    pub fn set_slice(&mut self, slice: &[f64]) -> crate::Result<i32> {
        assert_eq!(self.length(), slice.len());
        let (host, copy_mode) = (
            crate::MemType::Host as bind_ceed::CeedMemType,
            crate::CopyMode::CopyValues as bind_ceed::CeedCopyMode,
        );
        let ierr = unsafe {
            bind_ceed::CeedVectorSetArray(self.ptr, host, copy_mode, slice.as_ptr() as *mut f64)
        };
        self.ceed.check_error(ierr)
    }

    /// Sync the CeedVector to a specified memtype
    ///
    /// # arguments
    ///
    /// * `mtype` - Memtype to be synced
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let len = 10;
    /// let mut vec = ceed.vector(len).unwrap();
    ///
    /// let val = 42.0;
    /// vec.set_value(val);
    /// vec.sync(MemType::Host).unwrap();
    ///
    /// vec.view().iter().for_each(|v| {
    ///     assert_eq!(*v, val, "Value not set correctly");
    /// });
    /// ```
    pub fn sync(&self, mtype: crate::MemType) -> crate::Result<i32> {
        let ierr =
            unsafe { bind_ceed::CeedVectorSyncArray(self.ptr, mtype as bind_ceed::CeedMemType) };
        self.ceed.check_error(ierr)
    }

    /// Create an immutable view
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let vec = ceed.vector_from_slice(&[10., 11., 12., 13.]).unwrap();
    ///
    /// let v = vec.view();
    /// assert_eq!(v[0..2], [10., 11.]);
    ///
    /// // It is valid to have multiple immutable views
    /// let w = vec.view();
    /// assert_eq!(v[1..], w[1..]);
    /// ```
    pub fn view(&self) -> VectorView {
        VectorView::new(self)
    }

    /// Create an mutable view
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let mut vec = ceed.vector_from_slice(&[10., 11., 12., 13.]).unwrap();
    ///
    /// {
    ///     let mut v = vec.view_mut();
    ///     v[2] = 9.;
    /// }
    ///
    /// let w = vec.view();
    /// assert_eq!(w[2], 9., "View did not mutate data");
    /// ```
    pub fn view_mut(&mut self) -> VectorViewMut {
        VectorViewMut::new(self)
    }

    /// Return the norm of a CeedVector
    ///
    /// # arguments
    ///
    /// * `ntype` - Norm type One, Two, or Max
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let vec = ceed.vector_from_slice(&[1., 2., 3., 4.]).unwrap();
    ///
    /// let max_norm = vec.norm(NormType::Max).unwrap();
    /// assert_eq!(max_norm, 4.0, "Incorrect Max norm");
    ///
    /// let l1_norm = vec.norm(NormType::One).unwrap();
    /// assert_eq!(l1_norm, 10., "Incorrect L1 norm");
    ///
    /// let l2_norm = vec.norm(NormType::Two).unwrap();
    /// assert!((l2_norm - 5.477) < 1e-3, "Incorrect L2 norm");
    /// ```
    pub fn norm(&self, ntype: crate::NormType) -> crate::Result<f64> {
        let mut res: f64 = 0.0;
        let ierr = unsafe {
            bind_ceed::CeedVectorNorm(self.ptr, ntype as bind_ceed::CeedNormType, &mut res)
        };
        self.ceed.check_error(ierr)?;
        Ok(res)
    }
}

// -----------------------------------------------------------------------------
// Vector Viewer
// -----------------------------------------------------------------------------
/// A (host) view of a Vector with Deref to slice.  We can't make
/// Vector itself Deref to slice because we can't handle the drop to
/// call bind_ceed::CeedVectorRestoreArrayRead().
#[derive(Debug)]
pub struct VectorView<'a> {
    vec: &'a Vector<'a>,
    array: *const f64,
}

impl<'a> VectorView<'a> {
    /// Construct a VectorView from a Vector reference
    fn new(vec: &'a Vector) -> Self {
        let mut array = std::ptr::null();
        unsafe {
            bind_ceed::CeedVectorGetArrayRead(
                vec.ptr,
                crate::MemType::Host as bind_ceed::CeedMemType,
                &mut array,
            );
        }
        Self {
            vec: vec,
            array: array,
        }
    }
}

// Destructor
impl<'a> Drop for VectorView<'a> {
    fn drop(&mut self) {
        unsafe {
            bind_ceed::CeedVectorRestoreArrayRead(self.vec.ptr, &mut self.array);
        }
    }
}

// Data access
impl<'a> Deref for VectorView<'a> {
    type Target = [f64];
    fn deref(&self) -> &[f64] {
        unsafe { std::slice::from_raw_parts(self.array, self.vec.len()) }
    }
}

// Viewing
impl<'a> fmt::Display for VectorView<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VectorView({:?})", self.deref())
    }
}

// -----------------------------------------------------------------------------
// Vector Viewer Mutable
// -----------------------------------------------------------------------------
/// A mutable (host) view of a Vector with Deref to slice.
#[derive(Debug)]
pub struct VectorViewMut<'a> {
    vec: &'a Vector<'a>,
    array: *mut f64,
}

impl<'a> VectorViewMut<'a> {
    /// Construct a VectorViewMut from a Vector reference
    fn new(vec: &'a mut Vector) -> Self {
        let mut ptr = std::ptr::null_mut();
        unsafe {
            bind_ceed::CeedVectorGetArray(
                vec.ptr,
                crate::MemType::Host as bind_ceed::CeedMemType,
                &mut ptr,
            );
        }
        Self {
            vec: vec,
            array: ptr,
        }
    }
}

// Destructor
impl<'a> Drop for VectorViewMut<'a> {
    fn drop(&mut self) {
        unsafe {
            bind_ceed::CeedVectorRestoreArray(self.vec.ptr, &mut self.array);
        }
    }
}

// Data access
impl<'a> Deref for VectorViewMut<'a> {
    type Target = [f64];
    fn deref(&self) -> &[f64] {
        unsafe { std::slice::from_raw_parts(self.array, self.vec.len()) }
    }
}

// Mutable data access
impl<'a> DerefMut for VectorViewMut<'a> {
    fn deref_mut(&mut self) -> &mut [f64] {
        unsafe { std::slice::from_raw_parts_mut(self.array, self.vec.len()) }
    }
}

// Viewing
impl<'a> fmt::Display for VectorViewMut<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VectorViewMut({:?})", self.deref())
    }
}

// -----------------------------------------------------------------------------
