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
use std::cell::RefCell;
use std::convert::TryFrom;
use std::ffi::CString;
use std::fmt;
use std::ops::Deref;
use std::ops::DerefMut;
use std::os::raw::c_char;
use std::rc::{Rc, Weak};

// -----------------------------------------------------------------------------
// CeedVector context wrapper
// -----------------------------------------------------------------------------
#[derive(Debug)]
pub struct Vector<'a> {
    pub(crate) ceed: &'a crate::Ceed,
    pub(crate) ptr: bind_ceed::CeedVector,
    pub(crate) array_weak: RefCell<Weak<*const f64>>,
}

// -----------------------------------------------------------------------------
// Destructor
// -----------------------------------------------------------------------------
impl<'a> Drop for Vector<'a> {
    fn drop(&mut self) {
        unsafe {
            if self.ptr != bind_ceed::CEED_VECTOR_NONE && self.ptr != bind_ceed::CEED_VECTOR_ACTIVE
            {
                bind_ceed::CeedVectorDestroy(&mut self.ptr);
            }
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
    /// # let ceed = ceed::Ceed::default_init();
    /// let vec = ceed::vector::Vector::from_slice(&ceed, &[1., 2., 3.,]);
    /// println!("{}", vec);
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut ptr = std::ptr::null_mut();
        let mut sizeloc = crate::max_buffer_length;
        unsafe {
            let file = bind_ceed::open_memstream(&mut ptr, &mut sizeloc);
            let format = CString::new("%12.8f").expect("CString::new failed");
            let format_c: *const c_char = format.into_raw();
            bind_ceed::CeedVectorView(self.ptr, format_c, file);
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
impl<'a> Vector<'a> {
    // Constructors
    pub fn create(ceed: &'a crate::Ceed, n: usize) -> Self {
        let n = i32::try_from(n).unwrap();
        let mut ptr = std::ptr::null_mut();
        unsafe { bind_ceed::CeedVectorCreate(ceed.ptr, n, &mut ptr) };
        Self {
            ceed: ceed,
            ptr: ptr,
            array_weak: RefCell::new(Weak::new()),
        }
    }

    pub fn new(ceed: &'a crate::Ceed, ptr: bind_ceed::CeedVector) -> Self {
        Self {
            ceed: ceed,
            ptr: ptr,
            array_weak: RefCell::new(Weak::new()),
        }
    }

    /// Create a Vector from a slice
    ///
    /// # arguments
    ///
    /// * 'slice' values to initialize vector with
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let vec = ceed::vector::Vector::from_slice(&ceed, &[1., 2., 3.,]);
    /// assert_eq!(vec.length(), 3, "Incorrect length from slice");
    /// ```
    pub fn from_slice(ceed: &'a crate::Ceed, v: &[f64]) -> Self {
        let mut x = Self::create(ceed, v.len());
        x.set_slice(v);
        x
    }

    /// Returns the length of a CeedVector
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let vec = ceed.vector(10);
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
    /// # let ceed = ceed::Ceed::default_init();
    /// let vec = ceed.vector(10);
    /// assert_eq!(vec.len(), 10, "Incorrect length");
    /// ```
    pub fn len(&self) -> usize {
        self.length()
    }

    /// Set the array used by a CeedVector, freeing any previously allocated
    ///   array if applicable
    ///
    /// # arguments
    ///
    /// * 'mtype' - Memory type of  the array being passed
    /// * 'cmode' - Copy mode for the array
    /// * 'vec'   - Array to be used
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let vec = ceed.vector(4);
    /// let mut array = ndarray::Array::range(1., 5., 1.);
    /// vec.set_array(ceed::CopyMode::OwnPointer, array);
    ///
    /// let max_norm = vec.norm(ceed::NormType::Max);
    /// assert_eq!(max_norm, 4.0, "Incorrect max norm");
    /// ```
    pub fn set_array(&self, cmode: crate::CopyMode, mut array: ndarray::Array1<f64>) {
        unsafe {
            bind_ceed::CeedVectorSetArray(
                self.ptr,
                crate::MemType::Host as bind_ceed::CeedMemType,
                cmode as bind_ceed::CeedCopyMode,
                array.as_mut_ptr(),
            )
        };
        if cmode == crate::CopyMode::OwnPointer {
            std::mem::forget(array);
        }
    }

    /// Set the CeedVector to a constant value
    ///
    /// # arguments
    ///
    /// * 'val' - Value to be used
    ///
    /// ```
    /// let ceed = ceed::Ceed::default_init();
    /// let len = 10;
    /// let mut vec = ceed.vector(len);
    ///
    /// let val = 42.0;
    /// vec.set_value(val);
    ///
    /// let v = vec.view();
    /// for i in 0..len {
    ///   assert_eq!(v[i], val, "Value not set correctly");
    /// }
    /// ```
    pub fn set_value(&mut self, value: f64) {
        unsafe { bind_ceed::CeedVectorSetValue(self.ptr, value) };
    }

    /// Set values from a slice of the same length
    ///
    /// # arguments
    ///
    /// * 'slice' values to into self; length must match
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let mut vec = ceed.vector(4);
    /// vec.set_slice(&[10., 11., 12., 13.]);
    ///
    /// let v = vec.view();
    /// for i in 0..4 {
    ///   assert_eq!(v[i], 10. + i as f64, "Slice not set correctly");
    /// }
    /// ```
    pub fn set_slice(&mut self, slice: &[f64]) {
        assert_eq!(self.length(), slice.len());
        unsafe {
            bind_ceed::CeedVectorSetArray(
                self.ptr,
                crate::MemType::Host as bind_ceed::CeedMemType,
                crate::CopyMode::CopyValues as bind_ceed::CeedCopyMode,
                slice.as_ptr() as *mut f64,
            )
        };
    }

    /// Sync the CeedVector to a specified memtype
    ///
    /// # arguments
    ///
    /// * 'mtype' - Memtype to be synced
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let len = 10;
    /// let mut vec = ceed.vector(len);
    ///
    /// let val = 42.0;
    /// vec.set_value(val);
    /// vec.sync(ceed::MemType::Host);
    ///
    /// let v = vec.view();
    /// for i in 0..len {
    ///   assert_eq!(v[i], val, "Value not set correctly");
    /// }
    /// ```
    pub fn sync(&self, mtype: crate::MemType) {
        unsafe { bind_ceed::CeedVectorSyncArray(self.ptr, mtype as bind_ceed::CeedMemType) };
    }

    /// Create an immutable view
    ///
    /// ```
    /// # let ceed= ceed::Ceed::default_init();
    /// let vec = ceed.vector_from_slice(&[10., 11., 12., 13.]);
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
    /// # let ceed= ceed::Ceed::default_init();
    /// let mut vec = ceed.vector_from_slice(&[10., 11., 12., 13.]);
    ///
    /// {
    ///   let mut v = vec.view_mut();
    ///   v[2] = 9.;
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
    /// * 'ntype' - Norm type One, Two, or Max
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let vec = ceed.vector_from_slice(&[1., 2., 3., 4.]);
    ///
    /// let max_norm = vec.norm(ceed::NormType::Max);
    /// assert_eq!(max_norm, 4.0, "Incorrect Max norm");
    ///
    /// let l1_norm = vec.norm(ceed::NormType::One);
    /// assert_eq!(l1_norm, 10., "Incorrect L1 norm");
    ///
    /// let l2_norm = vec.norm(ceed::NormType::Two);
    /// assert!((l2_norm - 5.477) < 1e-3, "Incorrect L2 norm");
    /// ```
    pub fn norm(&self, ntype: crate::NormType) -> f64 {
        let mut res: f64 = 0.0;
        unsafe { bind_ceed::CeedVectorNorm(self.ptr, ntype as bind_ceed::CeedNormType, &mut res) };
        res
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
    array: Rc<*const f64>,
}

impl<'a> VectorView<'a> {
    // Constructor
    fn new(vec: &'a Vector) -> Self {
        if let Some(array) = vec.array_weak.borrow().upgrade() {
            return Self {
                vec: vec,
                array: Rc::clone(&array),
            };
        }
        let mut ptr = std::ptr::null();
        unsafe {
            bind_ceed::CeedVectorGetArrayRead(
                vec.ptr,
                crate::MemType::Host as bind_ceed::CeedMemType,
                &mut ptr,
            );
        }
        let array = std::rc::Rc::new(ptr);
        vec.array_weak.replace(Rc::downgrade(&array));
        Self {
            vec: vec,
            array: array,
        }
    }
}

impl<'a> Drop for VectorView<'a> {
    fn drop(&mut self) {
        if let Some(ptr) = Rc::get_mut(&mut self.array) {
            unsafe {
                bind_ceed::CeedVectorRestoreArrayRead(self.vec.ptr, &mut *ptr);
            }
        }
    }
}

impl<'a> Deref for VectorView<'a> {
    type Target = [f64];
    fn deref(&self) -> &[f64] {
        unsafe { std::slice::from_raw_parts(*self.array, self.vec.len()) }
    }
}

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
    // Constructor
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

impl<'a> Drop for VectorViewMut<'a> {
    fn drop(&mut self) {
        println!("HERE");
        unsafe {
            bind_ceed::CeedVectorRestoreArray(self.vec.ptr, &mut self.array);
        }
    }
}

impl<'a> Deref for VectorViewMut<'a> {
    type Target = [f64];
    fn deref(&self) -> &[f64] {
        unsafe { std::slice::from_raw_parts(self.array, self.vec.len()) }
    }
}

impl<'a> DerefMut for VectorViewMut<'a> {
    fn deref_mut(&mut self) -> &mut [f64] {
        unsafe { std::slice::from_raw_parts_mut(self.array, self.vec.len()) }
    }
}

impl<'a> fmt::Display for VectorViewMut<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VectorViewMut({:?})", self.deref())
    }
}

// -----------------------------------------------------------------------------
