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
use std::os::raw::c_char;
use std::rc::{Rc, Weak};
use std::slice;

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
    /// let x = ceed::vector::Vector::from_slice(&ceed, &[1., 2., 3.,]);
    /// println!("{}", x);
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
    /// let x = ceed::vector::Vector::from_slice(&ceed, &[1., 2., 3.,]);
    /// assert_eq!(x.length(), 3);
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
    /// let n = vec.length();
    /// assert_eq!(n, 10);
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
    /// assert_eq!(vec.len(), 10);
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
    /// let norm = vec.norm(ceed::NormType::Max);
    /// assert_eq!(norm, 4.0)
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
    /// let mut x = ceed.vector(10);
    /// x.set_value(42.0);
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
    /// let mut x = ceed.vector(4);
    /// x.set_slice(&[10., 11., 12., 13.]);
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
    /// let vec = ceed.vector(10);
    /// vec.sync(ceed::MemType::Host);
    /// ```
    pub fn sync(&self, mtype: crate::MemType) {
        unsafe { bind_ceed::CeedVectorSyncArray(self.ptr, mtype as bind_ceed::CeedMemType) };
    }

    /// Get read/write access to a CeedVector via the specified memory type
    ///
    /// # arguments
    ///
    /// * 'mtype' - Memory type on which to access the array
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let mut vec = ceed.vector(10);
    /// vec.set_value(42.0);
    /// let array = vec.get_array(ceed::MemType::Host);
    /// assert_eq!(array[5], 42.0);
    /// vec.restore_array(array);
    /// ```
    pub fn get_array(&self, mtype: crate::MemType) -> ndarray::ArrayViewMut1<f64> {
        let n = self.len();
        let mut ptr = std::ptr::null_mut();
        unsafe {
            bind_ceed::CeedVectorGetArray(self.ptr, mtype as bind_ceed::CeedMemType, &mut ptr)
        };
        let ptr_slice: &mut [f64] = unsafe { slice::from_raw_parts_mut(ptr, n) };
        ndarray::aview_mut1(ptr_slice)
    }

    /// Restore read/write access to a CeedVector via the specified memory type
    ///
    /// # arguments
    ///
    /// * 'array' - Array of vector data
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let mut vec = ceed.vector(10);
    /// let array = vec.get_array(ceed::MemType::Host);
    /// vec.restore_array(array);
    /// ```
    pub fn restore_array(&self, array: ndarray::ArrayViewMut1<f64>) {
        let mut ptr = std::ptr::null_mut();
        unsafe { bind_ceed::CeedVectorRestoreArray(self.ptr, &mut ptr) };
        drop(array);
    }

    /// Get read access to a CeedVector via the specified memory type
    ///
    /// # arguments
    ///
    /// * 'mtype' - Memory type on which to access the array
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let mut vec = ceed.vector(10);
    /// vec.set_value(42.0);
    /// let array = vec.get_array_read(ceed::MemType::Host);
    /// assert_eq!(array[5], 42.0);
    /// vec.restore_array_read(array);
    /// ```
    pub fn get_array_read(&self, mtype: crate::MemType) -> ndarray::ArrayView1<f64> {
        let n = self.len();
        let mut ptr = std::ptr::null();
        unsafe {
            bind_ceed::CeedVectorGetArrayRead(self.ptr, mtype as bind_ceed::CeedMemType, &mut ptr)
        };
        let ptr_slice: &[f64] = unsafe { slice::from_raw_parts(ptr, n) };
        ndarray::aview1(ptr_slice)
    }

    /// Restore read access to a CeedVector via the specified memory type
    ///
    /// # arguments
    ///
    /// * 'array' - Array of vector data
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let vec = ceed.vector(10);
    /// let array = vec.get_array_read(ceed::MemType::Host);
    /// vec.restore_array_read(array);
    /// ```
    pub fn restore_array_read(&self, array: ndarray::ArrayView1<f64>) {
        let mut ptr = std::ptr::null();
        unsafe { bind_ceed::CeedVectorRestoreArrayRead(self.ptr, &mut ptr) };
        drop(array);
    }

    /// Create an immutable view
    ///
    /// ```
    /// # let ceed= ceed::Ceed::default_init();
    /// let vec = ceed.vector_from_slice(&[10., 11., 12., 13.]);
    /// let v = vec.view();
    /// assert_eq!(v[0..2], [10., 11.]);
    /// // It is valid to have multiple immutable views
    /// let w = vec.view();
    /// assert_eq!(v[1..], w[1..]);
    /// ```
    pub fn view(&self) -> VectorView {
        VectorView::new(self)
    }

    /// Return the norm of a CeedVector
    ///
    /// # arguments
    ///
    /// * 'ntype' - Norm type CEED_NORM_1, CEED_NORM_2, or CEED_NORM_MAX
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let mut x = ceed.vector(10);
    /// x.set_value(42.0);
    /// let norm = x.norm(ceed::NormType::Max);
    /// assert_eq!(norm, 42.0)
    /// ```
    pub fn norm(&self, ntype: crate::NormType) -> f64 {
        let mut res: f64 = 0.0;
        unsafe { bind_ceed::CeedVectorNorm(self.ptr, ntype as bind_ceed::CeedNormType, &mut res) };
        res
    }
}

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
