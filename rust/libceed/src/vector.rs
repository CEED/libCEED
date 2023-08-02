// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

//! A Ceed Vector constitutes the main data structure and serves as input/output
//! for Ceed Operators.

use std::{
    ops::{Deref, DerefMut},
    os::raw::c_char,
};

use crate::prelude::*;

// -----------------------------------------------------------------------------
// Vector option
// -----------------------------------------------------------------------------
#[derive(Debug)]
pub enum VectorOpt<'a> {
    Some(&'a Vector<'a>),
    Active,
    None,
}
/// Construct a VectorOpt reference from a Vector reference
impl<'a> From<&'a Vector<'_>> for VectorOpt<'a> {
    fn from(vec: &'a Vector) -> Self {
        debug_assert!(vec.ptr != unsafe { bind_ceed::CEED_VECTOR_ACTIVE });
        debug_assert!(vec.ptr != unsafe { bind_ceed::CEED_VECTOR_NONE });
        Self::Some(vec)
    }
}
impl<'a> VectorOpt<'a> {
    /// Transform a Rust libCEED VectorOpt into C libCEED CeedVector
    pub(crate) fn to_raw(self) -> bind_ceed::CeedVector {
        match self {
            Self::Some(vec) => vec.ptr,
            Self::Active => unsafe { bind_ceed::CEED_VECTOR_ACTIVE },
            Self::None => unsafe { bind_ceed::CEED_VECTOR_NONE },
        }
    }

    /// Check if a VectorOpt is Some
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let vec = libceed::vector::Vector::from_slice(&ceed, &[1., 2., 3.])?;
    /// let vec_opt = VectorOpt::from(&vec);
    /// assert!(vec_opt.is_some(), "Incorrect VectorOpt");
    ///
    /// let vec_opt = VectorOpt::Active;
    /// assert!(!vec_opt.is_some(), "Incorrect VectorOpt");
    ///
    /// let vec_opt = VectorOpt::None;
    /// assert!(!vec_opt.is_some(), "Incorrect VectorOpt");
    /// # Ok(())
    /// # }
    /// ```
    pub fn is_some(&self) -> bool {
        match self {
            Self::Some(_) => true,
            Self::Active => false,
            Self::None => false,
        }
    }

    /// Check if a VectorOpt is Active
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let vec = libceed::vector::Vector::from_slice(&ceed, &[1., 2., 3.])?;
    /// let vec_opt = VectorOpt::from(&vec);
    /// assert!(!vec_opt.is_active(), "Incorrect VectorOpt");
    ///
    /// let vec_opt = VectorOpt::Active;
    /// assert!(vec_opt.is_active(), "Incorrect VectorOpt");
    ///
    /// let vec_opt = VectorOpt::None;
    /// assert!(!vec_opt.is_active(), "Incorrect VectorOpt");
    /// # Ok(())
    /// # }
    /// ```
    pub fn is_active(&self) -> bool {
        match self {
            Self::Some(_) => false,
            Self::Active => true,
            Self::None => false,
        }
    }

    /// Check if a VectorOpt is Some
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let vec = libceed::vector::Vector::from_slice(&ceed, &[1., 2., 3.])?;
    /// let vec_opt = VectorOpt::from(&vec);
    /// assert!(!vec_opt.is_none(), "Incorrect VectorOpt");
    ///
    /// let vec_opt = VectorOpt::Active;
    /// assert!(!vec_opt.is_none(), "Incorrect VectorOpt");
    ///
    /// let vec_opt = VectorOpt::None;
    /// assert!(vec_opt.is_none(), "Incorrect VectorOpt");
    /// # Ok(())
    /// # }
    /// ```
    pub fn is_none(&self) -> bool {
        match self {
            Self::Some(_) => false,
            Self::Active => false,
            Self::None => true,
        }
    }
}

// -----------------------------------------------------------------------------
// Vector borrowed slice wrapper
// -----------------------------------------------------------------------------
pub struct VectorSliceWrapper<'a> {
    pub(crate) vector: crate::Vector<'a>,
    pub(crate) _slice: &'a mut [crate::Scalar],
}

// -----------------------------------------------------------------------------
// Destructor
// -----------------------------------------------------------------------------
impl<'a> Drop for VectorSliceWrapper<'a> {
    fn drop(&mut self) {
        unsafe {
            bind_ceed::CeedVectorTakeArray(
                self.vector.ptr,
                crate::MemType::Host as bind_ceed::CeedMemType,
                std::ptr::null_mut(),
            )
        };
    }
}

// -----------------------------------------------------------------------------
// Convenience constructor
// -----------------------------------------------------------------------------
impl<'a> VectorSliceWrapper<'a> {
    fn from_vector_and_slice_mut<'b>(
        vec: &'b mut crate::Vector,
        slice: &'a mut [crate::Scalar],
    ) -> crate::Result<Self> {
        assert_eq!(vec.length(), slice.len());
        let (host, copy_mode) = (
            crate::MemType::Host as bind_ceed::CeedMemType,
            crate::CopyMode::UsePointer as bind_ceed::CeedCopyMode,
        );
        let ierr = unsafe {
            bind_ceed::CeedVectorSetArray(
                vec.ptr,
                host,
                copy_mode,
                slice.as_ptr() as *mut crate::Scalar,
            )
        };
        vec.check_error(ierr)?;

        Ok(Self {
            vector: crate::Vector::from_raw(vec.ptr_copy_mut()?)?,
            _slice: slice,
        })
    }
}

// -----------------------------------------------------------------------------
// Vector context wrapper
// -----------------------------------------------------------------------------
#[derive(Debug)]
pub struct Vector<'a> {
    pub(crate) ptr: bind_ceed::CeedVector,
    _lifeline: PhantomData<&'a ()>,
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
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let vec = libceed::vector::Vector::from_slice(&ceed, &[1., 2., 3.])?;
    /// assert_eq!(
    ///     vec.to_string(),
    ///     "CeedVector length 3
    ///     1.00000000
    ///     2.00000000
    ///     3.00000000
    /// "
    /// );
    /// # Ok(())
    /// # }
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
    pub fn create(ceed: &crate::Ceed, n: usize) -> crate::Result<Self> {
        let n = isize::try_from(n).unwrap();
        let mut ptr = std::ptr::null_mut();
        let ierr = unsafe { bind_ceed::CeedVectorCreate(ceed.ptr, n, &mut ptr) };
        ceed.check_error(ierr)?;
        Ok(Self {
            ptr,
            _lifeline: PhantomData,
        })
    }

    pub(crate) fn from_raw(ptr: bind_ceed::CeedVector) -> crate::Result<Self> {
        Ok(Self {
            ptr,
            _lifeline: PhantomData,
        })
    }

    fn ptr_copy_mut(&mut self) -> crate::Result<bind_ceed::CeedVector> {
        let mut ptr_copy = std::ptr::null_mut();
        let ierr = unsafe { bind_ceed::CeedVectorReferenceCopy(self.ptr, &mut ptr_copy) };
        self.check_error(ierr)?;
        Ok(ptr_copy)
    }

    /// Copy the array of self into vec_copy
    ///
    /// # arguments
    ///
    /// * `vec_source` - vector to copy array values from
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let a = ceed.vector_from_slice(&[1., 2., 3.])?;
    /// let mut b = ceed.vector(3)?;
    ///
    /// b.copy_from(&a)?;
    /// for (i, v) in b.view()?.iter().enumerate() {
    ///     assert_eq!(*v, (i + 1) as Scalar, "Copy contents not set correctly");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    /// ```
    pub fn copy_from(&mut self, vec_source: &crate::Vector) -> crate::Result<i32> {
        let ierr = unsafe { bind_ceed::CeedVectorCopy(vec_source.ptr, self.ptr) };
        self.check_error(ierr)
    }

    /// Create a Vector from a slice
    ///
    /// # arguments
    ///
    /// * `slice` - values to initialize vector with
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let vec = vector::Vector::from_slice(&ceed, &[1., 2., 3.])?;
    /// assert_eq!(vec.length(), 3, "Incorrect length from slice");
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_slice(ceed: &crate::Ceed, v: &[crate::Scalar]) -> crate::Result<Self> {
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
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let mut rust_vec = vec![1., 2., 3.];
    /// let vec = libceed::vector::Vector::from_array(&ceed, &mut rust_vec)?;
    ///
    /// assert_eq!(vec.length(), 3, "Incorrect length from slice");
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_array(ceed: &crate::Ceed, v: &mut [crate::Scalar]) -> crate::Result<Self> {
        let x = Self::create(ceed, v.len())?;
        let (host, user_pointer) = (
            crate::MemType::Host as bind_ceed::CeedMemType,
            crate::CopyMode::UsePointer as bind_ceed::CeedCopyMode,
        );
        let v = v.as_ptr() as *mut crate::Scalar;
        let ierr = unsafe { bind_ceed::CeedVectorSetArray(x.ptr, host, user_pointer, v) };
        ceed.check_error(ierr)?;
        Ok(x)
    }

    // Error handling
    #[doc(hidden)]
    fn check_error(&self, ierr: i32) -> crate::Result<i32> {
        let mut ptr = std::ptr::null_mut();
        unsafe {
            bind_ceed::CeedVectorGetCeed(self.ptr, &mut ptr);
        }
        crate::check_error(ptr, ierr)
    }

    /// Returns the length of a Vector
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let vec = ceed.vector(10)?;
    ///
    /// let n = vec.length();
    /// assert_eq!(n, 10, "Incorrect length");
    /// # Ok(())
    /// # }
    /// ```
    pub fn length(&self) -> usize {
        let mut n = 0;
        unsafe { bind_ceed::CeedVectorGetLength(self.ptr, &mut n) };
        usize::try_from(n).unwrap()
    }

    /// Returns the length of a Vector
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let vec = ceed.vector(10)?;
    /// assert_eq!(vec.len(), 10, "Incorrect length");
    /// # Ok(())
    /// # }
    /// ```
    pub fn len(&self) -> usize {
        self.length()
    }

    /// Set the Vector to a constant value
    ///
    /// # arguments
    ///
    /// * `val` - Value to be used
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let len = 10;
    /// let mut vec = ceed.vector(len)?;
    ///
    /// let val = 42.0;
    /// vec.set_value(val)?;
    ///
    /// for v in vec.view()?.iter() {
    ///     assert_eq!(*v, val, "Value not set correctly");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_value(&mut self, value: crate::Scalar) -> crate::Result<i32> {
        let ierr = unsafe { bind_ceed::CeedVectorSetValue(self.ptr, value) };
        self.check_error(ierr)
    }

    /// Set values from a slice of the same length
    ///
    /// # arguments
    ///
    /// * `slice` - values to into self; length must match
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let mut vec = ceed.vector(4)?;
    /// vec.set_slice(&[10., 11., 12., 13.])?;
    ///
    /// for (i, v) in vec.view()?.iter().enumerate() {
    ///     assert_eq!(*v, 10. + i as Scalar, "Slice not set correctly");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_slice(&mut self, slice: &[crate::Scalar]) -> crate::Result<i32> {
        assert_eq!(self.length(), slice.len());
        let (host, copy_mode) = (
            crate::MemType::Host as bind_ceed::CeedMemType,
            crate::CopyMode::CopyValues as bind_ceed::CeedCopyMode,
        );
        let ierr = unsafe {
            bind_ceed::CeedVectorSetArray(
                self.ptr,
                host,
                copy_mode,
                slice.as_ptr() as *mut crate::Scalar,
            )
        };
        self.check_error(ierr)
    }

    /// Wrap a mutable slice in a Vector of the same length
    ///
    /// # arguments
    ///
    /// * `slice` - values to wrap in self; length must match
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let mut vec = ceed.vector(4)?;
    /// let mut array = [10., 11., 12., 13.];
    ///
    /// {
    ///     // `wrapper` holds a mutable reference to the wrapped slice
    ///     //   that is dropped when `wrapper` goes out of scope
    ///     let wrapper = vec.wrap_slice_mut(&mut array)?;
    ///     for (i, v) in vec.view()?.iter().enumerate() {
    ///         assert_eq!(*v, 10. + i as Scalar, "Slice not set correctly");
    ///     }
    ///
    ///     // This line will not compile, as the `wrapper` holds mutable
    ///     //   access to the `array`
    ///     // array[0] = 5.0;
    ///
    ///     // Changes here are reflected in the `array`
    ///     vec.set_value(5.0)?;
    ///     for v in vec.view()?.iter() {
    ///         assert_eq!(*v, 5.0 as Scalar, "Value not set correctly");
    ///     }
    /// }
    ///
    /// // 'array' remains changed
    /// for v in array.iter() {
    ///     assert_eq!(*v, 5.0 as Scalar, "Array not mutated correctly");
    /// }
    ///
    /// // While changes to `vec` no longer affect `array`
    /// vec.set_value(6.0)?;
    /// for v in array.iter() {
    ///     assert_eq!(*v, 5.0 as Scalar, "Array mutated without permission");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn wrap_slice_mut<'b>(
        &mut self,
        slice: &'b mut [crate::Scalar],
    ) -> crate::Result<VectorSliceWrapper<'b>> {
        crate::VectorSliceWrapper::from_vector_and_slice_mut(self, slice)
    }

    /// Sync the Vector to a specified memtype
    ///
    /// # arguments
    ///
    /// * `mtype` - Memtype to be synced
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let len = 10;
    /// let mut vec = ceed.vector(len)?;
    ///
    /// let val = 42.0;
    /// vec.set_value(val);
    /// vec.sync(MemType::Host)?;
    ///
    /// for v in vec.view()?.iter() {
    ///     assert_eq!(*v, val, "Value not set correctly");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn sync(&self, mtype: crate::MemType) -> crate::Result<i32> {
        let ierr =
            unsafe { bind_ceed::CeedVectorSyncArray(self.ptr, mtype as bind_ceed::CeedMemType) };
        self.check_error(ierr)
    }

    /// Create an immutable view
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let vec = ceed.vector_from_slice(&[10., 11., 12., 13.])?;
    ///
    /// let v = vec.view()?;
    /// assert_eq!(v[0..2], [10., 11.]);
    ///
    /// // It is valid to have multiple immutable views
    /// let w = vec.view()?;
    /// assert_eq!(v[1..], w[1..]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn view(&self) -> crate::Result<VectorView> {
        VectorView::new(self)
    }

    /// Create an mutable view
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let mut vec = ceed.vector_from_slice(&[10., 11., 12., 13.])?;
    ///
    /// {
    ///     let mut v = vec.view_mut()?;
    ///     v[2] = 9.;
    /// }
    ///
    /// let w = vec.view()?;
    /// assert_eq!(w[2], 9., "View did not mutate data");
    /// # Ok(())
    /// # }
    /// ```
    pub fn view_mut(&mut self) -> crate::Result<VectorViewMut> {
        VectorViewMut::new(self)
    }

    /// Return the norm of a Vector
    ///
    /// # arguments
    ///
    /// * `ntype` - Norm type One, Two, or Max
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let vec = ceed.vector_from_slice(&[1., 2., 3., 4.])?;
    ///
    /// let max_norm = vec.norm(NormType::Max)?;
    /// assert_eq!(max_norm, 4.0, "Incorrect Max norm");
    ///
    /// let l1_norm = vec.norm(NormType::One)?;
    /// assert_eq!(l1_norm, 10., "Incorrect L1 norm");
    ///
    /// let l2_norm = vec.norm(NormType::Two)?;
    /// assert!((l2_norm - 5.477) < 1e-3, "Incorrect L2 norm");
    /// # Ok(())
    /// # }
    /// ```
    pub fn norm(&self, ntype: crate::NormType) -> crate::Result<crate::Scalar> {
        let mut res: crate::Scalar = 0.0;
        let ierr = unsafe {
            bind_ceed::CeedVectorNorm(self.ptr, ntype as bind_ceed::CeedNormType, &mut res)
        };
        self.check_error(ierr)?;
        Ok(res)
    }

    /// Compute x = alpha x for a Vector
    ///
    /// # arguments
    ///
    /// * `alpha` - scaling factor
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let mut vec = ceed.vector_from_slice(&[0., 1., 2., 3., 4.])?;
    ///
    /// vec = vec.scale(-1.0)?;
    /// for (i, v) in vec.view()?.iter().enumerate() {
    ///     assert_eq!(*v, -(i as Scalar), "Value not set correctly");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[allow(unused_mut)]
    pub fn scale(mut self, alpha: crate::Scalar) -> crate::Result<Self> {
        let ierr = unsafe { bind_ceed::CeedVectorScale(self.ptr, alpha) };
        self.check_error(ierr)?;
        Ok(self)
    }

    /// Compute y = alpha x + y for a pair of Vectors
    ///
    /// # arguments
    ///
    /// * `alpha` - scaling factor
    /// * `x`     - second vector, must be different than self
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let x = ceed.vector_from_slice(&[0., 1., 2., 3., 4.])?;
    /// let mut y = ceed.vector_from_slice(&[0., 1., 2., 3., 4.])?;
    ///
    /// y = y.axpy(-0.5, &x)?;
    /// for (i, y) in y.view()?.iter().enumerate() {
    ///     assert_eq!(*y, (i as Scalar) / 2.0, "Value not set correctly");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[allow(unused_mut)]
    pub fn axpy(mut self, alpha: crate::Scalar, x: &crate::Vector) -> crate::Result<Self> {
        let ierr = unsafe { bind_ceed::CeedVectorAXPY(self.ptr, alpha, x.ptr) };
        self.check_error(ierr)?;
        Ok(self)
    }

    /// Compute y = alpha x + beta y for a pair of Vectors
    ///
    /// # arguments
    ///
    /// * `alpha` - first scaling factor
    /// * `beta`  - second scaling factor
    /// * `x`     - second vector, must be different than self
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let x = ceed.vector_from_slice(&[0., 1., 2., 3., 4.])?;
    /// let mut y = ceed.vector_from_slice(&[0., 1., 2., 3., 4.])?;
    ///
    /// y = y.axpby(-0.5, 1.0, &x)?;
    /// for (i, y) in y.view()?.iter().enumerate() {
    ///     assert_eq!(*y, (i as Scalar) * 1.5, "Value not set correctly");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[allow(unused_mut)]
    pub fn axpby(
        mut self,
        alpha: crate::Scalar,
        beta: crate::Scalar,
        x: &crate::Vector,
    ) -> crate::Result<Self> {
        let ierr = unsafe { bind_ceed::CeedVectorAXPBY(self.ptr, alpha, beta, x.ptr) };
        self.check_error(ierr)?;
        Ok(self)
    }

    /// Compute the pointwise multiplication w = x .* y for Vectors
    ///
    /// # arguments
    ///
    /// * `x` - first vector for product
    /// * `y` - second vector for product
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let mut w = ceed.vector_from_slice(&[0., 1., 2., 3., 4.])?;
    /// let x = ceed.vector_from_slice(&[0., 1., 2., 3., 4.])?;
    /// let y = ceed.vector_from_slice(&[0., 1., 2., 3., 4.])?;
    ///
    /// w = w.pointwise_mult(&x, &y)?;
    /// for (i, w) in w.view()?.iter().enumerate() {
    ///     assert_eq!(*w, (i as Scalar).powf(2.0), "Value not set correctly");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[allow(unused_mut)]
    pub fn pointwise_mult(mut self, x: &crate::Vector, y: &crate::Vector) -> crate::Result<Self> {
        let ierr = unsafe { bind_ceed::CeedVectorPointwiseMult(self.ptr, x.ptr, y.ptr) };
        self.check_error(ierr)?;
        Ok(self)
    }

    /// Compute the pointwise multiplication w = w .* x for Vectors
    ///
    /// # arguments
    ///
    /// * `x` - second vector for product
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let mut w = ceed.vector_from_slice(&[0., 1., 2., 3., 4.])?;
    /// let x = ceed.vector_from_slice(&[0., 1., 2., 3., 4.])?;
    ///
    /// w = w.pointwise_scale(&x)?;
    /// for (i, w) in w.view()?.iter().enumerate() {
    ///     assert_eq!(*w, (i as Scalar).powf(2.0), "Value not set correctly");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[allow(unused_mut)]
    pub fn pointwise_scale(mut self, x: &crate::Vector) -> crate::Result<Self> {
        let ierr = unsafe { bind_ceed::CeedVectorPointwiseMult(self.ptr, self.ptr, x.ptr) };
        self.check_error(ierr)?;
        Ok(self)
    }

    /// Compute the pointwise multiplication w = w .* w for a Vector
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let mut w = ceed.vector_from_slice(&[0., 1., 2., 3., 4.])?;
    ///
    /// w = w.pointwise_square()?;
    /// for (i, w) in w.view()?.iter().enumerate() {
    ///     assert_eq!(*w, (i as Scalar).powf(2.0), "Value not set correctly");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[allow(unused_mut)]
    pub fn pointwise_square(mut self) -> crate::Result<Self> {
        let ierr = unsafe { bind_ceed::CeedVectorPointwiseMult(self.ptr, self.ptr, self.ptr) };
        self.check_error(ierr)?;
        Ok(self)
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
    array: *const crate::Scalar,
}

impl<'a> VectorView<'a> {
    /// Construct a VectorView from a Vector reference
    fn new(vec: &'a Vector) -> crate::Result<Self> {
        let mut array = std::ptr::null();
        let ierr = unsafe {
            bind_ceed::CeedVectorGetArrayRead(
                vec.ptr,
                crate::MemType::Host as bind_ceed::CeedMemType,
                &mut array,
            )
        };
        vec.check_error(ierr)?;
        Ok(Self {
            vec: vec,
            array: array,
        })
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
    type Target = [crate::Scalar];
    fn deref(&self) -> &[crate::Scalar] {
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
    array: *mut crate::Scalar,
}

impl<'a> VectorViewMut<'a> {
    /// Construct a VectorViewMut from a Vector reference
    fn new(vec: &'a mut Vector) -> crate::Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let ierr = unsafe {
            bind_ceed::CeedVectorGetArray(
                vec.ptr,
                crate::MemType::Host as bind_ceed::CeedMemType,
                &mut ptr,
            )
        };
        vec.check_error(ierr)?;
        Ok(Self {
            vec: vec,
            array: ptr,
        })
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
    type Target = [crate::Scalar];
    fn deref(&self) -> &[crate::Scalar] {
        unsafe { std::slice::from_raw_parts(self.array, self.vec.len()) }
    }
}

// Mutable data access
impl<'a> DerefMut for VectorViewMut<'a> {
    fn deref_mut(&mut self) -> &mut [crate::Scalar] {
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
