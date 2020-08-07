use crate::prelude::*;

/// CeedVector context wrapper
pub struct Vector<'a> {
    ceed: &'a crate::Ceed,
    pub ptr: bind_ceed::CeedVector,
}

pub enum NormType {
    One,
    Two,
    Max,
}

impl<'a> Vector<'a> {
    pub fn create(ceed: &'a crate::Ceed, n: i32) -> Self {
        let mut ptr = std::ptr::null_mut();
        unsafe { bind_ceed::CeedVectorCreate(ceed.ptr, n, &mut ptr) };
        Self { ceed, ptr }
    }

    pub fn new(ceed: &'a crate::Ceed, ptr: bind_ceed::CeedVector) -> Self {
        Self { ceed, ptr }
    }

    /// Returns the length of a CeedVector
    ///
    /// ```
    /// let ceed = ceed::Ceed::init("/cpu/self/ref/serial");
    /// let vec = ceed.vector(10);
    /// let n = vec.length();
    /// assert!(n == 10);
    /// ```
    pub fn length(&self) -> i32 {
        let mut n = 0;
        unsafe { bind_ceed::CeedVectorGetLength(self.ptr, &mut n) };
        n
    }

    /// Set the CeedVector to a constant value
    ///
    /// # arguments
    ///
    /// * 'val' - Value to be used
    ///
    /// ```
    /// let ceed = ceed::Ceed::init("/cpu/self/ref/serial");
    /// let vec = ceed.vector(10);
    /// vec.set_value(42.0);
    /// ```
    pub fn set_value(&self, value: f64) {
        unsafe { bind_ceed::CeedVectorSetValue(self.ptr, value) };
    }

    pub fn sync(&self, mtype: crate::MemType) {
        unsafe { bind_ceed::CeedVectorSyncArray(self.ptr, mtype as bind_ceed::CeedMemType) };
    }

    /// Return the norm of a CeedVector
    ///
    /// # arguments
    ///
    /// * 'ntype' - Norm type CEED_NORM_1, CEED_NORM_2, or CEED_NORM_MAX
    ///
    /// ```
    /// let ceed = ceed::Ceed::init("/cpu/self/ref/serial");
    /// let vec = ceed.vector(10);
    /// vec.set_value(42.0);
    /// let n = vec.norm(ceed::vector::NormType::Max);
    /// assert!(n == 42.0)
    /// ```
    pub fn norm(&self, ntype: NormType) -> f64 {
        let mut res: f64 = 0.0;
        unsafe { bind_ceed::CeedVectorNorm(self.ptr, ntype as bind_ceed::CeedNormType, &mut res) };
        res
    }
}

/// Destructor
impl<'a> Drop for Vector<'a> {
    fn drop(&mut self) {
        unsafe {
            bind_ceed::CeedVectorDestroy(&mut self.ptr);
        }
    }
}
