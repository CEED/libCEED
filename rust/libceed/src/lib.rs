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

// Fenced `rust` code blocks included from README.md are executed as part of doctests.
#![doc = include_str!("../README.md")]
// -----------------------------------------------------------------------------
// Exceptions
// -----------------------------------------------------------------------------
#![allow(non_snake_case)]

// -----------------------------------------------------------------------------
// Crate prelude
// -----------------------------------------------------------------------------
use crate::prelude::*;
use std::sync::Once;

pub mod prelude {
    pub use crate::{
        basis::{self, Basis, BasisOpt},
        elem_restriction::{self, ElemRestriction, ElemRestrictionOpt},
        operator::{self, CompositeOperator, Operator, OperatorField},
        qfunction::{
            self, QFunction, QFunctionByName, QFunctionField, QFunctionInputs, QFunctionOpt,
            QFunctionOutputs,
        },
        vector::{self, Vector, VectorOpt, VectorSliceWrapper},
        ElemTopology, EvalMode, MemType, NormType, QuadMode, Scalar, TransposeMode,
        CEED_STRIDES_BACKEND, EPSILON, MAX_QFUNCTION_FIELDS,
    };
    pub(crate) use libceed_sys::bind_ceed;
    pub(crate) use std::convert::TryFrom;
    pub(crate) use std::ffi::{CStr, CString};
    pub(crate) use std::fmt;
    pub(crate) use std::marker::PhantomData;
}

// -----------------------------------------------------------------------------
// Modules
// -----------------------------------------------------------------------------
pub mod basis;
pub mod elem_restriction;
pub mod operator;
pub mod qfunction;
pub mod vector;

// -----------------------------------------------------------------------------
// Typedef for scalar
// -----------------------------------------------------------------------------
pub type Scalar = bind_ceed::CeedScalar;

// -----------------------------------------------------------------------------
// Constants for library
// -----------------------------------------------------------------------------
const MAX_BUFFER_LENGTH: u64 = 4096;
pub const MAX_QFUNCTION_FIELDS: usize = 16;
pub const CEED_STRIDES_BACKEND: [i32; 3] = [0; 3];
pub const EPSILON: crate::Scalar = bind_ceed::CEED_EPSILON as crate::Scalar;

// -----------------------------------------------------------------------------
// Enums for libCEED
// -----------------------------------------------------------------------------
#[derive(Clone, Copy, PartialEq, Eq)]
/// Many Ceed interfaces take or return pointers to memory.  This enum is used to
/// specify where the memory being provided or requested must reside.
pub enum MemType {
    Host = bind_ceed::CeedMemType_CEED_MEM_HOST as isize,
    Device = bind_ceed::CeedMemType_CEED_MEM_DEVICE as isize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
// OwnPointer will not be used by user but is included for internal use
#[allow(dead_code)]
/// Conveys ownership status of arrays passed to Ceed interfaces.
pub(crate) enum CopyMode {
    CopyValues = bind_ceed::CeedCopyMode_CEED_COPY_VALUES as isize,
    UsePointer = bind_ceed::CeedCopyMode_CEED_USE_POINTER as isize,
    OwnPointer = bind_ceed::CeedCopyMode_CEED_OWN_POINTER as isize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
/// Denotes type of vector norm to be computed
pub enum NormType {
    One = bind_ceed::CeedNormType_CEED_NORM_1 as isize,
    Two = bind_ceed::CeedNormType_CEED_NORM_2 as isize,
    Max = bind_ceed::CeedNormType_CEED_NORM_MAX as isize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
/// Denotes whether a linear transformation or its transpose should be applied
pub enum TransposeMode {
    NoTranspose = bind_ceed::CeedTransposeMode_CEED_NOTRANSPOSE as isize,
    Transpose = bind_ceed::CeedTransposeMode_CEED_TRANSPOSE as isize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
/// Type of quadrature; also used for location of nodes
pub enum QuadMode {
    Gauss = bind_ceed::CeedQuadMode_CEED_GAUSS as isize,
    GaussLobatto = bind_ceed::CeedQuadMode_CEED_GAUSS_LOBATTO as isize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
/// Type of basis shape to create non-tensor H1 element basis
pub enum ElemTopology {
    Line = bind_ceed::CeedElemTopology_CEED_LINE as isize,
    Triangle = bind_ceed::CeedElemTopology_CEED_TRIANGLE as isize,
    Quad = bind_ceed::CeedElemTopology_CEED_QUAD as isize,
    Tet = bind_ceed::CeedElemTopology_CEED_TET as isize,
    Pyramid = bind_ceed::CeedElemTopology_CEED_PYRAMID as isize,
    Prism = bind_ceed::CeedElemTopology_CEED_PRISM as isize,
    Hex = bind_ceed::CeedElemTopology_CEED_HEX as isize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Basis evaluation mode
pub enum EvalMode {
    None = bind_ceed::CeedEvalMode_CEED_EVAL_NONE as isize,
    Interp = bind_ceed::CeedEvalMode_CEED_EVAL_INTERP as isize,
    Grad = bind_ceed::CeedEvalMode_CEED_EVAL_GRAD as isize,
    Div = bind_ceed::CeedEvalMode_CEED_EVAL_DIV as isize,
    Curl = bind_ceed::CeedEvalMode_CEED_EVAL_CURL as isize,
    Weight = bind_ceed::CeedEvalMode_CEED_EVAL_WEIGHT as isize,
}
impl EvalMode {
    pub(crate) fn from_u32(value: u32) -> EvalMode {
        match value {
            bind_ceed::CeedEvalMode_CEED_EVAL_NONE => EvalMode::None,
            bind_ceed::CeedEvalMode_CEED_EVAL_INTERP => EvalMode::Interp,
            bind_ceed::CeedEvalMode_CEED_EVAL_GRAD => EvalMode::Grad,
            bind_ceed::CeedEvalMode_CEED_EVAL_DIV => EvalMode::Div,
            bind_ceed::CeedEvalMode_CEED_EVAL_CURL => EvalMode::Curl,
            bind_ceed::CeedEvalMode_CEED_EVAL_WEIGHT => EvalMode::Weight,
            _ => panic!("Unknown value: {}", value),
        }
    }
}

// -----------------------------------------------------------------------------
// Ceed error
// -----------------------------------------------------------------------------
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub struct Error {
    pub message: String,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

// -----------------------------------------------------------------------------
// Internal error checker
// -----------------------------------------------------------------------------
#[doc(hidden)]
pub(crate) fn check_error(ceed_ptr: bind_ceed::Ceed, ierr: i32) -> Result<i32> {
    // Return early if code is clean
    if ierr == bind_ceed::CeedErrorType_CEED_ERROR_SUCCESS {
        return Ok(ierr);
    }
    // Retrieve error message
    let mut ptr: *const std::os::raw::c_char = std::ptr::null_mut();
    let c_str = unsafe {
        bind_ceed::CeedGetErrorMessage(ceed_ptr, &mut ptr);
        std::ffi::CStr::from_ptr(ptr)
    };
    let message = c_str.to_string_lossy().to_string();
    // Panic if negative code, otherwise return error
    if ierr < bind_ceed::CeedErrorType_CEED_ERROR_SUCCESS {
        panic!("{}", message);
    }
    Err(Error { message })
}

// -----------------------------------------------------------------------------
// Ceed error handler
// -----------------------------------------------------------------------------
pub enum ErrorHandler {
    ErrorAbort,
    ErrorExit,
    ErrorReturn,
    ErrorStore,
}

// -----------------------------------------------------------------------------
// Ceed context wrapper
// -----------------------------------------------------------------------------
/// A Ceed is a library context representing control of a logical hardware
/// resource.
#[derive(Debug)]
pub struct Ceed {
    ptr: bind_ceed::Ceed,
}

// -----------------------------------------------------------------------------
// Destructor
// -----------------------------------------------------------------------------
impl Drop for Ceed {
    fn drop(&mut self) {
        unsafe {
            bind_ceed::CeedDestroy(&mut self.ptr);
        }
    }
}

// -----------------------------------------------------------------------------
// Cloning
// -----------------------------------------------------------------------------
impl Clone for Ceed {
    /// Perform a shallow clone of a Ceed context
    ///
    /// ```
    /// let ceed = libceed::Ceed::init("/cpu/self/ref/serial");
    /// let ceed_clone = ceed.clone();
    ///
    /// println!("{}", ceed);
    /// println!("{}", ceed_clone);
    /// ```
    fn clone(&self) -> Self {
        let mut ptr_clone = std::ptr::null_mut();
        let ierr = unsafe { bind_ceed::CeedReferenceCopy(self.ptr, &mut ptr_clone) };
        self.check_error(ierr).expect("failed to clone Ceed");
        Self { ptr: ptr_clone }
    }
}

// -----------------------------------------------------------------------------
// Display
// -----------------------------------------------------------------------------
impl fmt::Display for Ceed {
    /// View a Ceed
    ///
    /// ```
    /// let ceed = libceed::Ceed::init("/cpu/self/ref/serial");
    /// println!("{}", ceed);
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut ptr = std::ptr::null_mut();
        let mut sizeloc = crate::MAX_BUFFER_LENGTH;
        let cstring = unsafe {
            let file = bind_ceed::open_memstream(&mut ptr, &mut sizeloc);
            bind_ceed::CeedView(self.ptr, file);
            bind_ceed::fclose(file);
            CString::from_raw(ptr)
        };
        cstring.to_string_lossy().fmt(f)
    }
}

static REGISTER: Once = Once::new();

// -----------------------------------------------------------------------------
// Object constructors
// -----------------------------------------------------------------------------
impl Ceed {
    /// Returns a Ceed context initialized with the specified resource
    ///
    /// # arguments
    ///
    /// * `resource` - Resource to use, e.g., "/cpu/self"
    ///
    /// ```
    /// let ceed = libceed::Ceed::init("/cpu/self/ref/serial");
    /// ```
    pub fn init(resource: &str) -> Self {
        Ceed::init_with_error_handler(resource, ErrorHandler::ErrorStore)
    }

    /// Returns a Ceed context initialized with the specified resource
    ///
    /// # arguments
    ///
    /// * `resource` - Resource to use, e.g., "/cpu/self"
    ///
    /// ```
    /// let ceed = libceed::Ceed::init_with_error_handler(
    ///     "/cpu/self/ref/serial",
    ///     libceed::ErrorHandler::ErrorAbort,
    /// );
    /// ```
    pub fn init_with_error_handler(resource: &str, handler: ErrorHandler) -> Self {
        REGISTER.call_once(|| unsafe {
            bind_ceed::CeedRegisterAll();
            bind_ceed::CeedQFunctionRegisterAll();
        });

        // Convert to C string
        let c_resource = CString::new(resource).expect("CString::new failed");

        // Get error handler pointer
        let eh = match handler {
            ErrorHandler::ErrorAbort => bind_ceed::CeedErrorAbort,
            ErrorHandler::ErrorExit => bind_ceed::CeedErrorExit,
            ErrorHandler::ErrorReturn => bind_ceed::CeedErrorReturn,
            ErrorHandler::ErrorStore => bind_ceed::CeedErrorStore,
        };

        // Call to libCEED
        let mut ptr = std::ptr::null_mut();
        let mut ierr = unsafe { bind_ceed::CeedInit(c_resource.as_ptr() as *const i8, &mut ptr) };
        if ierr != 0 {
            panic!("Error initializing backend resource: {}", resource)
        }
        ierr = unsafe { bind_ceed::CeedSetErrorHandler(ptr, Some(eh)) };
        let ceed = Ceed { ptr };
        ceed.check_error(ierr).unwrap();
        ceed
    }

    /// Default initializer for testing
    #[doc(hidden)]
    pub fn default_init() -> Self {
        // Convert to C string
        let resource = "/cpu/self/ref/serial";
        crate::Ceed::init(resource)
    }

    /// Internal error checker
    #[doc(hidden)]
    fn check_error(&self, ierr: i32) -> Result<i32> {
        // Return early if code is clean
        if ierr == bind_ceed::CeedErrorType_CEED_ERROR_SUCCESS {
            return Ok(ierr);
        }
        // Retrieve error message
        let mut ptr: *const std::os::raw::c_char = std::ptr::null_mut();
        let c_str = unsafe {
            bind_ceed::CeedGetErrorMessage(self.ptr, &mut ptr);
            std::ffi::CStr::from_ptr(ptr)
        };
        let message = c_str.to_string_lossy().to_string();
        // Panic if negative code, otherwise return error
        if ierr < bind_ceed::CeedErrorType_CEED_ERROR_SUCCESS {
            panic!("{}", message);
        }
        Err(Error { message })
    }

    /// Returns full resource name for a Ceed context
    ///
    /// ```
    /// let ceed = libceed::Ceed::init("/cpu/self/ref/serial");
    /// let resource = ceed.resource();
    ///
    /// assert_eq!(resource, "/cpu/self/ref/serial".to_string())
    /// ```
    pub fn resource(&self) -> String {
        let mut ptr: *const std::os::raw::c_char = std::ptr::null_mut();
        let c_str = unsafe {
            bind_ceed::CeedGetResource(self.ptr, &mut ptr);
            std::ffi::CStr::from_ptr(ptr)
        };
        c_str.to_string_lossy().to_string()
    }

    /// Returns a CeedVector of the specified length (does not allocate memory)
    ///
    /// # arguments
    ///
    /// * `n` - Length of vector
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let vec = ceed.vector(10)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn vector<'a>(&self, n: usize) -> Result<Vector<'a>> {
        Vector::create(self, n)
    }

    /// Create a Vector initialized with the data (copied) from a slice
    ///
    /// # arguments
    ///
    /// * `slice` - Slice containing data
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let vec = ceed.vector_from_slice(&[1., 2., 3.])?;
    /// assert_eq!(vec.length(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn vector_from_slice<'a>(&self, slice: &[crate::Scalar]) -> Result<Vector<'a>> {
        Vector::from_slice(self, slice)
    }

    /// Returns a ElemRestriction
    ///
    /// # arguments
    ///
    /// * `nelem`      - Number of elements described in the offsets array
    /// * `elemsize`   - Size (number of "nodes") per element
    /// * `ncomp`      - Number of field components per interpolation node (1
    ///                    for scalar fields)
    /// * `compstride` - Stride between components for the same Lvector "node".
    ///                    Data for node `i`, component `j`, element `k` can be
    ///                    found in the Lvector at index
    ///                    `offsets[i + k*elemsize] + j*compstride`.
    /// * `lsize`      - The size of the Lvector. This vector may be larger
    ///                    than the elements and fields given by this
    ///                    restriction.
    /// * `mtype`     - Memory type of the offsets array, see CeedMemType
    /// * `offsets`    - Array of shape `[nelem, elemsize]`. Row `i` holds the
    ///                    ordered list of the offsets (into the input CeedVector)
    ///                    for the unknowns corresponding to element `i`, where
    ///                    `0 <= i < nelem`. All offsets must be in the range
    ///                    `[0, lsize - 1]`.
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let nelem = 3;
    /// let mut ind: Vec<i32> = vec![0; 2 * nelem];
    /// for i in 0..nelem {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed.elem_restriction(nelem, 2, 1, 1, nelem + 1, MemType::Host, &ind)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn elem_restriction<'a>(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        compstride: usize,
        lsize: usize,
        mtype: MemType,
        offsets: &[i32],
    ) -> Result<ElemRestriction<'a>> {
        ElemRestriction::create(
            self, nelem, elemsize, ncomp, compstride, lsize, mtype, offsets,
        )
    }

    /// Returns a ElemRestriction
    ///
    /// # arguments
    ///
    /// * `nelem`      - Number of elements described in the offsets array
    /// * `elemsize`   - Size (number of "nodes") per element
    /// * `ncomp`      - Number of field components per interpolation node (1
    ///                    for scalar fields)
    /// * `compstride` - Stride between components for the same Lvector "node".
    ///                    Data for node `i`, component `j`, element `k` can be
    ///                    found in the Lvector at index
    ///                    `offsets[i + k*elemsize] + j*compstride`.
    /// * `lsize`      - The size of the Lvector. This vector may be larger
    ///   than the elements and fields given by this restriction.
    /// * `strides`   - Array for strides between `[nodes, components, elements]`.
    ///                   Data for node `i`, component `j`, element `k` can be
    ///                   found in the Lvector at index
    ///                   `i*strides[0] + j*strides[1] + k*strides[2]`.
    ///                   CEED_STRIDES_BACKEND may be used with vectors created
    ///                   by a Ceed backend.
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let nelem = 3;
    /// let strides: [i32; 3] = [1, 2, 2];
    /// let r = ceed.strided_elem_restriction(nelem, 2, 1, nelem * 2, strides)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn strided_elem_restriction<'a>(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        lsize: usize,
        strides: [i32; 3],
    ) -> Result<ElemRestriction<'a>> {
        ElemRestriction::create_strided(self, nelem, elemsize, ncomp, lsize, strides)
    }

    /// Returns a tensor-product basis
    ///
    /// # arguments
    ///
    /// * `dim`       - Topological dimension of element
    /// * `ncomp`     - Number of field components (1 for scalar fields)
    /// * `P1d`       - Number of Gauss-Lobatto nodes in one dimension.  The
    ///                   polynomial degree of the resulting `Q_k` element is
    ///                   `k=P-1`.
    /// * `Q1d`       - Number of quadrature points in one dimension
    /// * `interp1d`  - Row-major `(Q1d * P1d)` matrix expressing the values of
    ///                   nodal basis functions at quadrature points
    /// * `grad1d`    - Row-major `(Q1d * P1d)` matrix expressing derivatives of
    ///                   nodal basis functions at quadrature points
    /// * `qref1d`    - Array of length `Q1d` holding the locations of quadrature
    ///                   points on the 1D reference element `[-1, 1]`
    /// * `qweight1d` - Array of length `Q1d` holding the quadrature weights on
    ///                   the reference element
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let interp1d  = [ 0.62994317,  0.47255875, -0.14950343,  0.04700152,
    ///                  -0.07069480,  0.97297619,  0.13253993, -0.03482132,
    ///                  -0.03482132,  0.13253993,  0.97297619, -0.07069480,
    ///                   0.04700152, -0.14950343,  0.47255875,  0.62994317];
    /// let grad1d    = [-2.34183742,  2.78794489, -0.63510411,  0.18899664,
    ///                  -0.51670214, -0.48795249,  1.33790510, -0.33325047,
    //                    0.33325047, -1.33790510,  0.48795249,  0.51670214,
    ///                  -0.18899664,  0.63510411, -2.78794489,  2.34183742];
    /// let qref1d    = [-0.86113631, -0.33998104,  0.33998104,  0.86113631];
    /// let qweight1d = [ 0.34785485,  0.65214515,  0.65214515,  0.34785485];
    /// let b = ceed.
    /// basis_tensor_H1(2, 1, 4, 4, &interp1d, &grad1d, &qref1d, &qweight1d)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn basis_tensor_H1<'a>(
        &self,
        dim: usize,
        ncomp: usize,
        P1d: usize,
        Q1d: usize,
        interp1d: &[crate::Scalar],
        grad1d: &[crate::Scalar],
        qref1d: &[crate::Scalar],
        qweight1d: &[crate::Scalar],
    ) -> Result<Basis<'a>> {
        Basis::create_tensor_H1(
            self, dim, ncomp, P1d, Q1d, interp1d, grad1d, qref1d, qweight1d,
        )
    }

    /// Returns a tensor-product Lagrange basis
    ///
    /// # arguments
    ///
    /// * `dim`   - Topological dimension of element
    /// * `ncomp` - Number of field components (1 for scalar fields)
    /// * `P`     - Number of Gauss-Lobatto nodes in one dimension.  The
    ///               polynomial degree of the resulting `Q_k` element is `k=P-1`.
    /// * `Q`     - Number of quadrature points in one dimension
    /// * `qmode` - Distribution of the `Q` quadrature points (affects order of
    ///               accuracy for the quadrature)
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let b = ceed.basis_tensor_H1_Lagrange(2, 1, 3, 4, QuadMode::Gauss)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn basis_tensor_H1_Lagrange<'a>(
        &self,
        dim: usize,
        ncomp: usize,
        P: usize,
        Q: usize,
        qmode: QuadMode,
    ) -> Result<Basis<'a>> {
        Basis::create_tensor_H1_Lagrange(self, dim, ncomp, P, Q, qmode)
    }

    /// Returns a tensor-product basis
    ///
    /// # arguments
    ///
    /// * `topo`    - Topology of element, e.g. hypercube, simplex, ect
    /// * `ncomp`   - Number of field components (1 for scalar fields)
    /// * `nnodes`  - Total number of nodes
    /// * `nqpts`   - Total number of quadrature points
    /// * `interp`  - Row-major `(nqpts * nnodes)` matrix expressing the values of
    ///                 nodal basis functions at quadrature points
    /// * `grad`    - Row-major `(nqpts * dim * nnodes)` matrix expressing
    ///                 derivatives of nodal basis functions at quadrature points
    /// * `qref`    - Array of length `nqpts` holding the locations of quadrature
    ///                 points on the reference element `[-1, 1]`
    /// * `qweight` - Array of length `nqpts` holding the quadrature weights on
    ///                 the reference element
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let interp = [
    ///     0.12000000,
    ///     0.48000000,
    ///     -0.12000000,
    ///     0.48000000,
    ///     0.16000000,
    ///     -0.12000000,
    ///     -0.12000000,
    ///     0.48000000,
    ///     0.12000000,
    ///     0.16000000,
    ///     0.48000000,
    ///     -0.12000000,
    ///     -0.11111111,
    ///     0.44444444,
    ///     -0.11111111,
    ///     0.44444444,
    ///     0.44444444,
    ///     -0.11111111,
    ///     -0.12000000,
    ///     0.16000000,
    ///     -0.12000000,
    ///     0.48000000,
    ///     0.48000000,
    ///     0.12000000,
    /// ];
    /// let grad = [
    ///     -1.40000000,
    ///     1.60000000,
    ///     -0.20000000,
    ///     -0.80000000,
    ///     0.80000000,
    ///     0.00000000,
    ///     0.20000000,
    ///     -1.60000000,
    ///     1.40000000,
    ///     -0.80000000,
    ///     0.80000000,
    ///     0.00000000,
    ///     -0.33333333,
    ///     0.00000000,
    ///     0.33333333,
    ///     -1.33333333,
    ///     1.33333333,
    ///     0.00000000,
    ///     0.20000000,
    ///     0.00000000,
    ///     -0.20000000,
    ///     -2.40000000,
    ///     2.40000000,
    ///     0.00000000,
    ///     -1.40000000,
    ///     -0.80000000,
    ///     0.00000000,
    ///     1.60000000,
    ///     0.80000000,
    ///     -0.20000000,
    ///     0.20000000,
    ///     -2.40000000,
    ///     0.00000000,
    ///     0.00000000,
    ///     2.40000000,
    ///     -0.20000000,
    ///     -0.33333333,
    ///     -1.33333333,
    ///     0.00000000,
    ///     0.00000000,
    ///     1.33333333,
    ///     0.33333333,
    ///     0.20000000,
    ///     -0.80000000,
    ///     0.00000000,
    ///     -1.60000000,
    ///     0.80000000,
    ///     1.40000000,
    /// ];
    /// let qref = [
    ///     0.20000000, 0.60000000, 0.33333333, 0.20000000, 0.20000000, 0.20000000, 0.33333333,
    ///     0.60000000,
    /// ];
    /// let qweight = [0.26041667, 0.26041667, -0.28125000, 0.26041667];
    /// let b = ceed.basis_H1(
    ///     ElemTopology::Triangle,
    ///     1,
    ///     6,
    ///     4,
    ///     &interp,
    ///     &grad,
    ///     &qref,
    ///     &qweight,
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn basis_H1<'a>(
        &self,
        topo: ElemTopology,
        ncomp: usize,
        nnodes: usize,
        nqpts: usize,
        interp: &[crate::Scalar],
        grad: &[crate::Scalar],
        qref: &[crate::Scalar],
        qweight: &[crate::Scalar],
    ) -> Result<Basis<'a>> {
        Basis::create_H1(
            self, topo, ncomp, nnodes, nqpts, interp, grad, qref, qweight,
        )
    }

    #[cfg_attr(feature = "katexit", katexit::katexit)]
    /// Returns a CeedQFunction for evaluating interior (volumetric) terms
    ///
    /// $$
    /// v^T F(u) \sim \int_\Omega v^T f_0(u, \nabla u) + (\nabla v)^T f_1(u, \nabla u)
    /// $$
    ///
    /// # arguments
    ///
    /// * `vlength` - Vector length. Caller must ensure that number of
    ///                 quadrature points is a multiple of vlength.
    /// * `f`       - Boxed closure to evaluate action at quadrature points.
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let mut user_f = |[u, weights, ..]: QFunctionInputs, [v, ..]: QFunctionOutputs| {
    ///     // Iterate over quadrature points
    ///     v.iter_mut()
    ///         .zip(u.iter().zip(weights.iter()))
    ///         .for_each(|(v, (u, w))| *v = u * w);
    ///
    ///     // Return clean error code
    ///     0
    /// };
    ///
    /// let qf = ceed.q_function_interior(1, Box::new(user_f))?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn q_function_interior<'a>(
        &self,
        vlength: usize,
        f: Box<qfunction::QFunctionUserClosure>,
    ) -> Result<QFunction<'a>> {
        QFunction::create(self, vlength, f)
    }

    /// Returns a CeedQFunction for evaluating interior (volumetric) terms
    /// created by name
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let qf = ceed.q_function_interior_by_name("Mass1DBuild")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn q_function_interior_by_name<'a>(&self, name: &str) -> Result<QFunctionByName<'a>> {
        QFunctionByName::create(self, name)
    }

    /// Returns a Operator and associate a QFunction. A Basis and
    /// ElemRestriction can be   associated with QFunction fields with
    /// set_field().
    ///
    /// * `qf`   - QFunction defining the action of the operator at quadrature
    ///              points
    /// * `dqf`  - QFunction defining the action of the Jacobian of the qf (or
    ///              qfunction_none)
    /// * `dqfT` - QFunction defining the action of the transpose of the
    ///              Jacobian of the qf (or qfunction_none)
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let qf = ceed.q_function_interior_by_name("Mass1DBuild")?;
    /// let op = ceed.operator(&qf, QFunctionOpt::None, QFunctionOpt::None)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn operator<'a, 'b>(
        &self,
        qf: impl Into<QFunctionOpt<'b>>,
        dqf: impl Into<QFunctionOpt<'b>>,
        dqfT: impl Into<QFunctionOpt<'b>>,
    ) -> Result<Operator<'a>> {
        Operator::create(self, qf, dqf, dqfT)
    }

    /// Returns an Operator that composes the action of several Operators
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let op = ceed.composite_operator()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn composite_operator<'a>(&self) -> Result<CompositeOperator<'a>> {
        CompositeOperator::create(self)
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn ceed_t501() -> Result<()> {
        let resource = "/cpu/self/ref/blocked";
        let ceed = Ceed::init(resource);
        let nelem = 4;
        let p = 3;
        let q = 4;
        let ndofs = p * nelem - nelem + 1;

        // Vectors
        let x = ceed.vector_from_slice(&[-1., -0.5, 0.0, 0.5, 1.0])?;
        let mut qdata = ceed.vector(nelem * q)?;
        qdata.set_value(0.0)?;
        let mut u = ceed.vector(ndofs)?;
        u.set_value(1.0)?;
        let mut v = ceed.vector(ndofs)?;
        v.set_value(0.0)?;

        // Restrictions
        let mut indx: Vec<i32> = vec![0; 2 * nelem];
        for i in 0..nelem {
            indx[2 * i + 0] = i as i32;
            indx[2 * i + 1] = (i + 1) as i32;
        }
        let rx = ceed.elem_restriction(nelem, 2, 1, 1, nelem + 1, MemType::Host, &indx)?;
        let mut indu: Vec<i32> = vec![0; p * nelem];
        for i in 0..nelem {
            indu[p * i + 0] = i as i32;
            indu[p * i + 1] = (i + 1) as i32;
            indu[p * i + 2] = (i + 2) as i32;
        }
        let ru = ceed.elem_restriction(nelem, 3, 1, 1, ndofs, MemType::Host, &indu)?;
        let strides: [i32; 3] = [1, q as i32, q as i32];
        let rq = ceed.strided_elem_restriction(nelem, q, 1, q * nelem, strides)?;

        // Bases
        let bx = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, QuadMode::Gauss)?;
        let bu = ceed.basis_tensor_H1_Lagrange(1, 1, p, q, QuadMode::Gauss)?;

        // Build quadrature data
        let qf_build = ceed.q_function_interior_by_name("Mass1DBuild")?;
        ceed.operator(&qf_build, QFunctionOpt::None, QFunctionOpt::None)?
            .field("dx", &rx, &bx, VectorOpt::Active)?
            .field("weights", ElemRestrictionOpt::None, &bx, VectorOpt::None)?
            .field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active)?
            .apply(&x, &mut qdata)?;

        // Mass operator
        let qf_mass = ceed.q_function_interior_by_name("MassApply")?;
        let op_mass = ceed
            .operator(&qf_mass, QFunctionOpt::None, QFunctionOpt::None)?
            .field("u", &ru, &bu, VectorOpt::Active)?
            .field("qdata", &rq, BasisOpt::Collocated, &qdata)?
            .field("v", &ru, &bu, VectorOpt::Active)?
            .check()?;

        v.set_value(0.0)?;
        op_mass.apply(&u, &mut v)?;

        // Check
        let sum: Scalar = v.view()?.iter().sum();
        assert!(
            (sum - 2.0).abs() < 1e-15,
            "Incorrect interval length computed"
        );
        Ok(())
    }

    #[test]
    fn test_ceed_t501() {
        assert!(ceed_t501().is_ok());
    }
}

// -----------------------------------------------------------------------------
