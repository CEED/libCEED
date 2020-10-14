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

//! # libCEED Rust Interface
//!
//! This is the documentation for the high level libCEED Rust interface.
//! See the full libCEED user manual [here](https://libceed.readthedocs.io).

// -----------------------------------------------------------------------------
// Exceptions
// -----------------------------------------------------------------------------
#![allow(non_snake_case)]

use crate::prelude::*;

pub mod prelude {
    pub(crate) mod bind_ceed {
        #![allow(non_upper_case_globals)]
        #![allow(non_camel_case_types)]
        #![allow(dead_code)]
        include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
    }
    pub use crate::basis::{self, BasisOpt};
    pub use crate::elem_restriction::{self, ElemRestrictionOpt};
    pub use crate::qfunction::{self, QFunctionOpt};
    pub use crate::vector::{self, VectorOpt};
    pub(crate) use std::ffi::CString;
    pub(crate) use std::fmt;
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
// Constants for library interally
// -----------------------------------------------------------------------------
const MAX_BUFFER_LENGTH: u64 = 4096;
const MAX_QFUNCTION_FIELDS: usize = 16;

// -----------------------------------------------------------------------------
// Enums for libCEED
// -----------------------------------------------------------------------------
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MemType {
    Host,
    Device,
}

#[derive(Clone, Copy, PartialEq, Eq)]
// OwnPointer will not be used but is included for completeness
#[allow(dead_code)]
pub(crate) enum CopyMode {
    CopyValues,
    UsePointer,
    OwnPointer,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    One,
    Two,
    Max,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TransposeMode {
    NoTranspose,
    Transpose,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum QuadMode {
    Gauss,
    GaussLobatto,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ElemTopology {
    Line,
    Triangle,
    Quad,
    Tet,
    Pyramid,
    Prism,
    Hex,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum EvalMode {
    None,
    Interp,
    Grad,
    Div,
    Curl,
    Weight,
}

// -----------------------------------------------------------------------------
// Ceed context wrapper
// -----------------------------------------------------------------------------
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
// Display
// -----------------------------------------------------------------------------
impl fmt::Display for Ceed {
    /// View a Ceed
    ///
    /// ```
    /// let ceed = ceed::Ceed::init("/cpu/self/ref/serial");
    /// println!("{}", ceed);
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut ptr = std::ptr::null_mut();
        let mut sizeloc = crate::MAX_BUFFER_LENGTH;
        let file = unsafe { bind_ceed::open_memstream(&mut ptr, &mut sizeloc) };
        unsafe { bind_ceed::CeedView(self.ptr, file) };
        unsafe { bind_ceed::fclose(file) };
        let cstring = unsafe { CString::from_raw(ptr) };
        cstring.to_string_lossy().fmt(f)
    }
}

// -----------------------------------------------------------------------------
// Object constructors
// -----------------------------------------------------------------------------
impl Ceed {
    /// Returns a Ceed context initalized with the specified resource
    ///
    /// # arguments
    ///
    /// * 'resource' - Resource to use, e.g., "/cpu/self"
    ///
    /// ```
    /// let ceed = ceed::Ceed::init("/cpu/self/ref/serial");
    /// ```
    pub fn init(resource: &str) -> Self {
        // Convert to C string
        let c_resource = CString::new(resource).expect("CString::new failed");

        // Call to libCEED
        let mut ptr = std::ptr::null_mut();
        unsafe { bind_ceed::CeedInit(c_resource.as_ptr() as *const i8, &mut ptr) };
        Ceed { ptr }
    }

    // Default initalizer for testing
    pub fn default_init() -> Self {
        // Convert to C string
        let resource = "/cpu/self/ref/serial";
        crate::Ceed::init(resource)
    }

    /// Returns a CeedVector of the specified length (does not allocate memory)
    ///
    /// # arguments
    ///
    /// * 'n' - Length of vector
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let vec = ceed.vector(10);
    /// ```
    pub fn vector(&self, n: usize) -> crate::vector::Vector {
        crate::vector::Vector::create(self, n)
    }

    /// Create a Vector initialized with the data (copied) from a slice
    ///
    /// # arguments
    ///
    /// * 'slice' - Slice containing data
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let vec = ceed.vector_from_slice(&[1., 2., 3.]);
    /// assert_eq!(vec.length(), 3);
    /// ```
    pub fn vector_from_slice(&self, slice: &[f64]) -> crate::vector::Vector {
        crate::vector::Vector::from_slice(self, slice)
    }

    /// Returns a ElemRestriction
    ///
    /// # arguments
    ///
    /// * 'nelem'      - Number of elements described in the @a offsets array
    /// * 'elemsize'   - Size (number of "nodes") per element
    /// * 'ncomp'      - Number of field components per interpolation node (1
    ///                    for scalar fields)
    /// * 'compstride' - Stride between components for the same L-vector "node".
    ///                    Data for node i, component j, element k can be found
    ///                    in the L-vector at index
    ///                    offsets[i + k*elemsize] + j*compstride.
    /// * 'lsize'      - The size of the L-vector. This vector may be larger
    ///                    than the elements and fields given by this
    ///                    restriction.
    /// * 'mtype'     - Memory type of the @a offsets array, see CeedMemType
    /// * 'offsets'    - Array of shape [@a nelem, @a elemsize]. Row i holds the
    ///                    ordered list of the offsets (into the input CeedVector)
    ///                    for the unknowns corresponding to element i, where
    ///                    0 <= i < @a nelem. All offsets must be in the range
    ///                    [0, @a lsize - 1].
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let nelem = 3;
    /// let mut ind : Vec<i32> = vec![0; 2*nelem];
    /// for i in 0..nelem {
    ///   ind[2*i+0] = i as i32;
    ///   ind[2*i+1] = (i+1) as i32;
    /// }
    /// let r = ceed.elem_restriction(nelem, 2, 1, 1, nelem+1, ceed::MemType::Host, &ind);
    /// ```
    pub fn elem_restriction(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        compstride: usize,
        lsize: usize,
        mtype: MemType,
        offsets: &Vec<i32>,
    ) -> crate::elem_restriction::ElemRestriction {
        crate::elem_restriction::ElemRestriction::create(
            self, nelem, elemsize, ncomp, compstride, lsize, mtype, offsets,
        )
    }

    /// Returns a ElemRestriction
    ///
    /// # arguments
    ///
    /// * 'nelem'      - Number of elements described in the @a offsets array
    /// * 'elemsize'   - Size (number of "nodes") per element
    /// * 'ncomp'      - Number of field components per interpolation node (1
    ///                    for scalar fields)
    /// * 'compstride' - Stride between components for the same L-vector "node".
    ///                    Data for node i, component j, element k can be found
    ///                    in the L-vector at index
    ///                    offsets[i + k*elemsize] + j*compstride.
    /// * 'lsize'      - The size of the L-vector. This vector may be larger
    ///   than the elements and fields given by this restriction.
    /// * 'strides'   - Array for strides between [nodes, components, elements].
    ///                   Data for node i, component j, element k can be found
    ///                   in the L-vector at index
    ///                   i*strides\[0\] + j*strides\[1\] + k*strides\[2\].
    ///                   CEED_STRIDES_BACKEND may be used with vectors created
    ///                   by a Ceed backend.
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let nelem = 3;
    /// let strides : [i32; 3] = [1, 2, 2];
    /// let r = ceed.strided_elem_restriction(nelem, 2, 1, nelem*2, strides);
    /// ```
    pub fn strided_elem_restriction(
        &self,
        nelem: usize,
        elemsize: usize,
        ncomp: usize,
        lsize: usize,
        strides: [i32; 3],
    ) -> crate::elem_restriction::ElemRestriction {
        crate::elem_restriction::ElemRestriction::create_strided(
            self, nelem, elemsize, ncomp, lsize, strides,
        )
    }

    /// Returns a tensor-product basis
    ///
    /// # arguments
    ///
    /// * 'dim'       - Topological dimension of element
    /// * 'ncomp'     - Number of field components (1 for scalar fields)
    /// * 'P1d'       - Number of Gauss-Lobatto nodes in one dimension.  The
    ///                   polynomial degree of the resulting Q_k element is
    ///                   k=P-1.
    /// * 'Q1d'       - Number of quadrature points in one dimension
    /// * 'interp1d'  - Row-major (Q1d * P1d) matrix expressing the values of
    ///                   nodal basis functions at quadrature points
    /// * 'grad1d'    - Row-major (Q1d * P1d) matrix expressing derivatives of
    ///                   nodal basis functions at quadrature points
    /// * 'qref1d'    - Array of length Q1d holding the locations of quadrature
    ///                   points on the 1D reference element [-1, 1]
    /// * 'qweight1d' - Array of length Q1d holding the quadrature weights on
    ///                   the reference element
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let interp1d  = vec![ 0.62994317,  0.47255875, -0.14950343,  0.04700152,
    ///                      -0.07069480,  0.97297619,  0.13253993, -0.03482132,
    ///                      -0.03482132,  0.13253993,  0.97297619, -0.07069480,
    ///                       0.04700152, -0.14950343,  0.47255875,  0.62994317];
    /// let grad1d    = vec![-2.34183742,  2.78794489, -0.63510411,  0.18899664,
    ///                      -0.51670214, -0.48795249,  1.33790510, -0.33325047,
    //                        0.33325047, -1.33790510,  0.48795249,  0.51670214,
    ///                      -0.18899664,  0.63510411, -2.78794489,
    /// 2.34183742]; let qref1d    = vec![-0.86113631, -0.33998104,
    /// 0.33998104,  0.86113631]; let qweight1d = vec![ 0.34785485,
    /// 0.65214515,  0.65214515,  0.34785485]; let b = ceed.
    /// basis_tensor_H1(2, 1, 4, 4, &interp1d, &grad1d, &qref1d, &qweight1d);
    /// ```
    pub fn basis_tensor_H1(
        &self,
        dim: usize,
        ncomp: usize,
        P1d: usize,
        Q1d: usize,
        interp1d: &Vec<f64>,
        grad1d: &Vec<f64>,
        qref1d: &Vec<f64>,
        qweight1d: &Vec<f64>,
    ) -> crate::basis::Basis {
        crate::basis::Basis::create_tensor_H1(
            self, dim, ncomp, P1d, Q1d, interp1d, grad1d, qref1d, qweight1d,
        )
    }

    /// Returns a tensor-product Lagrange basis
    ///
    /// # arguments
    ///
    /// * 'dim'   - Topological dimension of element
    /// * 'ncomp' - Number of field components (1 for scalar fields)
    /// * 'P'     - Number of Gauss-Lobatto nodes in one dimension.  The
    ///               polynomial degree of the resulting Q_k element is k=P-1.
    /// * 'Q'     - Number of quadrature points in one dimension
    /// * 'qmode' - Distribution of the Q quadrature points (affects order of
    ///               accuracy for the quadrature)
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let b = ceed.basis_tensor_H1_Lagrange(2, 1, 3, 4, ceed::QuadMode::Gauss);
    /// ```
    pub fn basis_tensor_H1_Lagrange(
        &self,
        dim: usize,
        ncomp: usize,
        P: usize,
        Q: usize,
        qmode: QuadMode,
    ) -> crate::basis::Basis {
        crate::basis::Basis::create_tensor_H1_Lagrange(self, dim, ncomp, P, Q, qmode)
    }

    /// Returns a tensor-product basis
    ///
    /// # arguments
    ///
    /// * 'topo'    - Topology of element, e.g. hypercube, simplex, ect
    /// * 'ncomp'   - Number of field components (1 for scalar fields)
    /// * 'nnodes'  - Total number of nodes
    /// * 'nqpts'   - Total number of quadrature points
    /// * 'interp'  - Row-major (nqpts * nnodes) matrix expressing the values of
    ///                 nodal basis functions at quadrature points
    /// * 'grad'    - Row-major (nqpts * dim * nnodes) matrix expressing
    ///                 derivatives of nodal basis functions at quadrature points
    /// * 'qref'    - Array of length nqpts holding the locations of quadrature
    ///                 points on the reference element [-1, 1]
    /// * 'qweight' - Array of length nqpts holding the quadrature weights on
    ///                 the reference element
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let interp  = vec![ 0.12000000,  0.48000000, -0.12000000,  0.48000000,  0.16000000, -0.12000000,
    ///                    -0.12000000,  0.48000000,  0.12000000,  0.16000000,  0.48000000, -0.12000000,
    ///                    -0.11111111,  0.44444444, -0.11111111,  0.44444444,  0.44444444, -0.11111111,
    ///                    -0.12000000,  0.16000000, -0.12000000,  0.48000000,  0.48000000,  0.12000000];
    /// let grad    = vec![-1.40000000,  1.60000000, -0.20000000, -0.80000000,  0.80000000,  0.00000000,
    ///                     0.20000000, -1.60000000,  1.40000000, -0.80000000,  0.80000000,  0.00000000,
    ///                    -0.33333333,  0.00000000,  0.33333333, -1.33333333,  1.33333333,  0.00000000,
    ///                     0.20000000,  0.00000000, -0.20000000, -2.40000000,  2.40000000,  0.00000000,
    ///                    -1.40000000, -0.80000000,  0.00000000,  1.60000000,  0.80000000, -0.20000000,
    /// 	                0.20000000, -2.40000000,  0.00000000,  0.00000000,  2.40000000, -0.20000000,
    ///                    -0.33333333, -1.33333333,  0.00000000,  0.00000000,  1.33333333,  0.33333333,
    ///                     0.20000000, -0.80000000,  0.00000000, -1.60000000,  0.80000000,  1.40000000];
    /// let qref    = vec![ 0.20000000,  0.60000000,  0.33333333,  0.20000000,  0.20000000,  0.20000000,  0.33333333,  0.60000000];
    /// let qweight = vec![ 0.26041667,  0.26041667, -0.28125000,  0.26041667];
    /// let b = ceed.basis_H1(ceed::ElemTopology::Triangle, 1, 6, 4, &interp, &grad, &qref, &qweight);
    /// ```
    pub fn basis_H1(
        &self,
        topo: ElemTopology,
        ncomp: usize,
        nnodes: usize,
        nqpts: usize,
        interp: &Vec<f64>,
        grad: &Vec<f64>,
        qref: &Vec<f64>,
        qweight: &Vec<f64>,
    ) -> crate::basis::Basis {
        crate::basis::Basis::create_H1(
            self, topo, ncomp, nnodes, nqpts, interp, grad, qref, qweight,
        )
    }

    /// Returns a CeedQFunction for evaluating interior (volumetric) terms
    ///
    /// # arguments
    ///
    /// * 'vlength' - Vector length. Caller must ensure that number of
    ///                 quadrature points is a multiple of vlength.
    /// * 'f'       - Boxed closure to evaluate action at quadrature points.
    /// * 'source'  - Absolute path to source of QFunction,
    ///                 "\abs_path\file.h:function_name".
    ///                 For support across all backends, this source must only
    ///                 contain constructs supported by C99, C++11, and CUDA.
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let mut user_f = |
    ///   q: usize,
    ///   inputs: &Vec<&[f64]>,
    ///   outputs: &mut Vec<&mut [f64]>,
    /// | -> i32
    /// {
    ///   let u = &inputs[0];
    ///   let weights = &inputs[1];
    ///
    ///   let v = &mut outputs[0];
    ///
    ///   for i in 0..q {
    ///       v[i] = u[i] * weights[i];
    ///   }
    ///
    ///   return 0
    /// };
    ///
    /// let qf = ceed.q_function_interior(1, Box::new(user_f));
    /// ```
    pub fn q_function_interior(
        &self,
        vlength: i32,
        f: Box<crate::qfunction::QFunctionUserClosure>,
    ) -> crate::qfunction::QFunction {
        crate::qfunction::QFunction::create(self, vlength, f)
    }

    /// Returns a CeedQFunction for evaluating interior (volumetric) terms
    /// created by name
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let qf = ceed.q_function_interior_by_name("Mass1DBuild".to_string());
    /// ```
    pub fn q_function_interior_by_name(&self, name: String) -> crate::qfunction::QFunctionByName {
        crate::qfunction::QFunctionByName::create(self, name)
    }

    /// Returns a Operator and associate a QFunction. A Basis and
    /// ElemRestriction can be   associated with QFunction fields with
    /// set_field().
    ///
    /// * 'qf'   - QFunction defining the action of the operator at quadrature
    ///              points
    /// * 'dqf'  - QFunction defining the action of the Jacobian of the qf (or
    ///              qfunction_none)
    /// * 'dqfT' - QFunction defining teh action of the transpose of the
    ///              Jacobian of the qf (or qfunction_none)
    ///
    /// ```
    /// # use ceed::prelude::*;
    /// # let ceed = ceed::Ceed::default_init();
    /// let qf = ceed.q_function_interior_by_name("Mass1DBuild".to_string());
    /// let op = ceed.operator(&qf, QFunctionOpt::None, QFunctionOpt::None);
    /// ```
    pub fn operator<'b>(
        &self,
        qf: impl Into<crate::qfunction::QFunctionOpt<'b>>,
        dqf: impl Into<crate::qfunction::QFunctionOpt<'b>>,
        dqfT: impl Into<crate::qfunction::QFunctionOpt<'b>>,
    ) -> crate::operator::Operator {
        crate::operator::Operator::create(self, qf, dqf, dqfT)
    }

    /// Returns an Operator that composes the action of several Operators
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let op = ceed.composite_operator();
    /// ```
    pub fn composite_operator(&self) -> crate::operator::CompositeOperator {
        crate::operator::CompositeOperator::create(self)
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceed_t501() {
        let ceed = Ceed::default_init();
        let nelem = 4;
        let p = 3;
        let q = 4;
        let ndofs = p * nelem - nelem + 1;

        // Vectors
        let x = ceed.vector_from_slice(&[-1., -0.5, 0.0, 0.5, 1.0]);
        let mut qdata = ceed.vector(nelem * q);
        qdata.set_value(0.0);
        let mut u = ceed.vector(ndofs);
        u.set_value(1.0);
        let mut v = ceed.vector(ndofs);
        v.set_value(0.0);

        // Restrictions
        let mut indx: Vec<i32> = vec![0; 2 * nelem];
        for i in 0..nelem {
            indx[2 * i + 0] = i as i32;
            indx[2 * i + 1] = (i + 1) as i32;
        }
        let rx = ceed.elem_restriction(nelem, 2, 1, 1, nelem + 1, MemType::Host, &indx);
        let mut indu: Vec<i32> = vec![0; p * nelem];
        for i in 0..nelem {
            indu[p * i + 0] = i as i32;
            indu[p * i + 1] = (i + 1) as i32;
            indu[p * i + 2] = (i + 2) as i32;
        }
        let ru = ceed.elem_restriction(nelem, 3, 1, 1, ndofs, MemType::Host, &indu);
        let strides: [i32; 3] = [1, q as i32, q as i32];
        let rq = ceed.strided_elem_restriction(nelem, q, 1, q * nelem, strides);

        // Bases
        let bx = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, QuadMode::Gauss);
        let bu = ceed.basis_tensor_H1_Lagrange(1, 1, p, q, QuadMode::Gauss);

        // Set up operator
        let qf_build = ceed.q_function_interior_by_name("Mass1DBuild".to_string());
        let mut op_build = ceed.operator(&qf_build, QFunctionOpt::None, QFunctionOpt::None);
        op_build.set_field("dx", &rx, &bx, VectorOpt::Active);
        op_build.set_field("weights", ElemRestrictionOpt::None, &bx, VectorOpt::None);
        op_build.set_field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active);

        op_build.apply(&x, &mut qdata);

        // Mass operator
        let qf_mass = ceed.q_function_interior_by_name("MassApply".to_string());
        let mut op_mass = ceed.operator(&qf_mass, QFunctionOpt::None, QFunctionOpt::None);
        op_mass.set_field("u", &ru, &bu, VectorOpt::Active);
        op_mass.set_field("qdata", &rq, BasisOpt::Collocated, &qdata);
        op_mass.set_field("v", &ru, &bu, VectorOpt::Active);

        v.set_value(0.0);
        op_mass.apply(&u, &mut v);

        // Check
        let array = v.view();
        let mut sum = 0.0;
        for i in 0..ndofs {
            sum += array[i];
        }
        assert!(
            (sum - 2.0).abs() < 1e-15,
            "Incorrect interval length computed"
        );
    }
}

// -----------------------------------------------------------------------------
