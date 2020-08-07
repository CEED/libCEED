#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::ffi::CString;
use std::fmt;
use std::mem;
// use std::io::{self, Write};
use crate::prelude::*;

mod prelude {
    pub mod bind_ceed {
        include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
    }
}

pub mod basis;
pub mod elem_restriction;
pub mod operator;
pub mod qfunction;
pub mod qfunction_context;
pub mod vector;

/// Ceed context wrapper
pub struct Ceed {
    backend: String,
    // Pointer to C object
    ptr: bind_ceed::Ceed,
}

/// Display
impl fmt::Display for Ceed {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.backend)
    }
}

/// Destructor
impl Drop for Ceed {
    fn drop(&mut self) {
        unsafe {
            bind_ceed::CeedDestroy(&mut self.ptr);
        }
    }
}

/// Enums for libCEED
#[derive(Clone, Copy, PartialEq)]
pub enum MemType {
    Host,
    Device,
}

#[derive(Clone, Copy, PartialEq)]
pub enum CopyMode {
    CopyValues,
    UsePointer,
    OwnPointer,
}

#[derive(Clone, Copy, PartialEq)]
pub enum NormType {
    One,
    Two,
    Max,
}

#[derive(Clone, Copy, PartialEq)]
pub enum TransposeMode {
    NoTranspose,
    Transpose,
}

#[derive(Clone, Copy, PartialEq)]
pub enum QuadMode {
    Gauss,
    GaussLobatto,
}

#[derive(Clone, Copy, PartialEq)]
pub enum ElemTopology {
    Line,
    Triangle,
    Quad,
    Tet,
    Pyramid,
    Prism,
    Hex,
}

#[derive(Clone, Copy, PartialEq)]
pub enum EvalMode {
    None,
    Interp,
    Grad,
    Div,
    Curl,
    Weight,
}

// Object constructors
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
        Ceed {
            backend: resource.to_string(),
            ptr,
        }
    }

    /// Returns a CeedVector of the specified length (does not allocate memory)
    ///
    /// # arguments
    ///
    /// * 'n' - Length of vector
    ///
    /// ```
    /// let ceed = ceed::Ceed::init("/cpu/self/ref/serial");
    /// let vec = ceed.vector(10);
    /// ```
    pub fn vector(&self, n: usize) -> crate::vector::Vector {
        crate::vector::Vector::create(self, n)
    }

    /// Elem Restriction
    pub fn elem_restriction(
        &self,
        nelem: i32,
        elemsize: i32,
        ncomp: i32,
        compstride: i32,
        lsize: i32,
        mtype: MemType,
        cmode: CopyMode,
        offsets: &Vec<i32>,
    ) -> crate::elem_restriction::ElemRestriction {
        crate::elem_restriction::ElemRestriction::create(
            self, nelem, elemsize, ncomp, compstride, lsize, mtype, cmode, offsets,
        )
    }

    /// Basis
    pub fn basis_tensor_H1(
        &self,
        dim: i32,
        ncomp: i32,
        P1d: i32,
        Q1d: i32,
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
    /// let ceed = ceed::Ceed::init("/cpu/self/ref/serial");
    /// let b = ceed.basis_tensor_H1_Lagrange(2, 1, 3, 4, ceed::QuadMode::Gauss);
    /// ```
    pub fn basis_tensor_H1_Lagrange(
        &self,
        dim: i32,
        ncomp: i32,
        P: i32,
        Q: i32,
        qmode: QuadMode,
    ) -> crate::basis::Basis {
        crate::basis::Basis::create_tensor_H1_Lagrange(self, dim, ncomp, P, Q, qmode)
    }

    pub fn basis_H1(
        &self,
        topo: ElemTopology,
        ncomp: i32,
        nnodes: i32,
        nqpts: i32,
        interp: &Vec<f64>,
        grad: &Vec<f64>,
        qref: &Vec<f64>,
        qweight: &Vec<f64>,
    ) -> crate::basis::Basis {
        crate::basis::Basis::create_H1(
            self, topo, ncomp, nnodes, nqpts, interp, grad, qref, qweight,
        )
    }

    /// QFunction
    pub fn q_function_interior(
        &self,
        vlength: i32,
        f: bind_ceed::CeedQFunctionUser,
        source: impl Into<String>,
    ) -> crate::qfunction::QFunction {
        //TODO
        todo!()
    }

    /// Returns a CeedQFunctionContext for storing CeedQFunction user context data
    ///
    /// ```
    /// let ceed = ceed::Ceed::init("/cpu/self/ref/serial");
    /// let ctx = ceed.q_function_context();
    /// ```
    pub fn q_function_context(&self) -> crate::qfunction_context::QFunctionContext {
        crate::qfunction_context::QFunctionContext::create(self)
    }

    /// Operator
    pub fn operator(
        &self,
        qf: &crate::qfunction::QFunction,
        dqf: &crate::qfunction::QFunction,
        dqfT: &crate::qfunction::QFunction,
    ) -> crate::operator::Operator {
        crate::operator::Operator::create(self, qf, dqf, dqfT)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceed_t000() {
        let ceed = Ceed::init("/cpu/self/ref/serial");
        println!("{}", ceed);
    }

    fn ceed_t001() {
        let ceed = Ceed::init("/cpu/self/ref/serial");
        let vec = ceed.vector(10);
    }
}
