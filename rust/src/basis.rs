use crate::prelude::*;
use std::mem;

pub struct Basis<'a> {
    ceed: &'a crate::Ceed,
    ptr: bind_ceed::CeedBasis,
}
pub enum QuadMode {
    Gauss,
    GaussLobatto,
}
pub enum ElemTopology {
    Line,
    Triangle,
    Quad,
    Tet,
    Pyramid,
    Prism,
    Hex,
}
pub enum TransposeMode {
    NoTranspose,
    Transpose,
}
pub enum EvalMode {
    None,
    Interp,
    Grad,
    Div,
    Curl,
    Weight,
}
impl<'a> Basis<'a> {
    pub fn create_tensor_H1(
        ceed: &'a crate::Ceed,
        dim: i32,
        ncomp: i32,
        P1d: i32,
        Q1d: i32,
        interp1d: &Vec<f64>,
        grad1d: &Vec<f64>,
        qref1d: &Vec<f64>,
        qweight1d: &Vec<f64>,
    ) -> Self {
        let mut ptr =
            unsafe { libc::malloc(mem::size_of::<bind_ceed::CeedBasis>()) as bind_ceed::CeedBasis };
        unsafe {
            bind_ceed::CeedBasisCreateTensorH1(
                ceed.ptr,
                dim,
                ncomp,
                P1d,
                Q1d,
                interp1d.as_ptr(),
                grad1d.as_ptr(),
                qref1d.as_ptr(),
                qweight1d.as_ptr(),
                &mut ptr,
            )
        };
        Self { ceed, ptr }
    }

    pub fn create_tensor_H1_Lagrange(
        ceed: &'a crate::Ceed,
        dim: i32,
        ncomp: i32,
        P: i32,
        Q: i32,
        qmode: QuadMode,
    ) -> Self {
        let mut ptr =
            unsafe { libc::malloc(mem::size_of::<bind_ceed::CeedBasis>()) as bind_ceed::CeedBasis };
        unsafe {
            bind_ceed::CeedBasisCreateTensorH1Lagrange(
                ceed.ptr,
                dim,
                ncomp,
                P,
                Q,
                qmode as bind_ceed::CeedQuadMode,
                &mut ptr,
            );
        }
        Self { ceed, ptr }
    }

    pub fn create_H1(
        ceed: &'a crate::Ceed,
        topo: ElemTopology,
        ncomp: i32,
        nnodes: i32,
        nqpts: i32,
        interp: &Vec<f64>,
        grad: &Vec<f64>,
        qref: &Vec<f64>,
        qweight: &Vec<f64>,
     ) -> Self {
        let mut ptr =
            unsafe { libc::malloc(mem::size_of::<bind_ceed::CeedBasis>()) as bind_ceed::CeedBasis };
        unsafe {
            bind_ceed::CeedBasisCreateH1(
                ceed.ptr,
                topo as bind_ceed::CeedElemTopology,
                ncomp,
                nnodes,
                nqpts,
                interp.as_ptr(),
                grad.as_ptr(),
                qref.as_ptr(),
                qweight.as_ptr(),
                &mut ptr)
        };
        Self  { ceed, ptr }
    }

    pub fn apply(
        &self,
        nelem: i32,
        tmode: TransposeMode,
        emode: EvalMode,
        u: &crate::vector::Vector,
        v: &mut crate::vector::Vector,
    ) {
        unsafe {
            bind_ceed::CeedBasisApply(
                self.ptr,
                nelem,
                tmode as bind_ceed::CeedTransposeMode,
                emode as bind_ceed::CeedEvalMode,
                u.ptr,
                v.ptr,
            ) 
        };
    }
}
