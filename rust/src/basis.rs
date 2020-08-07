use crate::prelude::*;
use std::mem;

pub struct Basis<'a> {
    ceed: &'a crate::Ceed,
    ptr: bind_ceed::CeedBasis,
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
}
