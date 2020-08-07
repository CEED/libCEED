use crate::prelude::*;

pub struct Operator<'a> {
    ceed: &'a crate::Ceed,
    ptr: bind_ceed::CeedOperator,
}
impl<'a> Operator<'a> {
    pub fn create(
        ceed: &'a crate::Ceed,
        qf: &crate::qfunction::QFunction,
        dqf: &crate::qfunction::QFunction,
        dqfT: &crate::qfunction::QFunction,
    ) -> Self {
        let mut ptr = std::ptr::null_mut();
        unsafe { bind_ceed::CeedOperatorCreate(ceed.ptr, qf.ptr, dqf.ptr, dqfT.ptr, &mut ptr) };
        Self { ceed, ptr }
    }

    pub fn create_composite(
        ceed: &'a crate::Ceed,
    ) -> Self {
        let mut ptr = std::ptr::null_mut();
        unsafe { bind_ceed::CeedCompositeOperatorCreate(ceed.ptr, &mut ptr) };
        Self { ceed, ptr }
    }

    pub fn set_field(
        &mut self,
        fieldname: &str,
        r: &crate::elem_restriction::ElemRestriction,
        b: &crate::basis::Basis,
        v: &crate::vector::Vector,
    ) {
        unsafe {
            use std::ffi::CString;
            bind_ceed::CeedOperatorSetField(
                self.ptr,
                CString::new(fieldname).expect("CString::new failed").as_ptr() as *const i8,
                r.ptr,
                b.ptr,
                v.ptr,
            )
        };
    }

    pub fn add_sub_operator(
        &self,
        subop: &Operator,
    ) {
        unsafe {
            bind_ceed::CeedCompositeOperatorAddSub(self.ptr, subop.ptr)
        };
    }

    pub fn linear_assemble_qfunction(
        &self,
        assembled: &mut crate::vector::Vector,
        rstr: &mut crate::elem_restriction::ElemRestriction,
    ) {
        unsafe {
            bind_ceed::CeedOperatorLinearAssembleQFunction(
                self.ptr,
                &mut assembled.ptr,
                &mut rstr.ptr,
                bind_ceed::CEED_REQUEST_IMMEDIATE,
            )
        };
    }

    pub fn linear_asssemble_diagonal(
        &self,
        assembled: &mut crate::vector::Vector,
    ) {
        unsafe {
            bind_ceed::CeedOperatorLinearAssembleDiagonal(
                self.ptr,
                assembled.ptr,
                bind_ceed::CEED_REQUEST_IMMEDIATE,
            )
        };
    }

    pub fn linear_assemble_add_diagonal(
        &self,
        assembled: &mut crate::vector::Vector,
    ) {
        unsafe {
            bind_ceed::CeedOperatorLinearAssembleAddDiagonal(
                self.ptr,
                assembled.ptr,
                bind_ceed::CEED_REQUEST_IMMEDIATE,
            )
        };
    }

    pub fn linear_assemble_point_block_diagonal(
        &self,
        assembled: &mut crate::vector::Vector,
    ) {
        unsafe {
            bind_ceed::CeedOperatorLinearAssemblePointBlockDiagonal(
                self.ptr,
                assembled.ptr,
                bind_ceed::CEED_REQUEST_IMMEDIATE,
            )
        };
    }

    pub fn linear_assemble_add_point_block_diagonal(
        &self,
        assembled: &mut crate::vector::Vector,
    ) {
        unsafe {
            bind_ceed::CeedOperatorLinearAssembleAddPointBlockDiagonal(
                self.ptr,
                assembled.ptr,
                bind_ceed::CEED_REQUEST_IMMEDIATE,
            )
        };
    }
}
