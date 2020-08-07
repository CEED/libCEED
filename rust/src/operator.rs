use crate::prelude::*;

/// CeedOperator context wrapper
pub struct Operator<'a> {
    ceed: &'a crate::Ceed,
    ptr: bind_ceed::CeedOperator,
}

impl<'a> Operator<'a> {
    /// Constructor
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

    pub fn apply(
        &self,
        input: &crate::vector::Vector,
        output: &mut crate::vector::Vector,
    ) {
        unsafe {
            bind_ceed::CeedOperatorApply(
                self.ptr,
                input.ptr,
                output.ptr,
                bind_ceed::CEED_REQUEST_IMMEDIATE,
            )
        };
    }

    pub fn apply_add(
        &self,
        input: &crate::vector::Vector,
        output: &mut crate::vector::Vector,
    ) {
        unsafe {
            bind_ceed::CeedOperatorApplyAdd(
                self.ptr,
                input.ptr,
                output.ptr,
                bind_ceed::CEED_REQUEST_IMMEDIATE,
            )
        };
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

    pub fn create_multigrid_level(
        &self,
        p_mult_fine : &crate::vector::Vector,
        rstr_coarse : &crate::elem_restriction::ElemRestriction,
        basis_coarse: &crate::basis::Basis,
        op_coarse   : &mut crate::operator::Operator,
        op_prolong  : &mut crate::operator::Operator,
        op_restrict : &mut crate::operator::Operator,
    ) {
        unsafe {
            bind_ceed::CeedOperatorMultigridLevelCreate(
                self.ptr,
                p_mult_fine.ptr,
                rstr_coarse.ptr,
                basis_coarse.ptr,
                &mut op_coarse.ptr,
                &mut op_prolong.ptr,
                &mut op_restrict.ptr,
            )
        };
    }

    pub fn create_multigrid_level_tensor_H1(
        &self,
        p_mult_fine : &crate::vector::Vector,
        rstr_coarse : &crate::elem_restriction::ElemRestriction,
        basis_coarse: &crate::basis::Basis,
        interpCtoF  : &Vec<f64>,
        op_coarse   : &mut crate::operator::Operator,
        op_prolong  : &mut crate::operator::Operator,
        op_restrict : &mut crate::operator::Operator,
    ) {
        unsafe {
            bind_ceed::CeedOperatorMultigridLevelCreateTensorH1(
                self.ptr,
                p_mult_fine.ptr,
                rstr_coarse.ptr,
                basis_coarse.ptr,
                interpCtoF.as_ptr(),
                &mut op_coarse.ptr,
                &mut op_prolong.ptr,
                &mut op_restrict.ptr,
            )
        };
    }

    pub fn create_multigrid_level_H1(
        &self,
        p_mult_fine : &crate::vector::Vector,
        rstr_coarse : &crate::elem_restriction::ElemRestriction,
        basis_coarse: &crate::basis::Basis,
        interpCtoF  : &Vec<f64>,
        op_coarse   : &mut crate::operator::Operator,
        op_prolong  : &mut crate::operator::Operator,
        op_restrict : &mut crate::operator::Operator,
    ) {
        unsafe {
            bind_ceed::CeedOperatorMultigridLevelCreateH1(
                self.ptr,
                p_mult_fine.ptr,
                rstr_coarse.ptr,
                basis_coarse.ptr,
                interpCtoF.as_ptr(),
                &mut op_coarse.ptr,
                &mut op_prolong.ptr,
                &mut op_restrict.ptr,
            )
        };
    }

    pub fn create_FDME_element_inverse(
        &self,
        fdminv : &mut crate::operator::Operator,
    ) {
        unsafe {
            bind_ceed::CeedOperatorCreateFDMElementInverse(
                self.ptr,
                &mut fdminv.ptr,
                bind_ceed::CEED_REQUEST_IMMEDIATE,
            )
        };
    }
}

/// Destructor
impl<'a> Drop for Operator<'a> {
    fn drop(&mut self) {
        unsafe {
            bind_ceed::CeedOperatorDestroy(&mut self.ptr);
        }
    }
}
