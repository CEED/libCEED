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
use std::ffi::CString;
use std::fmt;

// -----------------------------------------------------------------------------
// CeedOperator context wrapper
// -----------------------------------------------------------------------------
pub struct OperatorCore<'a> {
    ceed: &'a crate::Ceed,
    ptr: bind_ceed::CeedOperator,
}

pub struct Operator<'a> {
    op_core: OperatorCore<'a>,
}

pub struct CompositeOperator<'a> {
    op_core: OperatorCore<'a>,
}

// -----------------------------------------------------------------------------
// Destructor
// -----------------------------------------------------------------------------
impl<'a> Drop for OperatorCore<'a> {
    fn drop(&mut self) {
        unsafe {
            bind_ceed::CeedOperatorDestroy(&mut self.ptr);
        }
    }
}

// -----------------------------------------------------------------------------
// Display
// -----------------------------------------------------------------------------
impl<'a> fmt::Display for OperatorCore<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut ptr = std::ptr::null_mut();
        let mut sizeloc = 202020;
        unsafe {
            let file = bind_ceed::open_memstream(&mut ptr, &mut sizeloc);
            bind_ceed::CeedOperatorView(self.ptr, file);
            bind_ceed::fclose(file);
            let cstring = CString::from_raw(ptr);
            let s = cstring.to_string_lossy().into_owned();
            write!(f, "{}", s)
        }
    }
}

impl<'a> fmt::Display for Operator<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.op_core.fmt(f)
    }
}

impl<'a> fmt::Display for CompositeOperator<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.op_core.fmt(f)
    }
}

// -----------------------------------------------------------------------------
// Core functionality
// -----------------------------------------------------------------------------
impl<'a> OperatorCore<'a> {
    // Constructor
    pub fn new(ceed: &'a crate::Ceed, ptr: bind_ceed::CeedOperator) -> Self {
        Self { ceed, ptr }
    }

    // Common implementations
    pub fn apply(&self, input: &crate::vector::Vector, output: &mut crate::vector::Vector) {
        unsafe {
            bind_ceed::CeedOperatorApply(
                self.ptr,
                input.ptr,
                output.ptr,
                bind_ceed::CEED_REQUEST_IMMEDIATE,
            )
        };
    }

    pub fn apply_add(&self, input: &crate::vector::Vector, output: &mut crate::vector::Vector) {
        unsafe {
            bind_ceed::CeedOperatorApplyAdd(
                self.ptr,
                input.ptr,
                output.ptr,
                bind_ceed::CEED_REQUEST_IMMEDIATE,
            )
        };
    }

    pub fn linear_asssemble_diagonal(&self, assembled: &mut crate::vector::Vector) {
        unsafe {
            bind_ceed::CeedOperatorLinearAssembleDiagonal(
                self.ptr,
                assembled.ptr,
                bind_ceed::CEED_REQUEST_IMMEDIATE,
            )
        };
    }

    pub fn linear_assemble_add_diagonal(&self, assembled: &mut crate::vector::Vector) {
        unsafe {
            bind_ceed::CeedOperatorLinearAssembleAddDiagonal(
                self.ptr,
                assembled.ptr,
                bind_ceed::CEED_REQUEST_IMMEDIATE,
            )
        };
    }

    pub fn linear_assemble_point_block_diagonal(&self, assembled: &mut crate::vector::Vector) {
        unsafe {
            bind_ceed::CeedOperatorLinearAssemblePointBlockDiagonal(
                self.ptr,
                assembled.ptr,
                bind_ceed::CEED_REQUEST_IMMEDIATE,
            )
        };
    }

    pub fn linear_assemble_add_point_block_diagonal(&self, assembled: &mut crate::vector::Vector) {
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
        p_mult_fine: &crate::vector::Vector,
        rstr_coarse: &crate::elem_restriction::ElemRestriction,
        basis_coarse: &crate::basis::Basis,
        op_coarse: &mut crate::operator::Operator,
        op_prolong: &mut crate::operator::Operator,
        op_restrict: &mut crate::operator::Operator,
    ) {
        unsafe {
            bind_ceed::CeedOperatorMultigridLevelCreate(
                self.ptr,
                p_mult_fine.ptr,
                rstr_coarse.ptr,
                basis_coarse.ptr,
                &mut op_coarse.op_core.ptr,
                &mut op_prolong.op_core.ptr,
                &mut op_restrict.op_core.ptr,
            )
        };
    }

    pub fn create_multigrid_level_tensor_H1(
        &self,
        p_mult_fine: &crate::vector::Vector,
        rstr_coarse: &crate::elem_restriction::ElemRestriction,
        basis_coarse: &crate::basis::Basis,
        interpCtoF: &Vec<f64>,
        op_coarse: &mut crate::operator::Operator,
        op_prolong: &mut crate::operator::Operator,
        op_restrict: &mut crate::operator::Operator,
    ) {
        unsafe {
            bind_ceed::CeedOperatorMultigridLevelCreateTensorH1(
                self.ptr,
                p_mult_fine.ptr,
                rstr_coarse.ptr,
                basis_coarse.ptr,
                interpCtoF.as_ptr(),
                &mut op_coarse.op_core.ptr,
                &mut op_prolong.op_core.ptr,
                &mut op_restrict.op_core.ptr,
            )
        };
    }

    pub fn create_multigrid_level_H1(
        &self,
        p_mult_fine: &crate::vector::Vector,
        rstr_coarse: &crate::elem_restriction::ElemRestriction,
        basis_coarse: &crate::basis::Basis,
        interpCtoF: &Vec<f64>,
        op_coarse: &mut crate::operator::Operator,
        op_prolong: &mut crate::operator::Operator,
        op_restrict: &mut crate::operator::Operator,
    ) {
        unsafe {
            bind_ceed::CeedOperatorMultigridLevelCreateH1(
                self.ptr,
                p_mult_fine.ptr,
                rstr_coarse.ptr,
                basis_coarse.ptr,
                interpCtoF.as_ptr(),
                &mut op_coarse.op_core.ptr,
                &mut op_prolong.op_core.ptr,
                &mut op_restrict.op_core.ptr,
            )
        };
    }
}

// -----------------------------------------------------------------------------
// Operator
// -----------------------------------------------------------------------------
impl<'a> Operator<'a> {
    // Constructor
    pub fn create(
        ceed: &'a crate::Ceed,
        qf: &crate::qfunction::QFunction,
        dqf: &crate::qfunction::QFunction,
        dqfT: &crate::qfunction::QFunction,
    ) -> Self {
        let mut ptr = std::ptr::null_mut();
        unsafe {
            bind_ceed::CeedOperatorCreate(
                ceed.ptr,
                qf.qf_core.ptr,
                dqf.qf_core.ptr,
                dqfT.qf_core.ptr,
                &mut ptr,
            )
        };
        let op_core = OperatorCore::new(ceed, ptr);
        Self { op_core }
    }

    pub fn apply(&self, input: &crate::vector::Vector, output: &mut crate::vector::Vector) {
        self.op_core.apply(input, output)
    }

    pub fn apply_add(&self, input: &crate::vector::Vector, output: &mut crate::vector::Vector) {
        self.op_core.apply_add(input, output)
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
                self.op_core.ptr,
                CString::new(fieldname)
                    .expect("CString::new failed")
                    .as_ptr() as *const i8,
                r.ptr,
                b.ptr,
                v.ptr,
            )
        };
    }

    pub fn linear_assemble_qfunction(
        &self,
        assembled: &mut crate::vector::Vector,
        rstr: &mut crate::elem_restriction::ElemRestriction,
    ) {
        unsafe {
            bind_ceed::CeedOperatorLinearAssembleQFunction(
                self.op_core.ptr,
                &mut assembled.ptr,
                &mut rstr.ptr,
                bind_ceed::CEED_REQUEST_IMMEDIATE,
            )
        };
    }

    pub fn linear_asssemble_diagonal(&self, assembled: &mut crate::vector::Vector) {
        self.op_core.linear_assemble_add_diagonal(assembled)
    }

    pub fn linear_assemble_add_diagonal(&self, assembled: &mut crate::vector::Vector) {
        self.op_core.linear_assemble_add_diagonal(assembled)
    }

    pub fn linear_assemble_point_block_diagonal(&self, assembled: &mut crate::vector::Vector) {
        self.op_core.linear_assemble_add_diagonal(assembled)
    }

    pub fn linear_assemble_add_point_block_diagonal(&self, assembled: &mut crate::vector::Vector) {
        self.op_core.linear_assemble_add_diagonal(assembled)
    }

    pub fn create_multigrid_level(
        &self,
        p_mult_fine: &crate::vector::Vector,
        rstr_coarse: &crate::elem_restriction::ElemRestriction,
        basis_coarse: &crate::basis::Basis,
        op_coarse: &mut crate::operator::Operator,
        op_prolong: &mut crate::operator::Operator,
        op_restrict: &mut crate::operator::Operator,
    ) {
        self.op_core.create_multigrid_level(
            p_mult_fine,
            rstr_coarse,
            basis_coarse,
            op_coarse,
            op_prolong,
            op_restrict,
        )
    }

    pub fn create_multigrid_level_tensor_H1(
        &self,
        p_mult_fine: &crate::vector::Vector,
        rstr_coarse: &crate::elem_restriction::ElemRestriction,
        basis_coarse: &crate::basis::Basis,
        interpCtoF: &Vec<f64>,
        op_coarse: &mut crate::operator::Operator,
        op_prolong: &mut crate::operator::Operator,
        op_restrict: &mut crate::operator::Operator,
    ) {
        self.op_core.create_multigrid_level_tensor_H1(
            p_mult_fine,
            rstr_coarse,
            basis_coarse,
            interpCtoF,
            op_coarse,
            op_prolong,
            op_restrict,
        )
    }

    pub fn create_multigrid_level_H1(
        &self,
        p_mult_fine: &crate::vector::Vector,
        rstr_coarse: &crate::elem_restriction::ElemRestriction,
        basis_coarse: &crate::basis::Basis,
        interpCtoF: &Vec<f64>,
        op_coarse: &mut crate::operator::Operator,
        op_prolong: &mut crate::operator::Operator,
        op_restrict: &mut crate::operator::Operator,
    ) {
        self.op_core.create_multigrid_level_H1(
            p_mult_fine,
            rstr_coarse,
            basis_coarse,
            interpCtoF,
            op_coarse,
            op_prolong,
            op_restrict,
        )
    }

    pub fn create_FDME_element_inverse(&self, fdminv: &mut crate::operator::Operator) {
        unsafe {
            bind_ceed::CeedOperatorCreateFDMElementInverse(
                self.op_core.ptr,
                &mut fdminv.op_core.ptr,
                bind_ceed::CEED_REQUEST_IMMEDIATE,
            )
        };
    }
}

// -----------------------------------------------------------------------------
// Composite Operator
// -----------------------------------------------------------------------------
impl<'a> CompositeOperator<'a> {
    // Constructor
    pub fn create(ceed: &'a crate::Ceed) -> Self {
        let mut ptr = std::ptr::null_mut();
        unsafe { bind_ceed::CeedCompositeOperatorCreate(ceed.ptr, &mut ptr) };
        let op_core = OperatorCore::new(ceed, ptr);
        Self { op_core }
    }

    pub fn apply(&self, input: &crate::vector::Vector, output: &mut crate::vector::Vector) {
        self.op_core.apply(input, output)
    }

    pub fn apply_add(&self, input: &crate::vector::Vector, output: &mut crate::vector::Vector) {
        self.op_core.apply_add(input, output)
    }

    pub fn add_sub_operator(&self, subop: &Operator) {
        unsafe { bind_ceed::CeedCompositeOperatorAddSub(self.op_core.ptr, subop.op_core.ptr) };
    }

    pub fn linear_asssemble_diagonal(&self, assembled: &mut crate::vector::Vector) {
        self.op_core.linear_assemble_add_diagonal(assembled)
    }

    pub fn linear_assemble_add_diagonal(&self, assembled: &mut crate::vector::Vector) {
        self.op_core.linear_assemble_add_diagonal(assembled)
    }

    pub fn linear_assemble_point_block_diagonal(&self, assembled: &mut crate::vector::Vector) {
        self.op_core.linear_assemble_add_diagonal(assembled)
    }

    pub fn linear_assemble_add_point_block_diagonal(&self, assembled: &mut crate::vector::Vector) {
        self.op_core.linear_assemble_add_diagonal(assembled)
    }

    pub fn create_multigrid_level(
        &self,
        p_mult_fine: &crate::vector::Vector,
        rstr_coarse: &crate::elem_restriction::ElemRestriction,
        basis_coarse: &crate::basis::Basis,
        op_coarse: &mut crate::operator::Operator,
        op_prolong: &mut crate::operator::Operator,
        op_restrict: &mut crate::operator::Operator,
    ) {
        self.op_core.create_multigrid_level(
            p_mult_fine,
            rstr_coarse,
            basis_coarse,
            op_coarse,
            op_prolong,
            op_restrict,
        )
    }

    pub fn create_multigrid_level_tensor_H1(
        &self,
        p_mult_fine: &crate::vector::Vector,
        rstr_coarse: &crate::elem_restriction::ElemRestriction,
        basis_coarse: &crate::basis::Basis,
        interpCtoF: &Vec<f64>,
        op_coarse: &mut crate::operator::Operator,
        op_prolong: &mut crate::operator::Operator,
        op_restrict: &mut crate::operator::Operator,
    ) {
        self.op_core.create_multigrid_level_tensor_H1(
            p_mult_fine,
            rstr_coarse,
            basis_coarse,
            interpCtoF,
            op_coarse,
            op_prolong,
            op_restrict,
        )
    }

    pub fn create_multigrid_level_H1(
        &self,
        p_mult_fine: &crate::vector::Vector,
        rstr_coarse: &crate::elem_restriction::ElemRestriction,
        basis_coarse: &crate::basis::Basis,
        interpCtoF: &Vec<f64>,
        op_coarse: &mut crate::operator::Operator,
        op_prolong: &mut crate::operator::Operator,
        op_restrict: &mut crate::operator::Operator,
    ) {
        self.op_core.create_multigrid_level_H1(
            p_mult_fine,
            rstr_coarse,
            basis_coarse,
            interpCtoF,
            op_coarse,
            op_prolong,
            op_restrict,
        )
    }
}

// -----------------------------------------------------------------------------
