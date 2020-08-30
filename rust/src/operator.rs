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

// -----------------------------------------------------------------------------
// CeedOperator context wrapper
// -----------------------------------------------------------------------------
pub struct OperatorCore { 
    ptr: bind_ceed::CeedOperator,
}

pub struct Operator {
    op_core: OperatorCore,
}

pub struct CompositeOperator {
    op_core: OperatorCore,
}

// -----------------------------------------------------------------------------
// Destructor
// -----------------------------------------------------------------------------
impl Drop for OperatorCore {
    fn drop(&mut self) {
        unsafe {
            bind_ceed::CeedOperatorDestroy(&mut self.ptr);
        }
    }
}

// -----------------------------------------------------------------------------
// Display
// -----------------------------------------------------------------------------
impl fmt::Display for OperatorCore {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut ptr = std::ptr::null_mut();
        let mut sizeloc = crate::MAX_BUFFER_LENGTH;
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

/// View an Operator
///
/// ```
/// # use ceed::prelude::*;
/// # let ceed = ceed::Ceed::default_init();
/// let qf = ceed.q_function_interior_by_name("Mass1DBuild".to_string());
/// let mut op = ceed.operator(&qf, QFunctionOpt::None, QFunctionOpt::None);
///
/// // Operator field arguments
/// let ne = 3;
/// let q = 4 as usize;
/// let mut ind : Vec<i32> = vec![0; 2*ne];
/// for i in 0..ne {
///   ind[2*i+0] = i as i32;
///   ind[2*i+1] = (i+1) as i32;
/// }
/// let r = ceed.elem_restriction(ne, 2, 1, 1, ne+1, ceed::MemType::Host, &ind);
/// let strides : [i32; 3] = [1, q as i32, q as i32];
/// let rq = ceed.strided_elem_restriction(ne, 2, 1, q*ne, strides);
///
/// let b = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, ceed::QuadMode::Gauss);
///
/// // Operator fields
/// op.set_field("dx", &r, &b, VectorOpt::Active);
/// op.set_field("weights", ElemRestrictionOpt::None, &b, VectorOpt::None);
/// op.set_field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active);
///
/// println!("{}", op);
/// ```
impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.op_core.fmt(f)
    }
}

/// View a composite Operator
///
/// ```
/// # let ceed = ceed::Ceed::default_init();
/// let op = ceed.composite_operator();
/// println!("{}", op);
/// ```
impl fmt::Display for CompositeOperator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.op_core.fmt(f)
    }
}

// -----------------------------------------------------------------------------
// Core functionality
// -----------------------------------------------------------------------------
impl OperatorCore {
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
}

// -----------------------------------------------------------------------------
// Operator
// -----------------------------------------------------------------------------
impl Operator {
    // Constructor
    pub fn create<'b>(
        ceed: & crate::Ceed,
        qf: impl Into<crate::qfunction::QFunctionOpt<'b>>,
        dqf: impl Into<crate::qfunction::QFunctionOpt<'b>>,
        dqfT: impl Into<crate::qfunction::QFunctionOpt<'b>>,
    ) -> Self {
        let mut ptr = std::ptr::null_mut();
        unsafe { bind_ceed::CeedOperatorCreate(ceed.ptr, qf.into().to_raw(), dqf.into().to_raw(), dqfT.into().to_raw(), &mut ptr) };
        let op_core = OperatorCore { ptr };
        Self { op_core }
    }

    /// Apply Operator to a vector
    ///
    /// * 'input'  - Input Vector
    /// * 'output' - Output Vector
    ///
    /// ```
    /// # use ceed::prelude::*;
    /// # let ceed = ceed::Ceed::default_init();
    /// let ne = 4;
    /// let p = 3;
    /// let q = 4;
    /// let ndofs = p*ne-ne+1;
    ///
    /// // Vectors
    /// let x = ceed.vector_from_slice(&[-1., -0.5, 0.0, 0.5, 1.0]);
    /// let mut qdata = ceed.vector(ne*q);
    /// qdata.set_value(0.0);
    /// let mut u = ceed.vector(ndofs);
    /// u.set_value(1.0);
    /// let mut v = ceed.vector(ndofs);
    /// v.set_value(0.0);
    ///
    /// // Restrictions
    /// let mut indx : Vec<i32> = vec![0; 2*ne];
    /// for i in 0..ne {
    ///   indx[2*i+0] = i as i32;
    ///   indx[2*i+1] = (i+1) as i32;
    /// }
    /// let rx = ceed.elem_restriction(ne, 2, 1, 1, ne+1, ceed::MemType::Host, &indx);
    /// let mut indu : Vec<i32> = vec![0; p*ne];
    /// for i in 0..ne {
    ///   indu[p*i+0] = i as i32;
    ///   indu[p*i+1] = (i+1) as i32;
    ///   indu[p*i+2] = (i+2) as i32;
    /// }
    /// let ru = ceed.elem_restriction(ne, 3, 1, 1, ndofs, ceed::MemType::Host, &indu);
    /// let strides : [i32; 3] = [1, q as i32, q as i32];
    /// let rq = ceed.strided_elem_restriction(ne, q, 1, q*ne, strides);
    ///
    /// // Bases
    /// let bx = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, ceed::QuadMode::Gauss);
    /// let bu = ceed.basis_tensor_H1_Lagrange(1, 1, p, q, ceed::QuadMode::Gauss);
    ///
    /// // Set up operator
    /// let qf_build = ceed.q_function_interior_by_name("Mass1DBuild".to_string());
    /// let mut op_build = ceed.operator(&qf_build, QFunctionOpt::None, QFunctionOpt::None);
    /// op_build.set_field("dx", &rx, &bx, VectorOpt::Active);
    /// op_build.set_field("weights", ElemRestrictionOpt::None, &bx, VectorOpt::None);
    /// op_build.set_field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active);
    ///
    /// op_build.apply(&x, &mut qdata);
    ///
    /// // Mass operator
    /// let qf_mass = ceed.q_function_interior_by_name("MassApply".to_string());
    /// let mut op_mass = ceed.operator(&qf_mass, QFunctionOpt::None, QFunctionOpt::None);
    /// op_mass.set_field("u", &ru, &bu, VectorOpt::Active);
    /// op_mass.set_field("qdata", &rq, BasisOpt::Collocated, &qdata);
    /// op_mass.set_field("v", &ru, &bu, VectorOpt::Active);
    ///
    /// v.set_value(0.0);
    /// op_mass.apply(&u, &mut v);
    ///
    /// // Check
    /// let array = v.view();
    /// let mut sum = 0.0;
    /// for i in 0..ndofs {
    ///   sum += array[i];
    /// }
    /// assert!((sum - 2.0).abs() < 1e-15, "Incorrect interval length computed");
    /// ```
    pub fn apply(&self, input: &crate::vector::Vector, output: &mut crate::vector::Vector) {
        self.op_core.apply(input, output)
    }

    /// Apply Operator to a vector and add result to output vector
    ///
    /// * 'input'  - Input Vector
    /// * 'output' - Output Vector
    ///
    /// ```
    /// # use ceed::prelude::*;
    /// # let ceed = ceed::Ceed::default_init();
    /// let ne = 4;
    /// let p = 3;
    /// let q = 4;
    /// let ndofs = p*ne-ne+1;
    ///
    /// // Vectors
    /// let x = ceed.vector_from_slice(&[-1., -0.5, 0.0, 0.5, 1.0]);
    /// let mut qdata = ceed.vector(ne*q);
    /// qdata.set_value(0.0);
    /// let mut u = ceed.vector(ndofs);
    /// u.set_value(1.0);
    /// let mut v = ceed.vector(ndofs);
    ///
    /// // Restrictions
    /// let mut indx : Vec<i32> = vec![0; 2*ne];
    /// for i in 0..ne {
    ///   indx[2*i+0] = i as i32;
    ///   indx[2*i+1] = (i+1) as i32;
    /// }
    /// let rx = ceed.elem_restriction(ne, 2, 1, 1, ne+1, ceed::MemType::Host, &indx);
    /// let mut indu : Vec<i32> = vec![0; p*ne];
    /// for i in 0..ne {
    ///   indu[p*i+0] = i as i32;
    ///   indu[p*i+1] = (i+1) as i32;
    ///   indu[p*i+2] = (i+2) as i32;
    /// }
    /// let ru = ceed.elem_restriction(ne, 3, 1, 1, ndofs, ceed::MemType::Host, &indu);
    /// let strides : [i32; 3] = [1, q as i32, q as i32];
    /// let rq = ceed.strided_elem_restriction(ne, q, 1, q*ne, strides);
    ///
    /// // Bases
    /// let bx = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, ceed::QuadMode::Gauss);
    /// let bu = ceed.basis_tensor_H1_Lagrange(1, 1, p, q, ceed::QuadMode::Gauss);
    ///
    /// // Set up operator
    /// let qf_build = ceed.q_function_interior_by_name("Mass1DBuild".to_string());
    /// let mut op_build = ceed.operator(&qf_build, QFunctionOpt::None, QFunctionOpt::None);
    /// op_build.set_field("dx", &rx, &bx, VectorOpt::Active);
    /// op_build.set_field("weights", ElemRestrictionOpt::None, &bx, VectorOpt::None);
    /// op_build.set_field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active);
    ///
    /// op_build.apply(&x, &mut qdata);
    ///
    /// // Mass operator
    /// let qf_mass = ceed.q_function_interior_by_name("MassApply".to_string());
    /// let mut op_mass = ceed.operator(&qf_mass, QFunctionOpt::None, QFunctionOpt::None);
    /// op_mass.set_field("u", &ru, &bu, VectorOpt::Active);
    /// op_mass.set_field("qdata", &rq, BasisOpt::Collocated, &qdata);
    /// op_mass.set_field("v", &ru, &bu, VectorOpt::Active);
    ///
    /// v.set_value(1.0);
    /// op_mass.apply_add(&u, &mut v);
    ///
    /// // Check
    /// let array = v.view();
    /// let mut sum = 0.0;
    /// for i in 0..ndofs {
    ///   sum += array[i];
    /// }
    /// assert!((sum - (2.0 + ndofs as f64)).abs() < 1e-15, "Incorrect interval length computed and added");
    /// ```
    pub fn apply_add(&self, input: &crate::vector::Vector, output: &mut crate::vector::Vector) {
        self.op_core.apply_add(input, output)
    }

    /// Provide a field to a Operator for use by its QFunction
    ///
    /// * 'fieldname' - Name of the field (to be matched with the
    ///                   name used by the QFunction)
    /// * 'r'         - ElemRestriction
    /// * 'b'         - Basis in which the field resides or BasisCollocated
    ///                   if collocated with quadrature points
    /// * 'v'         - Vector to be used by Operator or VectorActive if
    ///                   field is active or VectorNone if using
    ///                   Weight is the QFunction
    ///
    ///
    /// ```
    /// # use ceed::prelude::*;
    /// # let ceed = ceed::Ceed::default_init();
    /// let qf = ceed.q_function_interior_by_name("Mass1DBuild".to_string());
    /// let mut op = ceed.operator(&qf, QFunctionOpt::None, QFunctionOpt::None);
    ///
    /// // Operator field arguments
    /// let ne = 3;
    /// let q = 4;
    /// let mut ind : Vec<i32> = vec![0; 2*ne];
    /// for i in 0..ne {
    ///   ind[2*i+0] = i as i32;
    ///   ind[2*i+1] = (i+1) as i32;
    /// }
    /// let r = ceed.elem_restriction(ne, 2, 1, 1, ne+1, ceed::MemType::Host, &ind);
    ///
    /// let b = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, ceed::QuadMode::Gauss);
    ///
    /// // Operator field
    /// op.set_field("dx", &r, &b, VectorOpt::Active);
    /// ```
    pub fn set_field<'b>(
        &mut self,
        fieldname: &str,
        r: impl Into<crate::elem_restriction::ElemRestrictionOpt<'b>>,
        b: impl Into<crate::basis::BasisOpt<'b>>,
        v: impl Into<crate::vector::VectorOpt<'b>>,
    ) {
        unsafe {
            bind_ceed::CeedOperatorSetField(
                self.op_core.ptr,
                CString::new(fieldname)
                    .expect("CString::new failed")
                    .as_ptr() as *const i8,
                r.into().to_raw(),
                b.into().to_raw(),
                v.into().to_raw(),
            )
        };
    }

    /// Assemble a linear CeedQFunction associated with a CeedOperator
    ///
    /// This returns a CeedVector containing a matrix at each quadrature point
    ///   providing the action of the CeedQFunction associated with the CeedOperator.
    ///   The vector 'assembled' is of shape
    ///     [num_elements, num_input_fields, num_output_fields, num_quad_points]
    ///   and contains column-major matrices representing the action of the
    ///   CeedQFunction for a corresponding quadrature point on an element. Inputs and
    ///   outputs are in the order provided by the user when adding CeedOperator fields.
    ///   For example, a CeedQFunction with inputs 'u' and 'gradu' and outputs 'gradv' and
    ///   'v', provided in that order, would result in an assembled QFunction that
    ///   consists of (1 + dim) x (dim + 1) matrices at each quadrature point acting
    ///  on the input [u, du_0, du_1] and producing the output [dv_0, dv_1, v].
    ///
    /// * 'op'        - Operator to assemble QFunction
    /// * 'assembled' - Vector to store assembled QFunction at
    ///                   quadrature points
    /// * 'rstr'      - ElemRestriction for Vector containing assembled
    ///                   QFunction
    ///
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

    /// Assemble the diagonal of a square linear Operator
    ///
    /// This overwrites a Vector with the diagonal of a linear Operator.
    ///
    /// Note: Currently only non-composite Operators with a single field and
    ///      composite Operators with single field sub-operators are supported.
    ///
    /// * 'op'        -     Operator to assemble QFunction
    /// * 'assembled' - Vector to store assembled Operator diagonal
    ///
    pub fn linear_asssemble_diagonal(&self, assembled: &mut crate::vector::Vector) {
        self.op_core.linear_assemble_add_diagonal(assembled)
    }

    /// Assemble the diagonal of a square linear Operator
    ///
    /// This sums into a Vector with the diagonal of a linear Operator.
    ///
    /// Note: Currently only non-composite Operators with a single field and
    ///      composite Operators with single field sub-operators are supported.
    ///
    /// * 'op'        -     Operator to assemble QFunction
    /// * 'assembled' - Vector to store assembled Operator diagonal
    ///
    pub fn linear_assemble_add_diagonal(&self, assembled: &mut crate::vector::Vector) {
        self.op_core.linear_assemble_add_diagonal(assembled)
    }

    /// Assemble the diagonal of a square linear Operator
    ///
    /// This overwrites a Vector with the diagonal of a linear Operator.
    ///
    /// Note: Currently only non-composite Operators with a single field and
    ///      composite Operators with single field sub-operators are supported.
    ///
    /// * 'op'        -     Operator to assemble QFunction
    /// * 'assembled' - Vector to store assembled CeedOperator point block
    ///                   diagonal, provided in row-major form with an
    ///                   ncomp * ncomp block at each node. The dimensions
    ///                   of this vector are derived from the active vector
    ///                   for the CeedOperator. The array has shape
    ///                   [nodes, component out, component in].
    ///
    pub fn linear_assemble_point_block_diagonal(&self, assembled: &mut crate::vector::Vector) {
        self.op_core.linear_assemble_add_diagonal(assembled)
    }

    /// Assemble the diagonal of a square linear Operator
    ///
    /// This sums into a Vector with the diagonal of a linear Operator.
    ///
    /// Note: Currently only non-composite Operators with a single field and
    ///      composite Operators with single field sub-operators are supported.
    ///
    /// * 'op'        -     Operator to assemble QFunction
    /// * 'assembled' - Vector to store assembled CeedOperator point block
    ///                   diagonal, provided in row-major form with an
    ///                   ncomp * ncomp block at each node. The dimensions
    ///                   of this vector are derived from the active vector
    ///                   for the CeedOperator. The array has shape
    ///                   [nodes, component out, component in].
    ///
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
        unsafe {
            bind_ceed::CeedOperatorMultigridLevelCreate(
                self.op_core.ptr,
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
                self.op_core.ptr,
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
                self.op_core.ptr,
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
impl CompositeOperator {
    // Constructor
    pub fn create(ceed: & crate::Ceed) -> Self {
        let mut ptr = std::ptr::null_mut();
        unsafe { bind_ceed::CeedCompositeOperatorCreate(ceed.ptr, &mut ptr) };
        let op_core = OperatorCore { ptr };
        Self { op_core }
    }

    /// Apply Operator to a vector
    ///
    /// * 'input'  - Input Vector
    /// * 'output' - Output Vector
    ///
    pub fn apply(&self, input: &crate::vector::Vector, output: &mut crate::vector::Vector) {
        self.op_core.apply(input, output)
    }

    /// Apply Operator to a vector and add result to output vector
    ///
    /// * 'input'  - Input Vector
    /// * 'output' - Output Vector
    ///
    pub fn apply_add(&self, input: &crate::vector::Vector, output: &mut crate::vector::Vector) {
        self.op_core.apply_add(input, output)
    }

    /// Add a sub-Operator to a Composite Operator
    ///
    /// * 'subop' - Sub-Operator
    ///
    pub fn add_sub_operator(&self, subop: &Operator) {
        unsafe { bind_ceed::CeedCompositeOperatorAddSub(self.op_core.ptr, subop.op_core.ptr) };
    }

    /// Assemble the diagonal of a square linear Operator
    ///
    /// This overwrites a Vector with the diagonal of a linear Operator.
    ///
    /// Note: Currently only non-composite Operators with a single field and
    ///      composite Operators with single field sub-operators are supported.
    ///
    /// * 'op'        -     Operator to assemble QFunction
    /// * 'assembled' - Vector to store assembled Operator diagonal
    ///
    pub fn linear_asssemble_diagonal(&self, assembled: &mut crate::vector::Vector) {
        self.op_core.linear_assemble_add_diagonal(assembled)
    }

    /// Assemble the point block diagonal of a square linear Operator
    ///
    /// This overwrites a Vector with the point block diagonal of a linear
    ///   Operator.
    ///
    /// Note: Currently only non-composite Operators with a single field and
    ///      composite Operators with single field sub-operators are supported.
    ///
    /// * 'op'        -     Operator to assemble QFunction
    /// * 'assembled' - Vector to store assembled Operator diagonal
    ///
    pub fn linear_assemble_add_diagonal(&self, assembled: &mut crate::vector::Vector) {
        self.op_core.linear_assemble_add_diagonal(assembled)
    }

    /// Assemble the diagonal of a square linear Operator
    ///
    /// This overwrites a Vector with the diagonal of a linear Operator.
    ///
    /// Note: Currently only non-composite Operators with a single field and
    ///      composite Operators with single field sub-operators are supported.
    ///
    /// * 'op'        -     Operator to assemble QFunction
    /// * 'assembled' - Vector to store assembled CeedOperator point block
    ///                   diagonal, provided in row-major form with an
    ///                   ncomp * ncomp block at each node. The dimensions
    ///                   of this vector are derived from the active vector
    ///                   for the CeedOperator. The array has shape
    ///                   [nodes, component out, component in].
    ///
    pub fn linear_assemble_point_block_diagonal(&self, assembled: &mut crate::vector::Vector) {
        self.op_core.linear_assemble_add_diagonal(assembled)
    }

    /// Assemble the diagonal of a square linear Operator
    ///
    /// This sums into a Vector with the diagonal of a linear Operator.
    ///
    /// Note: Currently only non-composite Operators with a single field and
    ///      composite Operators with single field sub-operators are supported.
    ///
    /// * 'op'        -     Operator to assemble QFunction
    /// * 'assembled' - Vector to store assembled CeedOperator point block
    ///                   diagonal, provided in row-major form with an
    ///                   ncomp * ncomp block at each node. The dimensions
    ///                   of this vector are derived from the active vector
    ///                   for the CeedOperator. The array has shape
    ///                   [nodes, component out, component in].
    ///
    pub fn linear_assemble_add_point_block_diagonal(&self, assembled: &mut crate::vector::Vector) {
        self.op_core.linear_assemble_add_diagonal(assembled)
    }
}

// -----------------------------------------------------------------------------
