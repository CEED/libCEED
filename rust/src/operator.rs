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
pub(crate) struct OperatorCore {
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
/// # use ceed::prelude::*;
/// # let ceed = ceed::Ceed::default_init();
/// let mut op = ceed.composite_operator();
///
/// // Sub operator field arguments
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
/// let qdata_mass = ceed.vector(q*ne);
/// let qdata_diff = ceed.vector(q*ne);
///
/// let qf_mass = ceed.q_function_interior_by_name("MassApply".to_string());
/// let mut op_mass = ceed.operator(&qf_mass, QFunctionOpt::None, QFunctionOpt::None);
/// op_mass.set_field("u", &r, &b, VectorOpt::Active);
/// op_mass.set_field("qdata", &rq, BasisOpt::Collocated, &qdata_mass);
/// op_mass.set_field("v", &r, &b, VectorOpt::Active);
///
/// op.add_sub_operator(&op_mass);
///
/// let qf_diff = ceed.q_function_interior_by_name("Poisson1DApply".to_string());
/// let mut op_diff = ceed.operator(&qf_diff, QFunctionOpt::None, QFunctionOpt::None);
/// op_diff.set_field("du", &r, &b, VectorOpt::Active);
/// op_diff.set_field("qdata", &rq, BasisOpt::Collocated, &qdata_diff);
/// op_diff.set_field("dv", &r, &b, VectorOpt::Active);
///
/// op.add_sub_operator(&op_diff);
///
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

    pub fn linear_assemble_diagonal(&self, assembled: &mut crate::vector::Vector) {
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
        ceed: &crate::Ceed,
        qf: impl Into<crate::qfunction::QFunctionOpt<'b>>,
        dqf: impl Into<crate::qfunction::QFunctionOpt<'b>>,
        dqfT: impl Into<crate::qfunction::QFunctionOpt<'b>>,
    ) -> Self {
        let mut ptr = std::ptr::null_mut();
        unsafe {
            bind_ceed::CeedOperatorCreate(
                ceed.ptr,
                qf.into().to_raw(),
                dqf.into().to_raw(),
                dqfT.into().to_raw(),
                &mut ptr,
            )
        };
        let op_core = OperatorCore { ptr };
        Self { op_core }
    }

    fn from_raw(ptr: bind_ceed::CeedOperator) -> Self {
        let op_core = OperatorCore { ptr };
        Self { op_core }
    }

    /// Apply Operator to a vector
    ///
    /// * `input`  - Input Vector
    /// * `output` - Output Vector
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
    /// * `input`  - Input Vector
    /// * `output` - Output Vector
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
    /// assert!(
    ///   (sum - (2.0 + ndofs as f64)).abs() < 1e-15,
    ///   "Incorrect interval length computed and added"
    /// );
    /// ```
    pub fn apply_add(&self, input: &crate::vector::Vector, output: &mut crate::vector::Vector) {
        self.op_core.apply_add(input, output)
    }

    /// Provide a field to a Operator for use by its QFunction
    ///
    /// * `fieldname` - Name of the field (to be matched with the name used by
    ///                   the QFunction)
    /// * `r`         - ElemRestriction
    /// * `b`         - Basis in which the field resides or `BasisCollocated` if
    ///                   collocated with quadrature points
    /// * `v`         - Vector to be used by Operator or `VectorActive` if field
    ///                   is active or `VectorNone` if using `Weight` with the
    ///                   QFunction
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

    /// Assemble the diagonal of a square linear Operator
    ///
    /// This overwrites a Vector with the diagonal of a linear Operator.
    ///
    /// Note: Currently only non-composite Operators with a single field and
    ///      composite Operators with single field sub-operators are supported.
    ///
    /// * `op`        - Operator to assemble QFunction
    /// * `assembled` - Vector to store assembled Operator diagonal
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
    ///   indu[p*i+0] = (2*i) as i32;
    ///   indu[p*i+1] = (2*i+1) as i32;
    ///   indu[p*i+2] = (2*i+2) as i32;
    /// }
    /// let ru = ceed.elem_restriction(ne, p, 1, 1, ndofs, ceed::MemType::Host, &indu);
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
    /// // Diagonal
    /// let mut diag = ceed.vector(ndofs);
    /// diag.set_value(0.0);
    /// op_mass.linear_assemble_diagonal(&mut diag);
    ///
    /// // Manual diagonal computation
    /// let mut true_diag = ceed.vector(ndofs);
    /// for i in 0..ndofs {
    ///   u.set_value(0.0);
    ///   {
    ///     let mut u_array = u.view_mut();
    ///     u_array[i] = 1.;
    ///   }
    ///
    ///   op_mass.apply(&u, &mut v);
    ///
    ///   {
    ///     let v_array = v.view_mut();
    ///     let mut true_array = true_diag.view_mut();
    ///     true_array[i] = v_array[i];
    ///   }
    /// }
    ///
    /// // Check
    /// let diag_array = diag.view();
    /// let true_array = true_diag.view();
    /// for i in 0..ndofs {
    ///   assert!(
    ///     (diag_array[i] - true_array[i]).abs() < 1e-15,
    ///     "Diagonal entry incorrect"
    ///   );
    /// }
    /// ```
    pub fn linear_assemble_diagonal(&self, assembled: &mut crate::vector::Vector) {
        self.op_core.linear_assemble_diagonal(assembled)
    }

    /// Assemble the diagonal of a square linear Operator
    ///
    /// This sums into a Vector with the diagonal of a linear Operator.
    ///
    /// Note: Currently only non-composite Operators with a single field and
    ///      composite Operators with single field sub-operators are supported.
    ///
    /// * `op`        - Operator to assemble QFunction
    /// * `assembled` - Vector to store assembled Operator diagonal
    ///
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
    ///   indu[p*i+0] = (2*i) as i32;
    ///   indu[p*i+1] = (2*i+1) as i32;
    ///   indu[p*i+2] = (2*i+2) as i32;
    /// }
    /// let ru = ceed.elem_restriction(ne, p, 1, 1, ndofs, ceed::MemType::Host, &indu);
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
    /// // Diagonal
    /// let mut diag = ceed.vector(ndofs);
    /// diag.set_value(1.0);
    /// op_mass.linear_assemble_add_diagonal(&mut diag);
    ///
    /// // Manual diagonal computation
    /// let mut true_diag = ceed.vector(ndofs);
    /// for i in 0..ndofs {
    ///   u.set_value(0.0);
    ///   {
    ///     let mut u_array = u.view_mut();
    ///     u_array[i] = 1.;
    ///   }
    ///
    ///   op_mass.apply(&u, &mut v);
    ///
    ///   {
    ///     let v_array = v.view_mut();
    ///     let mut true_array = true_diag.view_mut();
    ///     true_array[i] = v_array[i] + 1.0;
    ///   }
    /// }
    ///
    /// // Check
    /// let diag_array = diag.view();
    /// let true_array = true_diag.view();
    /// for i in 0..ndofs {
    ///   assert!(
    ///     (diag_array[i] - true_array[i]).abs() < 1e-15,
    ///     "Diagonal entry incorrect"
    ///   );
    /// }
    /// ```
    pub fn linear_assemble_add_diagonal(&self, assembled: &mut crate::vector::Vector) {
        self.op_core.linear_assemble_add_diagonal(assembled)
    }

    /// Assemble the point block diagonal of a square linear Operator
    ///
    /// This overwrites a Vector with the point block diagonal of a linear
    /// Operator.
    ///
    /// Note: Currently only non-composite Operators with a single field and
    ///      composite Operators with single field sub-operators are supported.
    ///
    /// * `op`        - Operator to assemble QFunction
    /// * `assembled` - Vector to store assembled CeedOperator point block
    ///                   diagonal, provided in row-major form with an
    ///                   `ncomp * ncomp` block at each node. The dimensions of
    ///                   this vector are derived from the active vector for
    ///                   the CeedOperator. The array has shape
    ///                   `[nodes, component out, component in]`.
    ///
    /// ```
    /// # use ceed::prelude::*;
    /// # let ceed = ceed::Ceed::default_init();
    /// let ne = 4;
    /// let p = 3;
    /// let q = 4;
    /// let ncomp = 2;
    /// let ndofs = p*ne-ne+1;
    ///
    /// // Vectors
    /// let x = ceed.vector_from_slice(&[-1., -0.5, 0.0, 0.5, 1.0]);
    /// let mut qdata = ceed.vector(ne*q);
    /// qdata.set_value(0.0);
    /// let mut u = ceed.vector(ncomp*ndofs);
    /// u.set_value(1.0);
    /// let mut v = ceed.vector(ncomp*ndofs);
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
    ///   indu[p*i+0] = (2*i) as i32;
    ///   indu[p*i+1] = (2*i+1) as i32;
    ///   indu[p*i+2] = (2*i+2) as i32;
    /// }
    /// let ru = ceed.elem_restriction(ne, p, ncomp, ndofs, ncomp*ndofs, ceed::MemType::Host, &indu);
    /// let strides : [i32; 3] = [1, q as i32, q as i32];
    /// let rq = ceed.strided_elem_restriction(ne, q, 1, q*ne, strides);
    ///
    /// // Bases
    /// let bx = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, ceed::QuadMode::Gauss);
    /// let bu = ceed.basis_tensor_H1_Lagrange(1, ncomp, p, q, ceed::QuadMode::Gauss);
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
    /// let mut mass_2_comp = |
    ///   q: usize,
    ///   inputs: &Vec<&[f64]>,
    ///   outputs: &mut Vec<&mut [f64]>,
    /// | -> i32
    /// {
    ///   let u = &inputs[0];
    ///   let qdata = &inputs[1];
    ///
    ///   let v = &mut outputs[0];
    ///
    ///   for i in 0..q {
    ///     v[i + 0*q] = u[i + 1*q] * qdata[i];
    ///     v[i + 1*q] = u[i + 0*q] * qdata[i];
    ///   }
    ///
    ///   return 0
    /// };
    ///
    /// let mut qf_mass = ceed.q_function_interior(1, Box::new(mass_2_comp));
    /// qf_mass.add_input("u", 2, ceed::EvalMode::Interp);
    /// qf_mass.add_input("qdata", 1, ceed::EvalMode::None);
    /// qf_mass.add_output("v", 2, ceed::EvalMode::Interp);
    ///
    /// let mut op_mass = ceed.operator(&qf_mass, QFunctionOpt::None, QFunctionOpt::None);
    /// op_mass.set_field("u", &ru, &bu, VectorOpt::Active);
    /// op_mass.set_field("qdata", &rq, BasisOpt::Collocated, &qdata);
    /// op_mass.set_field("v", &ru, &bu, VectorOpt::Active);
    ///
    /// // Diagonal
    /// let mut diag = ceed.vector(ncomp*ncomp*ndofs);
    /// diag.set_value(0.0);
    /// op_mass.linear_assemble_point_block_diagonal(&mut diag);
    ///
    /// // Manual diagonal computation
    /// let mut true_diag = ceed.vector(ncomp*ncomp*ndofs);
    /// for i in 0..ndofs {
    ///   for j in 0..ncomp {
    ///     u.set_value(0.0);
    ///     {
    ///       let mut u_array = u.view_mut();
    ///       u_array[i + j*ndofs] = 1.;
    ///     }
    ///
    ///     op_mass.apply(&u, &mut v);
    ///
    ///     {
    ///       let v_array = v.view_mut();
    ///       let mut true_array = true_diag.view_mut();
    ///       for k in 0..ncomp {
    ///         true_array[i*ncomp*ncomp + k*ncomp + j] = v_array[i + k*ndofs];
    ///       }
    ///     }
    ///   }
    /// }
    ///
    /// // Check
    /// let diag_array = diag.view();
    /// let true_array = true_diag.view();
    /// for i in 0..ncomp*ncomp*ndofs {
    ///   assert!(
    ///     (diag_array[i] - true_array[i]).abs() < 1e-15,
    ///     "Diagonal entry incorrect"
    ///   );
    /// }
    /// ```
    pub fn linear_assemble_point_block_diagonal(&self, assembled: &mut crate::vector::Vector) {
        self.op_core.linear_assemble_point_block_diagonal(assembled)
    }

    /// Assemble the point block diagonal of a square linear Operator
    ///
    /// This sums into a Vector with the point block diagonal of a linear
    /// Operator.
    ///
    /// Note: Currently only non-composite Operators with a single field and
    ///      composite Operators with single field sub-operators are supported.
    ///
    /// * `op`        -     Operator to assemble QFunction
    /// * `assembled` - Vector to store assembled CeedOperator point block
    ///                   diagonal, provided in row-major form with an
    ///                   `ncomp * ncomp` block at each node. The dimensions of
    ///                   this vector are derived from the active vector for
    ///                   the CeedOperator. The array has shape
    ///                   `[nodes, component out, component in]`.
    ///
    /// ```
    /// # use ceed::prelude::*;
    /// # let ceed = ceed::Ceed::default_init();
    /// let ne = 4;
    /// let p = 3;
    /// let q = 4;
    /// let ncomp = 2;
    /// let ndofs = p*ne-ne+1;
    ///
    /// // Vectors
    /// let x = ceed.vector_from_slice(&[-1., -0.5, 0.0, 0.5, 1.0]);
    /// let mut qdata = ceed.vector(ne*q);
    /// qdata.set_value(0.0);
    /// let mut u = ceed.vector(ncomp*ndofs);
    /// u.set_value(1.0);
    /// let mut v = ceed.vector(ncomp*ndofs);
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
    ///   indu[p*i+0] = (2*i) as i32;
    ///   indu[p*i+1] = (2*i+1) as i32;
    ///   indu[p*i+2] = (2*i+2) as i32;
    /// }
    /// let ru = ceed.elem_restriction(ne, p, ncomp, ndofs, ncomp*ndofs, ceed::MemType::Host, &indu);
    /// let strides : [i32; 3] = [1, q as i32, q as i32];
    /// let rq = ceed.strided_elem_restriction(ne, q, 1, q*ne, strides);
    ///
    /// // Bases
    /// let bx = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, ceed::QuadMode::Gauss);
    /// let bu = ceed.basis_tensor_H1_Lagrange(1, ncomp, p, q, ceed::QuadMode::Gauss);
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
    /// let mut mass_2_comp = |
    ///   q: usize,
    ///   inputs: &Vec<&[f64]>,
    ///   outputs: &mut Vec<&mut [f64]>,
    /// | -> i32
    /// {
    ///   let u = &inputs[0];
    ///   let qdata = &inputs[1];
    ///
    ///   let v = &mut outputs[0];
    ///
    ///   for i in 0..q {
    ///     v[i + 0*q] = u[i + 1*q] * qdata[i];
    ///     v[i + 1*q] = u[i + 0*q] * qdata[i];
    ///   }
    ///
    ///   return 0
    /// };
    ///
    /// let mut qf_mass = ceed.q_function_interior(1, Box::new(mass_2_comp));
    /// qf_mass.add_input("u", 2, ceed::EvalMode::Interp);
    /// qf_mass.add_input("qdata", 1, ceed::EvalMode::None);
    /// qf_mass.add_output("v", 2, ceed::EvalMode::Interp);
    ///
    /// let mut op_mass = ceed.operator(&qf_mass, QFunctionOpt::None, QFunctionOpt::None);
    /// op_mass.set_field("u", &ru, &bu, VectorOpt::Active);
    /// op_mass.set_field("qdata", &rq, BasisOpt::Collocated, &qdata);
    /// op_mass.set_field("v", &ru, &bu, VectorOpt::Active);
    ///
    /// // Diagonal
    /// let mut diag = ceed.vector(ncomp*ncomp*ndofs);
    /// diag.set_value(1.0);
    /// op_mass.linear_assemble_add_point_block_diagonal(&mut diag);
    ///
    /// // Manual diagonal computation
    /// let mut true_diag = ceed.vector(ncomp*ncomp*ndofs);
    /// for i in 0..ndofs {
    ///   for j in 0..ncomp {
    ///     u.set_value(0.0);
    ///     {
    ///       let mut u_array = u.view_mut();
    ///       u_array[i + j*ndofs] = 1.;
    ///     }
    ///
    ///     op_mass.apply(&u, &mut v);
    ///
    ///     {
    ///       let v_array = v.view_mut();
    ///       let mut true_array = true_diag.view_mut();
    ///       for k in 0..ncomp {
    ///         true_array[i*ncomp*ncomp + k*ncomp + j] = v_array[i + k*ndofs];
    ///       }
    ///     }
    ///   }
    /// }
    ///
    /// // Check
    /// let diag_array = diag.view();
    /// let true_array = true_diag.view();
    /// for i in 0..ncomp*ncomp*ndofs {
    ///   assert!(
    ///     (diag_array[i] - 1.0 - true_array[i]).abs() < 1e-15,
    ///     "Diagonal entry incorrect"
    ///   );
    /// }
    /// ```
    pub fn linear_assemble_add_point_block_diagonal(&self, assembled: &mut crate::vector::Vector) {
        self.op_core
            .linear_assemble_add_point_block_diagonal(assembled)
    }

    /// Create a multigrid coarse Operator and level transfer Operators for a
    ///   given Operator, creating the prolongation basis from the fine and
    ///   coarse grid interpolation
    ///
    /// * `p_mult_fine`  - Lvector multiplicity in parallel gather/scatter
    /// * `rstr_coarse`  - Coarse grid restriction
    /// * `basis_coarse` - Coarse grid active vector basis
    ///
    /// ```
    /// # use ceed::prelude::*;
    /// # let ceed = ceed::Ceed::default_init();
    /// let ne = 15;
    /// let p_coarse = 3;
    /// let p_fine = 5;
    /// let q = 6;
    /// let ndofs_coarse = p_coarse*ne-ne+1;
    /// let ndofs_fine = p_fine*ne-ne+1;
    ///
    /// // Vectors
    /// let x_array = (0..ne+1).map(|i| 2.0 * i as f64 / ne as f64 - 1.0).collect::<Vec<f64>>();
    /// let x = ceed.vector_from_slice(&x_array);
    /// let mut qdata = ceed.vector(ne*q);
    /// qdata.set_value(0.0);
    /// let mut u_coarse = ceed.vector(ndofs_coarse);
    /// u_coarse.set_value(1.0);
    /// let mut u_fine = ceed.vector(ndofs_fine);
    /// u_fine.set_value(1.0);
    /// let mut v_coarse = ceed.vector(ndofs_coarse);
    /// v_coarse.set_value(0.0);
    /// let mut v_fine = ceed.vector(ndofs_fine);
    /// v_fine.set_value(0.0);
    /// let mut multiplicity = ceed.vector(ndofs_fine);
    /// multiplicity.set_value(1.0);
    ///
    /// // Restrictions
    /// let strides : [i32; 3] = [1, q as i32, q as i32];
    /// let rq = ceed.strided_elem_restriction(ne, q, 1, q*ne, strides);
    ///
    /// let mut indx : Vec<i32> = vec![0; 2*ne];
    /// for i in 0..ne {
    ///   indx[2*i+0] = i as i32;
    ///   indx[2*i+1] = (i+1) as i32;
    /// }
    /// let rx = ceed.elem_restriction(ne, 2, 1, 1, ne+1, ceed::MemType::Host, &indx);
    ///
    /// let mut indu_coarse : Vec<i32> = vec![0; p_coarse*ne];
    /// for i in 0..ne {
    ///   for j in 0..p_coarse {
    ///     indu_coarse[p_coarse*i+j] = (i+j) as i32;
    ///   }
    /// }
    /// let ru_coarse = ceed.elem_restriction(ne, p_coarse, 1, 1, ndofs_coarse, ceed::MemType::Host, &indu_coarse);
    ///
    /// let mut indu_fine : Vec<i32> = vec![0; p_fine*ne];
    /// for i in 0..ne {
    ///   for j in 0..p_fine {
    ///     indu_fine[p_fine*i+j] = (i+j) as i32;
    ///   }
    /// }
    /// let ru_fine = ceed.elem_restriction(ne, p_fine, 1, 1, ndofs_fine, ceed::MemType::Host, &indu_fine);
    ///
    /// // Bases
    /// let bx = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, ceed::QuadMode::Gauss);
    /// let bu_coarse = ceed.basis_tensor_H1_Lagrange(1, 1, p_coarse, q, ceed::QuadMode::Gauss);
    /// let bu_fine = ceed.basis_tensor_H1_Lagrange(1, 1, p_fine, q, ceed::QuadMode::Gauss);
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
    /// let mut op_mass_fine = ceed.operator(&qf_mass, QFunctionOpt::None, QFunctionOpt::None);
    /// op_mass_fine.set_field("u", &ru_fine, &bu_fine, VectorOpt::Active);
    /// op_mass_fine.set_field("qdata", &rq, BasisOpt::Collocated, &qdata);
    /// op_mass_fine.set_field("v", &ru_fine, &bu_fine, VectorOpt::Active);
    ///
    /// // Multigrid setup
    /// let (op_mass_coarse, op_prolong, op_restrict) =
    ///   op_mass_fine.create_multigrid_level(&multiplicity, &ru_coarse, &bu_coarse);
    ///
    /// // Coarse problem
    /// u_coarse.set_value(1.0);
    /// op_mass_coarse.apply(&u_coarse, &mut v_coarse);
    ///
    /// // Check
    /// let array = v_coarse.view();
    /// let mut sum = 0.0;
    /// for i in 0..ndofs_coarse {
    ///   sum += array[i];
    /// }
    /// drop(array);
    /// assert!((sum - 2.0).abs() < 1e-15, "Incorrect interval length computed");
    ///
    /// // Prolong
    /// op_prolong.apply(&u_coarse, &mut u_fine);
    ///
    /// // Fine problem
    /// op_mass_fine.apply(&u_fine, &mut v_fine);
    ///
    /// // Check
    /// let array = v_fine.view();
    /// let mut sum = 0.0;
    /// for i in 0..ndofs_fine {
    ///   sum += array[i];
    /// }
    /// assert!((sum - 2.0).abs() < 1e-15, "Incorrect interval length computed");
    ///
    /// // Restrict
    /// op_restrict.apply(&v_fine, &mut v_coarse);
    ///
    /// // Check
    /// let array = v_coarse.view();
    /// let mut sum = 0.0;
    /// for i in 0..ndofs_coarse {
    ///   sum += array[i];
    /// }
    /// assert!((sum - 2.0).abs() < 1e-15, "Incorrect interval length computed");
    /// ```
    pub fn create_multigrid_level(
        &self,
        p_mult_fine: &crate::vector::Vector,
        rstr_coarse: &crate::elem_restriction::ElemRestriction,
        basis_coarse: &crate::basis::Basis,
    ) -> (
        crate::operator::Operator,
        crate::operator::Operator,
        crate::operator::Operator,
    ) {
        let mut ptr_coarse = std::ptr::null_mut();
        let mut ptr_prolong = std::ptr::null_mut();
        let mut ptr_restrict = std::ptr::null_mut();
        unsafe {
            bind_ceed::CeedOperatorMultigridLevelCreate(
                self.op_core.ptr,
                p_mult_fine.ptr,
                rstr_coarse.ptr,
                basis_coarse.ptr,
                &mut ptr_coarse,
                &mut ptr_prolong,
                &mut ptr_restrict,
            )
        };
        (
            Operator::from_raw(ptr_coarse),
            Operator::from_raw(ptr_prolong),
            Operator::from_raw(ptr_restrict),
        )
    }

    /// Create a multigrid coarse Operator and level transfer Operators for a
    ///   given Operator with a tensor basis for the active basis
    ///
    /// * `p_mult_fine`   - Lvector multiplicity in parallel gather/scatter
    /// * `rstr_coarse`   - Coarse grid restriction
    /// * `basis_coarse`  - Coarse grid active vector basis
    /// * `interp_c_to_f` - Matrix for coarse to fine
    ///
    /// ```
    /// # use ceed::prelude::*;
    /// # let ceed = ceed::Ceed::default_init();
    /// let ne = 15;
    /// let p_coarse = 3;
    /// let p_fine = 5;
    /// let q = 6;
    /// let ndofs_coarse = p_coarse*ne-ne+1;
    /// let ndofs_fine = p_fine*ne-ne+1;
    ///
    /// // Vectors
    /// let x_array = (0..ne+1).map(|i| 2.0 * i as f64 / ne as f64 - 1.0).collect::<Vec<f64>>();
    /// let x = ceed.vector_from_slice(&x_array);
    /// let mut qdata = ceed.vector(ne*q);
    /// qdata.set_value(0.0);
    /// let mut u_coarse = ceed.vector(ndofs_coarse);
    /// u_coarse.set_value(1.0);
    /// let mut u_fine = ceed.vector(ndofs_fine);
    /// u_fine.set_value(1.0);
    /// let mut v_coarse = ceed.vector(ndofs_coarse);
    /// v_coarse.set_value(0.0);
    /// let mut v_fine = ceed.vector(ndofs_fine);
    /// v_fine.set_value(0.0);
    /// let mut multiplicity = ceed.vector(ndofs_fine);
    /// multiplicity.set_value(1.0);
    ///
    /// // Restrictions
    /// let strides : [i32; 3] = [1, q as i32, q as i32];
    /// let rq = ceed.strided_elem_restriction(ne, q, 1, q*ne, strides);
    ///
    /// let mut indx : Vec<i32> = vec![0; 2*ne];
    /// for i in 0..ne {
    ///   indx[2*i+0] = i as i32;
    ///   indx[2*i+1] = (i+1) as i32;
    /// }
    /// let rx = ceed.elem_restriction(ne, 2, 1, 1, ne+1, ceed::MemType::Host, &indx);
    ///
    /// let mut indu_coarse : Vec<i32> = vec![0; p_coarse*ne];
    /// for i in 0..ne {
    ///   for j in 0..p_coarse {
    ///     indu_coarse[p_coarse*i+j] = (i+j) as i32;
    ///   }
    /// }
    /// let ru_coarse = ceed.elem_restriction(ne, p_coarse, 1, 1, ndofs_coarse, ceed::MemType::Host, &indu_coarse);
    ///
    /// let mut indu_fine : Vec<i32> = vec![0; p_fine*ne];
    /// for i in 0..ne {
    ///   for j in 0..p_fine {
    ///     indu_fine[p_fine*i+j] = (i+j) as i32;
    ///   }
    /// }
    /// let ru_fine = ceed.elem_restriction(ne, p_fine, 1, 1, ndofs_fine, ceed::MemType::Host, &indu_fine);
    ///
    /// // Bases
    /// let bx = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, ceed::QuadMode::Gauss);
    /// let bu_coarse = ceed.basis_tensor_H1_Lagrange(1, 1, p_coarse, q, ceed::QuadMode::Gauss);
    /// let bu_fine = ceed.basis_tensor_H1_Lagrange(1, 1, p_fine, q, ceed::QuadMode::Gauss);
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
    /// let mut op_mass_fine = ceed.operator(&qf_mass, QFunctionOpt::None, QFunctionOpt::None);
    /// op_mass_fine.set_field("u", &ru_fine, &bu_fine, VectorOpt::Active);
    /// op_mass_fine.set_field("qdata", &rq, BasisOpt::Collocated, &qdata);
    /// op_mass_fine.set_field("v", &ru_fine, &bu_fine, VectorOpt::Active);
    ///
    /// // Multigrid setup
    /// let mut interp_c_to_f : Vec<f64> = vec![0.; p_coarse*p_fine];
    /// {
    ///   let mut coarse = ceed.vector(p_coarse);
    ///   let mut fine = ceed.vector(p_fine);
    ///   let basis_c_to_f = ceed.basis_tensor_H1_Lagrange(1, 1, p_coarse, p_fine, ceed::QuadMode::GaussLobatto);
    ///   for i in 0..p_coarse {
    ///     coarse.set_value(0.0);
    ///     {
    ///        let mut array = coarse.view_mut();
    ///        array[i] = 1.;
    ///     }
    ///     basis_c_to_f.apply(1, ceed::TransposeMode::NoTranspose, ceed::EvalMode::Interp,
    ///                        &coarse, &mut fine);
    ///     let array = fine.view();
    ///     for j in 0..p_fine {
    ///       interp_c_to_f[j*p_coarse + i] = array[j];
    ///     }
    ///   }
    /// }
    /// let (op_mass_coarse, op_prolong, op_restrict) =
    ///   op_mass_fine.create_multigrid_level_tensor_H1(&multiplicity, &ru_coarse, &bu_coarse, &interp_c_to_f);
    ///
    /// // Coarse problem
    /// u_coarse.set_value(1.0);
    /// op_mass_coarse.apply(&u_coarse, &mut v_coarse);
    ///
    /// // Check
    /// let array = v_coarse.view();
    /// let mut sum = 0.0;
    /// for i in 0..ndofs_coarse {
    ///   sum += array[i];
    /// }
    /// drop(array);
    /// assert!((sum - 2.0).abs() < 1e-15, "Incorrect interval length computed");
    ///
    /// // Prolong
    /// op_prolong.apply(&u_coarse, &mut u_fine);
    ///
    /// // Fine problem
    /// op_mass_fine.apply(&u_fine, &mut v_fine);
    ///
    /// // Check
    /// let array = v_fine.view();
    /// let mut sum = 0.0;
    /// for i in 0..ndofs_fine {
    ///   sum += array[i];
    /// }
    /// assert!((sum - 2.0).abs() < 1e-15, "Incorrect interval length computed");
    ///
    /// // Restrict
    /// op_restrict.apply(&v_fine, &mut v_coarse);
    ///
    /// // Check
    /// let array = v_coarse.view();
    /// let mut sum = 0.0;
    /// for i in 0..ndofs_coarse {
    ///   sum += array[i];
    /// }
    /// assert!((sum - 2.0).abs() < 1e-15, "Incorrect interval length computed");
    /// ```
    pub fn create_multigrid_level_tensor_H1(
        &self,
        p_mult_fine: &crate::vector::Vector,
        rstr_coarse: &crate::elem_restriction::ElemRestriction,
        basis_coarse: &crate::basis::Basis,
        interpCtoF: &Vec<f64>,
    ) -> (
        crate::operator::Operator,
        crate::operator::Operator,
        crate::operator::Operator,
    ) {
        let mut ptr_coarse = std::ptr::null_mut();
        let mut ptr_prolong = std::ptr::null_mut();
        let mut ptr_restrict = std::ptr::null_mut();
        unsafe {
            bind_ceed::CeedOperatorMultigridLevelCreateTensorH1(
                self.op_core.ptr,
                p_mult_fine.ptr,
                rstr_coarse.ptr,
                basis_coarse.ptr,
                interpCtoF.as_ptr(),
                &mut ptr_coarse,
                &mut ptr_prolong,
                &mut ptr_restrict,
            )
        };
        (
            Operator::from_raw(ptr_coarse),
            Operator::from_raw(ptr_prolong),
            Operator::from_raw(ptr_restrict),
        )
    }

    /// Create a multigrid coarse Operator and level transfer Operators for a
    ///   given Operator with a non-tensor basis for the active basis
    ///
    /// * `p_mult_fine`   - Lvector multiplicity in parallel gather/scatter
    /// * `rstr_coarse`   - Coarse grid restriction
    /// * `basis_coarse`  - Coarse grid active vector basis
    /// * `interp_c_to_f` - Matrix for coarse to fine
    ///
    /// ```
    /// # use ceed::prelude::*;
    /// # let ceed = ceed::Ceed::default_init();
    /// let ne = 15;
    /// let p_coarse = 3;
    /// let p_fine = 5;
    /// let q = 6;
    /// let ndofs_coarse = p_coarse*ne-ne+1;
    /// let ndofs_fine = p_fine*ne-ne+1;
    ///
    /// // Vectors
    /// let x_array = (0..ne+1).map(|i| 2.0 * i as f64 / ne as f64 - 1.0).collect::<Vec<f64>>();
    /// let x = ceed.vector_from_slice(&x_array);
    /// let mut qdata = ceed.vector(ne*q);
    /// qdata.set_value(0.0);
    /// let mut u_coarse = ceed.vector(ndofs_coarse);
    /// u_coarse.set_value(1.0);
    /// let mut u_fine = ceed.vector(ndofs_fine);
    /// u_fine.set_value(1.0);
    /// let mut v_coarse = ceed.vector(ndofs_coarse);
    /// v_coarse.set_value(0.0);
    /// let mut v_fine = ceed.vector(ndofs_fine);
    /// v_fine.set_value(0.0);
    /// let mut multiplicity = ceed.vector(ndofs_fine);
    /// multiplicity.set_value(1.0);
    ///
    /// // Restrictions
    /// let strides : [i32; 3] = [1, q as i32, q as i32];
    /// let rq = ceed.strided_elem_restriction(ne, q, 1, q*ne, strides);
    ///
    /// let mut indx : Vec<i32> = vec![0; 2*ne];
    /// for i in 0..ne {
    ///   indx[2*i+0] = i as i32;
    ///   indx[2*i+1] = (i+1) as i32;
    /// }
    /// let rx = ceed.elem_restriction(ne, 2, 1, 1, ne+1, ceed::MemType::Host, &indx);
    ///
    /// let mut indu_coarse : Vec<i32> = vec![0; p_coarse*ne];
    /// for i in 0..ne {
    ///   for j in 0..p_coarse {
    ///     indu_coarse[p_coarse*i+j] = (i+j) as i32;
    ///   }
    /// }
    /// let ru_coarse = ceed.elem_restriction(ne, p_coarse, 1, 1, ndofs_coarse, ceed::MemType::Host, &indu_coarse);
    ///
    /// let mut indu_fine : Vec<i32> = vec![0; p_fine*ne];
    /// for i in 0..ne {
    ///   for j in 0..p_fine {
    ///     indu_fine[p_fine*i+j] = (i+j) as i32;
    ///   }
    /// }
    /// let ru_fine = ceed.elem_restriction(ne, p_fine, 1, 1, ndofs_fine, ceed::MemType::Host, &indu_fine);
    ///
    /// // Bases
    /// let bx = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, ceed::QuadMode::Gauss);
    /// let bu_coarse = ceed.basis_tensor_H1_Lagrange(1, 1, p_coarse, q, ceed::QuadMode::Gauss);
    /// let bu_fine = ceed.basis_tensor_H1_Lagrange(1, 1, p_fine, q, ceed::QuadMode::Gauss);
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
    /// let mut op_mass_fine = ceed.operator(&qf_mass, QFunctionOpt::None, QFunctionOpt::None);
    /// op_mass_fine.set_field("u", &ru_fine, &bu_fine, VectorOpt::Active);
    /// op_mass_fine.set_field("qdata", &rq, BasisOpt::Collocated, &qdata);
    /// op_mass_fine.set_field("v", &ru_fine, &bu_fine, VectorOpt::Active);
    ///
    /// // Multigrid setup
    /// let mut interp_c_to_f : Vec<f64> = vec![0.; p_coarse*p_fine];
    /// {
    ///   let mut coarse = ceed.vector(p_coarse);
    ///   let mut fine = ceed.vector(p_fine);
    ///   let basis_c_to_f = ceed.basis_tensor_H1_Lagrange(1, 1, p_coarse, p_fine, ceed::QuadMode::GaussLobatto);
    ///   for i in 0..p_coarse {
    ///     coarse.set_value(0.0);
    ///     {
    ///        let mut array = coarse.view_mut();
    ///        array[i] = 1.;
    ///     }
    ///     basis_c_to_f.apply(1, ceed::TransposeMode::NoTranspose, ceed::EvalMode::Interp,
    ///                        &coarse, &mut fine);
    ///     let array = fine.view();
    ///     for j in 0..p_fine {
    ///       interp_c_to_f[j*p_coarse + i] = array[j];
    ///     }
    ///   }
    /// }
    /// let (op_mass_coarse, op_prolong, op_restrict) =
    ///   op_mass_fine.create_multigrid_level_H1(&multiplicity, &ru_coarse, &bu_coarse, &interp_c_to_f);
    ///
    /// // Coarse problem
    /// u_coarse.set_value(1.0);
    /// op_mass_coarse.apply(&u_coarse, &mut v_coarse);
    ///
    /// // Check
    /// let array = v_coarse.view();
    /// let mut sum = 0.0;
    /// for i in 0..ndofs_coarse {
    ///   sum += array[i];
    /// }
    /// drop(array);
    /// assert!((sum - 2.0).abs() < 1e-15, "Incorrect interval length computed");
    ///
    /// // Prolong
    /// op_prolong.apply(&u_coarse, &mut u_fine);
    ///
    /// // Fine problem
    /// op_mass_fine.apply(&u_fine, &mut v_fine);
    ///
    /// // Check
    /// let array = v_fine.view();
    /// let mut sum = 0.0;
    /// for i in 0..ndofs_fine {
    ///   sum += array[i];
    /// }
    /// assert!((sum - 2.0).abs() < 1e-15, "Incorrect interval length computed");
    ///
    /// // Restrict
    /// op_restrict.apply(&v_fine, &mut v_coarse);
    ///
    /// // Check
    /// let array = v_coarse.view();
    /// let mut sum = 0.0;
    /// for i in 0..ndofs_coarse {
    ///   sum += array[i];
    /// }
    /// assert!((sum - 2.0).abs() < 1e-15, "Incorrect interval length computed");
    /// ```
    pub fn create_multigrid_level_H1(
        &self,
        p_mult_fine: &crate::vector::Vector,
        rstr_coarse: &crate::elem_restriction::ElemRestriction,
        basis_coarse: &crate::basis::Basis,
        interpCtoF: &Vec<f64>,
    ) -> (
        crate::operator::Operator,
        crate::operator::Operator,
        crate::operator::Operator,
    ) {
        let mut ptr_coarse = std::ptr::null_mut();
        let mut ptr_prolong = std::ptr::null_mut();
        let mut ptr_restrict = std::ptr::null_mut();
        unsafe {
            bind_ceed::CeedOperatorMultigridLevelCreateH1(
                self.op_core.ptr,
                p_mult_fine.ptr,
                rstr_coarse.ptr,
                basis_coarse.ptr,
                interpCtoF.as_ptr(),
                &mut ptr_coarse,
                &mut ptr_prolong,
                &mut ptr_restrict,
            )
        };
        (
            Operator::from_raw(ptr_coarse),
            Operator::from_raw(ptr_prolong),
            Operator::from_raw(ptr_restrict),
        )
    }
}

// -----------------------------------------------------------------------------
// Composite Operator
// -----------------------------------------------------------------------------
impl CompositeOperator {
    // Constructor
    pub fn create(ceed: &crate::Ceed) -> Self {
        let mut ptr = std::ptr::null_mut();
        unsafe { bind_ceed::CeedCompositeOperatorCreate(ceed.ptr, &mut ptr) };
        let op_core = OperatorCore { ptr };
        Self { op_core }
    }

    /// Apply Operator to a vector
    ///
    /// * `input`  - Input Vector
    /// * `output` - Output Vector
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
    /// let mut qdata_mass = ceed.vector(ne*q);
    /// qdata_mass.set_value(0.0);
    /// let mut qdata_diff = ceed.vector(ne*q);
    /// qdata_diff.set_value(0.0);
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
    /// // Set up operators
    /// let qf_build_mass = ceed.q_function_interior_by_name("Mass1DBuild".to_string());
    /// let mut op_build_mass = ceed.operator(&qf_build_mass, QFunctionOpt::None, QFunctionOpt::None);
    /// op_build_mass.set_field("dx", &rx, &bx, VectorOpt::Active);
    /// op_build_mass.set_field("weights", ElemRestrictionOpt::None, &bx, VectorOpt::None);
    /// op_build_mass.set_field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active);
    ///
    /// op_build_mass.apply(&x, &mut qdata_mass);
    ///
    /// let qf_build_diff = ceed.q_function_interior_by_name("Poisson1DBuild".to_string());
    /// let mut op_build_diff = ceed.operator(&qf_build_diff, QFunctionOpt::None, QFunctionOpt::None);
    /// op_build_diff.set_field("dx", &rx, &bx, VectorOpt::Active);
    /// op_build_diff.set_field("weights", ElemRestrictionOpt::None, &bx, VectorOpt::None);
    /// op_build_diff.set_field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active);
    ///
    /// op_build_diff.apply(&x, &mut qdata_diff);
    ///
    /// // Application operator
    /// let mut op_composite = ceed.composite_operator();
    ///
    /// let qf_mass = ceed.q_function_interior_by_name("MassApply".to_string());
    /// let mut op_mass = ceed.operator(&qf_mass, QFunctionOpt::None, QFunctionOpt::None);
    /// op_mass.set_field("u", &ru, &bu, VectorOpt::Active);
    /// op_mass.set_field("qdata", &rq, BasisOpt::Collocated, &qdata_mass);
    /// op_mass.set_field("v", &ru, &bu, VectorOpt::Active);
    ///
    /// op_composite.add_sub_operator(&op_mass);
    ///
    /// let qf_diff = ceed.q_function_interior_by_name("Poisson1DApply".to_string());
    /// let mut op_diff = ceed.operator(&qf_diff, QFunctionOpt::None, QFunctionOpt::None);
    /// op_diff.set_field("du", &ru, &bu, VectorOpt::Active);
    /// op_diff.set_field("qdata", &rq, BasisOpt::Collocated, &qdata_diff);
    /// op_diff.set_field("dv", &ru, &bu, VectorOpt::Active);
    ///
    /// op_composite.add_sub_operator(&op_diff);
    ///
    /// v.set_value(0.0);
    /// op_composite.apply(&u, &mut v);
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
    /// * `input`  - Input Vector
    /// * `output` - Output Vector
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
    /// let mut qdata_mass = ceed.vector(ne*q);
    /// qdata_mass.set_value(0.0);
    /// let mut qdata_diff = ceed.vector(ne*q);
    /// qdata_diff.set_value(0.0);
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
    /// // Set up operators
    /// let qf_build_mass = ceed.q_function_interior_by_name("Mass1DBuild".to_string());
    /// let mut op_build_mass = ceed.operator(&qf_build_mass, QFunctionOpt::None, QFunctionOpt::None);
    /// op_build_mass.set_field("dx", &rx, &bx, VectorOpt::Active);
    /// op_build_mass.set_field("weights", ElemRestrictionOpt::None, &bx, VectorOpt::None);
    /// op_build_mass.set_field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active);
    ///
    /// op_build_mass.apply(&x, &mut qdata_mass);
    ///
    /// let qf_build_diff = ceed.q_function_interior_by_name("Poisson1DBuild".to_string());
    /// let mut op_build_diff = ceed.operator(&qf_build_diff, QFunctionOpt::None, QFunctionOpt::None);
    /// op_build_diff.set_field("dx", &rx, &bx, VectorOpt::Active);
    /// op_build_diff.set_field("weights", ElemRestrictionOpt::None, &bx, VectorOpt::None);
    /// op_build_diff.set_field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active);
    ///
    /// op_build_diff.apply(&x, &mut qdata_diff);
    ///
    /// // Application operator
    /// let mut op_composite = ceed.composite_operator();
    ///
    /// let qf_mass = ceed.q_function_interior_by_name("MassApply".to_string());
    /// let mut op_mass = ceed.operator(&qf_mass, QFunctionOpt::None, QFunctionOpt::None);
    /// op_mass.set_field("u", &ru, &bu, VectorOpt::Active);
    /// op_mass.set_field("qdata", &rq, BasisOpt::Collocated, &qdata_mass);
    /// op_mass.set_field("v", &ru, &bu, VectorOpt::Active);
    ///
    /// op_composite.add_sub_operator(&op_mass);
    ///
    /// let qf_diff = ceed.q_function_interior_by_name("Poisson1DApply".to_string());
    /// let mut op_diff = ceed.operator(&qf_diff, QFunctionOpt::None, QFunctionOpt::None);
    /// op_diff.set_field("du", &ru, &bu, VectorOpt::Active);
    /// op_diff.set_field("qdata", &rq, BasisOpt::Collocated, &qdata_diff);
    /// op_diff.set_field("dv", &ru, &bu, VectorOpt::Active);
    ///
    /// op_composite.add_sub_operator(&op_diff);
    ///
    /// v.set_value(1.0);
    /// op_composite.apply_add(&u, &mut v);
    ///
    /// // Check
    /// let array = v.view();
    /// let mut sum = 0.0;
    /// for i in 0..ndofs {
    ///   sum += array[i];
    /// }
    /// assert!((sum - (2.0 + ndofs as f64)).abs() < 1e-15, "Incorrect interval length computed");
    /// ```
    pub fn apply_add(&self, input: &crate::vector::Vector, output: &mut crate::vector::Vector) {
        self.op_core.apply_add(input, output)
    }

    /// Add a sub-Operator to a Composite Operator
    ///
    /// * `subop` - Sub-Operator
    ///
    /// ```
    /// # use ceed::prelude::*;
    /// # let ceed = ceed::Ceed::default_init();
    /// let mut op = ceed.composite_operator();
    ///
    /// let qf_mass = ceed.q_function_interior_by_name("MassApply".to_string());
    /// let op_mass = ceed.operator(&qf_mass, QFunctionOpt::None, QFunctionOpt::None);
    /// op.add_sub_operator(&op_mass);
    ///
    /// let qf_diff = ceed.q_function_interior_by_name("Poisson1DApply".to_string());
    /// let op_diff = ceed.operator(&qf_diff, QFunctionOpt::None, QFunctionOpt::None);
    /// op.add_sub_operator(&op_diff);
    /// ```
    pub fn add_sub_operator(&mut self, subop: &Operator) {
        unsafe { bind_ceed::CeedCompositeOperatorAddSub(self.op_core.ptr, subop.op_core.ptr) };
    }

    /// Assemble the diagonal of a square linear Operator
    ///
    /// This overwrites a Vector with the diagonal of a linear Operator.
    ///
    /// Note: Currently only non-composite Operators with a single field and
    ///      composite Operators with single field sub-operators are supported.
    ///
    /// * `op`        - Operator to assemble QFunction
    /// * `assembled` - Vector to store assembled Operator diagonal
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
    /// * `op`        - Operator to assemble QFunction
    /// * `assembled` - Vector to store assembled Operator diagonal
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
    /// * `op`        - Operator to assemble QFunction
    /// * `assembled` - Vector to store assembled CeedOperator point block
    ///                   diagonal, provided in row-major form with an
    ///                   `ncomp * ncomp` block at each node. The dimensions of
    ///                   this vector are derived from the active vector for
    ///                   the CeedOperator. The array has shape
    ///                   `[nodes, component out, component in]`.
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
    /// * `op`        - Operator to assemble QFunction
    /// * `assembled` - Vector to store assembled CeedOperator point block
    ///                   diagonal, provided in row-major form with an
    ///                   `ncomp * ncomp` block at each node. The dimensions of
    ///                   this vector are derived from the active vector for
    ///                   the CeedOperator. The array has shape
    ///                   `[nodes, component out, component in]`.
    pub fn linear_assemble_add_point_block_diagonal(&self, assembled: &mut crate::vector::Vector) {
        self.op_core.linear_assemble_add_diagonal(assembled)
    }
}

// -----------------------------------------------------------------------------
