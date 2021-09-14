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

//! A Ceed Operator defines the finite/spectral element operator associated to a
//! Ceed QFunction. A Ceed Operator connects Ceed ElemRestrictions,
//! Ceed Bases, and Ceed QFunctions.

use crate::prelude::*;

// -----------------------------------------------------------------------------
// CeedOperator Field context wrapper
// -----------------------------------------------------------------------------
#[derive(Debug)]
pub struct OperatorField<'a> {
    pub(crate) ptr: bind_ceed::CeedOperatorField,
    _lifeline: PhantomData<&'a ()>,
}

// -----------------------------------------------------------------------------
// Implementations
// -----------------------------------------------------------------------------
impl<'a> OperatorField<'a> {
    /// Get the name of an OperatorField
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let qf = ceed.q_function_interior_by_name("Mass1DBuild")?;
    ///
    /// // Operator field arguments
    /// let ne = 3;
    /// let q = 4 as usize;
    /// let mut ind: Vec<i32> = vec![0; 2 * ne];
    /// for i in 0..ne {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed.elem_restriction(ne, 2, 1, 1, ne + 1, MemType::Host, &ind)?;
    /// let strides: [i32; 3] = [1, q as i32, q as i32];
    /// let rq = ceed.strided_elem_restriction(ne, 2, 1, q * ne, strides)?;
    ///
    /// let b = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, QuadMode::Gauss)?;
    ///
    /// // Operator fields
    /// let op = ceed
    ///     .operator(&qf, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("dx", &r, &b, VectorOpt::Active)?
    ///     .field("weights", ElemRestrictionOpt::None, &b, VectorOpt::None)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active)?;
    ///
    /// let inputs = op.inputs()?;
    ///
    /// assert_eq!(inputs[0].name(), "dx", "Incorrect input name");
    /// assert_eq!(inputs[1].name(), "weights", "Incorrect input name");
    /// # Ok(())
    /// # }
    /// ```
    pub fn name(&self) -> &str {
        let mut name_ptr: *mut std::os::raw::c_char = std::ptr::null_mut();
        unsafe {
            bind_ceed::CeedOperatorFieldGetName(self.ptr, &mut name_ptr);
        }
        unsafe { CStr::from_ptr(name_ptr) }.to_str().unwrap()
    }

    /// Get the ElemRestriction of an OperatorField
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let qf = ceed.q_function_interior_by_name("Mass1DBuild")?;
    ///
    /// // Operator field arguments
    /// let ne = 3;
    /// let q = 4 as usize;
    /// let mut ind: Vec<i32> = vec![0; 2 * ne];
    /// for i in 0..ne {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed.elem_restriction(ne, 2, 1, 1, ne + 1, MemType::Host, &ind)?;
    /// let strides: [i32; 3] = [1, q as i32, q as i32];
    /// let rq = ceed.strided_elem_restriction(ne, 2, 1, q * ne, strides)?;
    ///
    /// let b = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, QuadMode::Gauss)?;
    ///
    /// // Operator fields
    /// let op = ceed
    ///     .operator(&qf, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("dx", &r, &b, VectorOpt::Active)?
    ///     .field("weights", ElemRestrictionOpt::None, &b, VectorOpt::None)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active)?;
    ///
    /// let inputs = op.inputs()?;
    ///
    /// assert!(
    ///     inputs[0].elem_restriction().is_some(),
    ///     "Incorrect field ElemRestriction"
    /// );
    /// if let ElemRestrictionOpt::Some(r) = inputs[0].elem_restriction() {
    ///     assert_eq!(
    ///         r.num_elements(),
    ///         ne,
    ///         "Incorrect field ElemRestriction number of elements"
    ///     );
    /// }
    ///
    /// assert!(
    ///     inputs[1].elem_restriction().is_none(),
    ///     "Incorrect field ElemRestriction"
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn elem_restriction(&self) -> ElemRestrictionOpt {
        let mut ptr = std::ptr::null_mut();
        unsafe {
            bind_ceed::CeedOperatorFieldGetElemRestriction(self.ptr, &mut ptr);
        }
        if ptr == unsafe { bind_ceed::CEED_ELEMRESTRICTION_NONE } {
            ElemRestrictionOpt::None
        } else {
            let slice = unsafe {
                std::slice::from_raw_parts(
                    &ptr as *const bind_ceed::CeedElemRestriction as *const crate::ElemRestriction,
                    1 as usize,
                )
            };
            ElemRestrictionOpt::Some(&slice[0])
        }
    }

    /// Get the Basis of an OperatorField
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let qf = ceed.q_function_interior_by_name("Mass1DBuild")?;
    ///
    /// // Operator field arguments
    /// let ne = 3;
    /// let q = 4 as usize;
    /// let mut ind: Vec<i32> = vec![0; 2 * ne];
    /// for i in 0..ne {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed.elem_restriction(ne, 2, 1, 1, ne + 1, MemType::Host, &ind)?;
    /// let strides: [i32; 3] = [1, q as i32, q as i32];
    /// let rq = ceed.strided_elem_restriction(ne, 2, 1, q * ne, strides)?;
    ///
    /// let b = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, QuadMode::Gauss)?;
    ///
    /// // Operator fields
    /// let op = ceed
    ///     .operator(&qf, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("dx", &r, &b, VectorOpt::Active)?
    ///     .field("weights", ElemRestrictionOpt::None, &b, VectorOpt::None)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active)?;
    ///
    /// let inputs = op.inputs()?;
    ///
    /// assert!(inputs[0].basis().is_some(), "Incorrect field Basis");
    /// if let BasisOpt::Some(b) = inputs[0].basis() {
    ///     assert_eq!(
    ///         b.num_quadrature_points(),
    ///         q,
    ///         "Incorrect field Basis number of quadrature points"
    ///     );
    /// }
    /// assert!(inputs[1].basis().is_some(), "Incorrect field Basis");
    /// if let BasisOpt::Some(b) = inputs[0].basis() {
    ///     assert_eq!(
    ///         b.num_quadrature_points(),
    ///         q,
    ///         "Incorrect field Basis number of quadrature points"
    ///     );
    /// }
    ///
    /// let outputs = op.outputs()?;
    ///
    /// assert!(outputs[0].basis().is_collocated(), "Incorrect field Basis");
    /// # Ok(())
    /// # }
    /// ```
    pub fn basis(&self) -> BasisOpt {
        let mut ptr = std::ptr::null_mut();
        unsafe {
            bind_ceed::CeedOperatorFieldGetBasis(self.ptr, &mut ptr);
        }
        if ptr == unsafe { bind_ceed::CEED_BASIS_COLLOCATED } {
            BasisOpt::Collocated
        } else {
            let slice = unsafe {
                std::slice::from_raw_parts(
                    &ptr as *const bind_ceed::CeedBasis as *const crate::Basis,
                    1 as usize,
                )
            };
            BasisOpt::Some(&slice[0])
        }
    }

    /// Get the Vector of an OperatorField
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let qf = ceed.q_function_interior_by_name("Mass1DBuild")?;
    ///
    /// // Operator field arguments
    /// let ne = 3;
    /// let q = 4 as usize;
    /// let mut ind: Vec<i32> = vec![0; 2 * ne];
    /// for i in 0..ne {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed.elem_restriction(ne, 2, 1, 1, ne + 1, MemType::Host, &ind)?;
    /// let strides: [i32; 3] = [1, q as i32, q as i32];
    /// let rq = ceed.strided_elem_restriction(ne, 2, 1, q * ne, strides)?;
    ///
    /// let b = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, QuadMode::Gauss)?;
    ///
    /// // Operator fields
    /// let op = ceed
    ///     .operator(&qf, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("dx", &r, &b, VectorOpt::Active)?
    ///     .field("weights", ElemRestrictionOpt::None, &b, VectorOpt::None)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active)?;
    ///
    /// let inputs = op.inputs()?;
    ///
    /// assert!(inputs[0].vector().is_active(), "Incorrect field Vector");
    /// assert!(inputs[1].vector().is_none(), "Incorrect field Vector");
    /// # Ok(())
    /// # }
    /// ```
    pub fn vector(&self) -> VectorOpt {
        let mut ptr = std::ptr::null_mut();
        unsafe {
            bind_ceed::CeedOperatorFieldGetVector(self.ptr, &mut ptr);
        }
        if ptr == unsafe { bind_ceed::CEED_VECTOR_ACTIVE } {
            VectorOpt::Active
        } else if ptr == unsafe { bind_ceed::CEED_VECTOR_NONE } {
            VectorOpt::None
        } else {
            let slice = unsafe {
                std::slice::from_raw_parts(
                    &ptr as *const bind_ceed::CeedVector as *const crate::Vector,
                    1 as usize,
                )
            };
            VectorOpt::Some(&slice[0])
        }
    }
}

// -----------------------------------------------------------------------------
// CeedOperator context wrapper
// -----------------------------------------------------------------------------
#[derive(Debug)]
pub(crate) struct OperatorCore<'a> {
    ptr: bind_ceed::CeedOperator,
    _lifeline: PhantomData<&'a ()>,
}

#[derive(Debug)]
pub struct Operator<'a> {
    op_core: OperatorCore<'a>,
}

#[derive(Debug)]
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
        let mut sizeloc = crate::MAX_BUFFER_LENGTH;
        let cstring = unsafe {
            let file = bind_ceed::open_memstream(&mut ptr, &mut sizeloc);
            bind_ceed::CeedOperatorView(self.ptr, file);
            bind_ceed::fclose(file);
            CString::from_raw(ptr)
        };
        cstring.to_string_lossy().fmt(f)
    }
}

/// View an Operator
///
/// ```
/// # use libceed::prelude::*;
/// # fn main() -> libceed::Result<()> {
/// # let ceed = libceed::Ceed::default_init();
/// let qf = ceed.q_function_interior_by_name("Mass1DBuild")?;
///
/// // Operator field arguments
/// let ne = 3;
/// let q = 4 as usize;
/// let mut ind: Vec<i32> = vec![0; 2 * ne];
/// for i in 0..ne {
///     ind[2 * i + 0] = i as i32;
///     ind[2 * i + 1] = (i + 1) as i32;
/// }
/// let r = ceed.elem_restriction(ne, 2, 1, 1, ne + 1, MemType::Host, &ind)?;
/// let strides: [i32; 3] = [1, q as i32, q as i32];
/// let rq = ceed.strided_elem_restriction(ne, 2, 1, q * ne, strides)?;
///
/// let b = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, QuadMode::Gauss)?;
///
/// // Operator fields
/// let op = ceed
///     .operator(&qf, QFunctionOpt::None, QFunctionOpt::None)?
///     .field("dx", &r, &b, VectorOpt::Active)?
///     .field("weights", ElemRestrictionOpt::None, &b, VectorOpt::None)?
///     .field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active)?;
///
/// println!("{}", op);
/// # Ok(())
/// # }
/// ```
impl<'a> fmt::Display for Operator<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.op_core.fmt(f)
    }
}

/// View a composite Operator
///
/// ```
/// # use libceed::prelude::*;
/// # fn main() -> libceed::Result<()> {
/// # let ceed = libceed::Ceed::default_init();
///
/// // Sub operator field arguments
/// let ne = 3;
/// let q = 4 as usize;
/// let mut ind: Vec<i32> = vec![0; 2 * ne];
/// for i in 0..ne {
///     ind[2 * i + 0] = i as i32;
///     ind[2 * i + 1] = (i + 1) as i32;
/// }
/// let r = ceed.elem_restriction(ne, 2, 1, 1, ne + 1, MemType::Host, &ind)?;
/// let strides: [i32; 3] = [1, q as i32, q as i32];
/// let rq = ceed.strided_elem_restriction(ne, 2, 1, q * ne, strides)?;
///
/// let b = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, QuadMode::Gauss)?;
///
/// let qdata_mass = ceed.vector(q * ne)?;
/// let qdata_diff = ceed.vector(q * ne)?;
///
/// let qf_mass = ceed.q_function_interior_by_name("MassApply")?;
/// let op_mass = ceed
///     .operator(&qf_mass, QFunctionOpt::None, QFunctionOpt::None)?
///     .field("u", &r, &b, VectorOpt::Active)?
///     .field("qdata", &rq, BasisOpt::Collocated, &qdata_mass)?
///     .field("v", &r, &b, VectorOpt::Active)?;
///
/// let qf_diff = ceed.q_function_interior_by_name("Poisson1DApply")?;
/// let op_diff = ceed
///     .operator(&qf_diff, QFunctionOpt::None, QFunctionOpt::None)?
///     .field("du", &r, &b, VectorOpt::Active)?
///     .field("qdata", &rq, BasisOpt::Collocated, &qdata_diff)?
///     .field("dv", &r, &b, VectorOpt::Active)?;
///
/// let op = ceed
///     .composite_operator()?
///     .sub_operator(&op_mass)?
///     .sub_operator(&op_diff)?;
///
/// println!("{}", op);
/// # Ok(())
/// # }
/// ```
impl<'a> fmt::Display for CompositeOperator<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.op_core.fmt(f)
    }
}

// -----------------------------------------------------------------------------
// Core functionality
// -----------------------------------------------------------------------------
impl<'a> OperatorCore<'a> {
    // Error handling
    #[doc(hidden)]
    fn check_error(&self, ierr: i32) -> crate::Result<i32> {
        let mut ptr = std::ptr::null_mut();
        unsafe {
            bind_ceed::CeedOperatorGetCeed(self.ptr, &mut ptr);
        }
        crate::check_error(ptr, ierr)
    }

    // Common implementations
    pub fn apply(&self, input: &Vector, output: &mut Vector) -> crate::Result<i32> {
        let ierr = unsafe {
            bind_ceed::CeedOperatorApply(
                self.ptr,
                input.ptr,
                output.ptr,
                bind_ceed::CEED_REQUEST_IMMEDIATE,
            )
        };
        self.check_error(ierr)
    }

    pub fn apply_add(&self, input: &Vector, output: &mut Vector) -> crate::Result<i32> {
        let ierr = unsafe {
            bind_ceed::CeedOperatorApplyAdd(
                self.ptr,
                input.ptr,
                output.ptr,
                bind_ceed::CEED_REQUEST_IMMEDIATE,
            )
        };
        self.check_error(ierr)
    }

    pub fn linear_assemble_diagonal(&self, assembled: &mut Vector) -> crate::Result<i32> {
        let ierr = unsafe {
            bind_ceed::CeedOperatorLinearAssembleDiagonal(
                self.ptr,
                assembled.ptr,
                bind_ceed::CEED_REQUEST_IMMEDIATE,
            )
        };
        self.check_error(ierr)
    }

    pub fn linear_assemble_add_diagonal(&self, assembled: &mut Vector) -> crate::Result<i32> {
        let ierr = unsafe {
            bind_ceed::CeedOperatorLinearAssembleAddDiagonal(
                self.ptr,
                assembled.ptr,
                bind_ceed::CEED_REQUEST_IMMEDIATE,
            )
        };
        self.check_error(ierr)
    }

    pub fn linear_assemble_point_block_diagonal(
        &self,
        assembled: &mut Vector,
    ) -> crate::Result<i32> {
        let ierr = unsafe {
            bind_ceed::CeedOperatorLinearAssemblePointBlockDiagonal(
                self.ptr,
                assembled.ptr,
                bind_ceed::CEED_REQUEST_IMMEDIATE,
            )
        };
        self.check_error(ierr)
    }

    pub fn linear_assemble_add_point_block_diagonal(
        &self,
        assembled: &mut Vector,
    ) -> crate::Result<i32> {
        let ierr = unsafe {
            bind_ceed::CeedOperatorLinearAssembleAddPointBlockDiagonal(
                self.ptr,
                assembled.ptr,
                bind_ceed::CEED_REQUEST_IMMEDIATE,
            )
        };
        self.check_error(ierr)
    }
}

// -----------------------------------------------------------------------------
// Operator
// -----------------------------------------------------------------------------
impl<'a> Operator<'a> {
    // Constructor
    pub fn create<'b>(
        ceed: &'a crate::Ceed,
        qf: impl Into<QFunctionOpt<'b>>,
        dqf: impl Into<QFunctionOpt<'b>>,
        dqfT: impl Into<QFunctionOpt<'b>>,
    ) -> crate::Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let ierr = unsafe {
            bind_ceed::CeedOperatorCreate(
                ceed.ptr,
                qf.into().to_raw(),
                dqf.into().to_raw(),
                dqfT.into().to_raw(),
                &mut ptr,
            )
        };
        ceed.check_error(ierr)?;
        Ok(Self {
            op_core: OperatorCore {
                ptr,
                _lifeline: PhantomData,
            },
        })
    }

    fn from_raw(ptr: bind_ceed::CeedOperator) -> crate::Result<Self> {
        Ok(Self {
            op_core: OperatorCore {
                ptr,
                _lifeline: PhantomData,
            },
        })
    }

    /// Apply Operator to a vector
    ///
    /// * `input`  - Input Vector
    /// * `output` - Output Vector
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let ne = 4;
    /// let p = 3;
    /// let q = 4;
    /// let ndofs = p * ne - ne + 1;
    ///
    /// // Vectors
    /// let x = ceed.vector_from_slice(&[-1., -0.5, 0.0, 0.5, 1.0])?;
    /// let mut qdata = ceed.vector(ne * q)?;
    /// qdata.set_value(0.0);
    /// let u = ceed.vector_from_slice(&vec![1.0; ndofs])?;
    /// let mut v = ceed.vector(ndofs)?;
    /// v.set_value(0.0);
    ///
    /// // Restrictions
    /// let mut indx: Vec<i32> = vec![0; 2 * ne];
    /// for i in 0..ne {
    ///     indx[2 * i + 0] = i as i32;
    ///     indx[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let rx = ceed.elem_restriction(ne, 2, 1, 1, ne + 1, MemType::Host, &indx)?;
    /// let mut indu: Vec<i32> = vec![0; p * ne];
    /// for i in 0..ne {
    ///     indu[p * i + 0] = i as i32;
    ///     indu[p * i + 1] = (i + 1) as i32;
    ///     indu[p * i + 2] = (i + 2) as i32;
    /// }
    /// let ru = ceed.elem_restriction(ne, 3, 1, 1, ndofs, MemType::Host, &indu)?;
    /// let strides: [i32; 3] = [1, q as i32, q as i32];
    /// let rq = ceed.strided_elem_restriction(ne, q, 1, q * ne, strides)?;
    ///
    /// // Bases
    /// let bx = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, QuadMode::Gauss)?;
    /// let bu = ceed.basis_tensor_H1_Lagrange(1, 1, p, q, QuadMode::Gauss)?;
    ///
    /// // Build quadrature data
    /// let qf_build = ceed.q_function_interior_by_name("Mass1DBuild")?;
    /// ceed.operator(&qf_build, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("dx", &rx, &bx, VectorOpt::Active)?
    ///     .field("weights", ElemRestrictionOpt::None, &bx, VectorOpt::None)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active)?
    ///     .apply(&x, &mut qdata)?;
    ///
    /// // Mass operator
    /// let qf_mass = ceed.q_function_interior_by_name("MassApply")?;
    /// let op_mass = ceed
    ///     .operator(&qf_mass, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("u", &ru, &bu, VectorOpt::Active)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, &qdata)?
    ///     .field("v", &ru, &bu, VectorOpt::Active)?;
    ///
    /// v.set_value(0.0);
    /// op_mass.apply(&u, &mut v)?;
    ///
    /// // Check
    /// let sum: Scalar = v.view()?.iter().sum();
    /// assert!(
    ///     (sum - 2.0).abs() < 10.0 * libceed::EPSILON,
    ///     "Incorrect interval length computed"
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn apply(&self, input: &Vector, output: &mut Vector) -> crate::Result<i32> {
        self.op_core.apply(input, output)
    }

    /// Apply Operator to a vector and add result to output vector
    ///
    /// * `input`  - Input Vector
    /// * `output` - Output Vector
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let ne = 4;
    /// let p = 3;
    /// let q = 4;
    /// let ndofs = p * ne - ne + 1;
    ///
    /// // Vectors
    /// let x = ceed.vector_from_slice(&[-1., -0.5, 0.0, 0.5, 1.0])?;
    /// let mut qdata = ceed.vector(ne * q)?;
    /// qdata.set_value(0.0);
    /// let u = ceed.vector_from_slice(&vec![1.0; ndofs])?;
    /// let mut v = ceed.vector(ndofs)?;
    ///
    /// // Restrictions
    /// let mut indx: Vec<i32> = vec![0; 2 * ne];
    /// for i in 0..ne {
    ///     indx[2 * i + 0] = i as i32;
    ///     indx[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let rx = ceed.elem_restriction(ne, 2, 1, 1, ne + 1, MemType::Host, &indx)?;
    /// let mut indu: Vec<i32> = vec![0; p * ne];
    /// for i in 0..ne {
    ///     indu[p * i + 0] = i as i32;
    ///     indu[p * i + 1] = (i + 1) as i32;
    ///     indu[p * i + 2] = (i + 2) as i32;
    /// }
    /// let ru = ceed.elem_restriction(ne, 3, 1, 1, ndofs, MemType::Host, &indu)?;
    /// let strides: [i32; 3] = [1, q as i32, q as i32];
    /// let rq = ceed.strided_elem_restriction(ne, q, 1, q * ne, strides)?;
    ///
    /// // Bases
    /// let bx = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, QuadMode::Gauss)?;
    /// let bu = ceed.basis_tensor_H1_Lagrange(1, 1, p, q, QuadMode::Gauss)?;
    ///
    /// // Build quadrature data
    /// let qf_build = ceed.q_function_interior_by_name("Mass1DBuild")?;
    /// ceed.operator(&qf_build, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("dx", &rx, &bx, VectorOpt::Active)?
    ///     .field("weights", ElemRestrictionOpt::None, &bx, VectorOpt::None)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active)?
    ///     .apply(&x, &mut qdata)?;
    ///
    /// // Mass operator
    /// let qf_mass = ceed.q_function_interior_by_name("MassApply")?;
    /// let op_mass = ceed
    ///     .operator(&qf_mass, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("u", &ru, &bu, VectorOpt::Active)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, &qdata)?
    ///     .field("v", &ru, &bu, VectorOpt::Active)?;
    ///
    /// v.set_value(1.0);
    /// op_mass.apply_add(&u, &mut v)?;
    ///
    /// // Check
    /// let sum: Scalar = v.view()?.iter().sum();
    /// assert!(
    ///     (sum - (2.0 + ndofs as Scalar)).abs() < 10.0 * libceed::EPSILON,
    ///     "Incorrect interval length computed and added"
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn apply_add(&self, input: &Vector, output: &mut Vector) -> crate::Result<i32> {
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
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let qf = ceed.q_function_interior_by_name("Mass1DBuild")?;
    /// let mut op = ceed.operator(&qf, QFunctionOpt::None, QFunctionOpt::None)?;
    ///
    /// // Operator field arguments
    /// let ne = 3;
    /// let q = 4;
    /// let mut ind: Vec<i32> = vec![0; 2 * ne];
    /// for i in 0..ne {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed.elem_restriction(ne, 2, 1, 1, ne + 1, MemType::Host, &ind)?;
    ///
    /// let b = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, QuadMode::Gauss)?;
    ///
    /// // Operator field
    /// op = op.field("dx", &r, &b, VectorOpt::Active)?;
    /// # Ok(())
    /// # }
    /// ```
    #[allow(unused_mut)]
    pub fn field<'b>(
        mut self,
        fieldname: &str,
        r: impl Into<ElemRestrictionOpt<'b>>,
        b: impl Into<BasisOpt<'b>>,
        v: impl Into<VectorOpt<'b>>,
    ) -> crate::Result<Self> {
        let fieldname = CString::new(fieldname).expect("CString::new failed");
        let fieldname = fieldname.as_ptr() as *const i8;
        let ierr = unsafe {
            bind_ceed::CeedOperatorSetField(
                self.op_core.ptr,
                fieldname,
                r.into().to_raw(),
                b.into().to_raw(),
                v.into().to_raw(),
            )
        };
        self.op_core.check_error(ierr)?;
        Ok(self)
    }

    /// Get a slice of Operator inputs
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let qf = ceed.q_function_interior_by_name("Mass1DBuild")?;
    ///
    /// // Operator field arguments
    /// let ne = 3;
    /// let q = 4 as usize;
    /// let mut ind: Vec<i32> = vec![0; 2 * ne];
    /// for i in 0..ne {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed.elem_restriction(ne, 2, 1, 1, ne + 1, MemType::Host, &ind)?;
    /// let strides: [i32; 3] = [1, q as i32, q as i32];
    /// let rq = ceed.strided_elem_restriction(ne, 2, 1, q * ne, strides)?;
    ///
    /// let b = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, QuadMode::Gauss)?;
    ///
    /// // Operator fields
    /// let op = ceed
    ///     .operator(&qf, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("dx", &r, &b, VectorOpt::Active)?
    ///     .field("weights", ElemRestrictionOpt::None, &b, VectorOpt::None)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active)?;
    ///
    /// let inputs = op.inputs()?;
    ///
    /// assert_eq!(inputs.len(), 2, "Incorrect inputs array");
    /// # Ok(())
    /// # }
    /// ```
    pub fn inputs(&self) -> crate::Result<&[crate::OperatorField]> {
        // Get array of raw C pointers for inputs
        let mut num_inputs = 0;
        let mut inputs_ptr = std::ptr::null_mut();
        let ierr = unsafe {
            bind_ceed::CeedOperatorGetFields(
                self.op_core.ptr,
                &mut num_inputs,
                &mut inputs_ptr,
                std::ptr::null_mut() as *mut bind_ceed::CeedInt,
                std::ptr::null_mut() as *mut *mut bind_ceed::CeedOperatorField,
            )
        };
        self.op_core.check_error(ierr)?;
        // Convert raw C pointers to fixed length slice
        let inputs_slice = unsafe {
            std::slice::from_raw_parts(
                inputs_ptr as *const crate::OperatorField,
                num_inputs as usize,
            )
        };
        Ok(inputs_slice)
    }

    /// Get a slice of Operator outputs
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let qf = ceed.q_function_interior_by_name("Mass1DBuild")?;
    ///
    /// // Operator field arguments
    /// let ne = 3;
    /// let q = 4 as usize;
    /// let mut ind: Vec<i32> = vec![0; 2 * ne];
    /// for i in 0..ne {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed.elem_restriction(ne, 2, 1, 1, ne + 1, MemType::Host, &ind)?;
    /// let strides: [i32; 3] = [1, q as i32, q as i32];
    /// let rq = ceed.strided_elem_restriction(ne, 2, 1, q * ne, strides)?;
    ///
    /// let b = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, QuadMode::Gauss)?;
    ///
    /// // Operator fields
    /// let op = ceed
    ///     .operator(&qf, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("dx", &r, &b, VectorOpt::Active)?
    ///     .field("weights", ElemRestrictionOpt::None, &b, VectorOpt::None)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active)?;
    ///
    /// let outputs = op.outputs()?;
    ///
    /// assert_eq!(outputs.len(), 1, "Incorrect outputs array");
    /// # Ok(())
    /// # }
    /// ```
    pub fn outputs(&self) -> crate::Result<&[crate::OperatorField]> {
        // Get array of raw C pointers for outputs
        let mut num_outputs = 0;
        let mut outputs_ptr = std::ptr::null_mut();
        let ierr = unsafe {
            bind_ceed::CeedOperatorGetFields(
                self.op_core.ptr,
                std::ptr::null_mut() as *mut bind_ceed::CeedInt,
                std::ptr::null_mut() as *mut *mut bind_ceed::CeedOperatorField,
                &mut num_outputs,
                &mut outputs_ptr,
            )
        };
        self.op_core.check_error(ierr)?;
        // Convert raw C pointers to fixed length slice
        let outputs_slice = unsafe {
            std::slice::from_raw_parts(
                outputs_ptr as *const crate::OperatorField,
                num_outputs as usize,
            )
        };
        Ok(outputs_slice)
    }

    /// Get the number of elements associated with an Operator
    ///
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let qf = ceed.q_function_interior_by_name("Mass1DBuild")?;
    /// let mut op = ceed.operator(&qf, QFunctionOpt::None, QFunctionOpt::None)?;
    ///
    /// // Operator field arguments
    /// let ne = 3;
    /// let q = 4;
    /// let mut ind: Vec<i32> = vec![0; 2 * ne];
    /// for i in 0..ne {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed.elem_restriction(ne, 2, 1, 1, ne + 1, MemType::Host, &ind)?;
    ///
    /// let b = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, QuadMode::Gauss)?;
    ///
    /// // Operator field
    /// op = op.field("dx", &r, &b, VectorOpt::Active)?;
    ///
    /// // Check number of elements
    /// assert_eq!(op.num_elements(), ne, "Incorrect number of elements");
    /// # Ok(())
    /// # }
    /// ```
    pub fn num_elements(&self) -> usize {
        let mut nelem = 0;
        unsafe { bind_ceed::CeedOperatorGetNumElements(self.op_core.ptr, &mut nelem) };
        usize::try_from(nelem).unwrap()
    }

    /// Get the number of quadrature points associated with each element of
    ///   an Operator
    ///
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let qf = ceed.q_function_interior_by_name("Mass1DBuild")?;
    /// let mut op = ceed.operator(&qf, QFunctionOpt::None, QFunctionOpt::None)?;
    ///
    /// // Operator field arguments
    /// let ne = 3;
    /// let q = 4;
    /// let mut ind: Vec<i32> = vec![0; 2 * ne];
    /// for i in 0..ne {
    ///     ind[2 * i + 0] = i as i32;
    ///     ind[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let r = ceed.elem_restriction(ne, 2, 1, 1, ne + 1, MemType::Host, &ind)?;
    ///
    /// let b = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, QuadMode::Gauss)?;
    ///
    /// // Operator field
    /// op = op.field("dx", &r, &b, VectorOpt::Active)?;
    ///
    /// // Check number of quadrature points
    /// assert_eq!(
    ///     op.num_quadrature_points(),
    ///     q,
    ///     "Incorrect number of quadrature points"
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn num_quadrature_points(&self) -> usize {
        let mut Q = 0;
        unsafe { bind_ceed::CeedOperatorGetNumQuadraturePoints(self.op_core.ptr, &mut Q) };
        usize::try_from(Q).unwrap()
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
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let ne = 4;
    /// let p = 3;
    /// let q = 4;
    /// let ndofs = p * ne - ne + 1;
    ///
    /// // Vectors
    /// let x = ceed.vector_from_slice(&[-1., -0.5, 0.0, 0.5, 1.0])?;
    /// let mut qdata = ceed.vector(ne * q)?;
    /// qdata.set_value(0.0);
    /// let mut u = ceed.vector(ndofs)?;
    /// u.set_value(1.0);
    /// let mut v = ceed.vector(ndofs)?;
    /// v.set_value(0.0);
    ///
    /// // Restrictions
    /// let mut indx: Vec<i32> = vec![0; 2 * ne];
    /// for i in 0..ne {
    ///     indx[2 * i + 0] = i as i32;
    ///     indx[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let rx = ceed.elem_restriction(ne, 2, 1, 1, ne + 1, MemType::Host, &indx)?;
    /// let mut indu: Vec<i32> = vec![0; p * ne];
    /// for i in 0..ne {
    ///     indu[p * i + 0] = (2 * i) as i32;
    ///     indu[p * i + 1] = (2 * i + 1) as i32;
    ///     indu[p * i + 2] = (2 * i + 2) as i32;
    /// }
    /// let ru = ceed.elem_restriction(ne, p, 1, 1, ndofs, MemType::Host, &indu)?;
    /// let strides: [i32; 3] = [1, q as i32, q as i32];
    /// let rq = ceed.strided_elem_restriction(ne, q, 1, q * ne, strides)?;
    ///
    /// // Bases
    /// let bx = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, QuadMode::Gauss)?;
    /// let bu = ceed.basis_tensor_H1_Lagrange(1, 1, p, q, QuadMode::Gauss)?;
    ///
    /// // Build quadrature data
    /// let qf_build = ceed.q_function_interior_by_name("Mass1DBuild")?;
    /// ceed.operator(&qf_build, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("dx", &rx, &bx, VectorOpt::Active)?
    ///     .field("weights", ElemRestrictionOpt::None, &bx, VectorOpt::None)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active)?
    ///     .apply(&x, &mut qdata)?;
    ///
    /// // Mass operator
    /// let qf_mass = ceed.q_function_interior_by_name("MassApply")?;
    /// let op_mass = ceed
    ///     .operator(&qf_mass, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("u", &ru, &bu, VectorOpt::Active)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, &qdata)?
    ///     .field("v", &ru, &bu, VectorOpt::Active)?;
    ///
    /// // Diagonal
    /// let mut diag = ceed.vector(ndofs)?;
    /// diag.set_value(0.0);
    /// op_mass.linear_assemble_diagonal(&mut diag)?;
    ///
    /// // Manual diagonal computation
    /// let mut true_diag = ceed.vector(ndofs)?;
    /// for i in 0..ndofs {
    ///     u.set_value(0.0);
    ///     {
    ///         let mut u_array = u.view_mut()?;
    ///         u_array[i] = 1.;
    ///     }
    ///
    ///     op_mass.apply(&u, &mut v)?;
    ///
    ///     {
    ///         let v_array = v.view_mut()?;
    ///         let mut true_array = true_diag.view_mut()?;
    ///         true_array[i] = v_array[i];
    ///     }
    /// }
    ///
    /// // Check
    /// diag.view()?
    ///     .iter()
    ///     .zip(true_diag.view()?.iter())
    ///     .for_each(|(computed, actual)| {
    ///         assert!(
    ///             (*computed - *actual).abs() < 10.0 * libceed::EPSILON,
    ///             "Diagonal entry incorrect"
    ///         );
    ///     });
    /// # Ok(())
    /// # }
    /// ```
    pub fn linear_assemble_diagonal(&self, assembled: &mut Vector) -> crate::Result<i32> {
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
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let ne = 4;
    /// let p = 3;
    /// let q = 4;
    /// let ndofs = p * ne - ne + 1;
    ///
    /// // Vectors
    /// let x = ceed.vector_from_slice(&[-1., -0.5, 0.0, 0.5, 1.0])?;
    /// let mut qdata = ceed.vector(ne * q)?;
    /// qdata.set_value(0.0);
    /// let mut u = ceed.vector(ndofs)?;
    /// u.set_value(1.0);
    /// let mut v = ceed.vector(ndofs)?;
    /// v.set_value(0.0);
    ///
    /// // Restrictions
    /// let mut indx: Vec<i32> = vec![0; 2 * ne];
    /// for i in 0..ne {
    ///     indx[2 * i + 0] = i as i32;
    ///     indx[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let rx = ceed.elem_restriction(ne, 2, 1, 1, ne + 1, MemType::Host, &indx)?;
    /// let mut indu: Vec<i32> = vec![0; p * ne];
    /// for i in 0..ne {
    ///     indu[p * i + 0] = (2 * i) as i32;
    ///     indu[p * i + 1] = (2 * i + 1) as i32;
    ///     indu[p * i + 2] = (2 * i + 2) as i32;
    /// }
    /// let ru = ceed.elem_restriction(ne, p, 1, 1, ndofs, MemType::Host, &indu)?;
    /// let strides: [i32; 3] = [1, q as i32, q as i32];
    /// let rq = ceed.strided_elem_restriction(ne, q, 1, q * ne, strides)?;
    ///
    /// // Bases
    /// let bx = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, QuadMode::Gauss)?;
    /// let bu = ceed.basis_tensor_H1_Lagrange(1, 1, p, q, QuadMode::Gauss)?;
    ///
    /// // Build quadrature data
    /// let qf_build = ceed.q_function_interior_by_name("Mass1DBuild")?;
    /// ceed.operator(&qf_build, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("dx", &rx, &bx, VectorOpt::Active)?
    ///     .field("weights", ElemRestrictionOpt::None, &bx, VectorOpt::None)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active)?
    ///     .apply(&x, &mut qdata)?;
    ///
    /// // Mass operator
    /// let qf_mass = ceed.q_function_interior_by_name("MassApply")?;
    /// let op_mass = ceed
    ///     .operator(&qf_mass, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("u", &ru, &bu, VectorOpt::Active)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, &qdata)?
    ///     .field("v", &ru, &bu, VectorOpt::Active)?;
    ///
    /// // Diagonal
    /// let mut diag = ceed.vector(ndofs)?;
    /// diag.set_value(1.0);
    /// op_mass.linear_assemble_add_diagonal(&mut diag)?;
    ///
    /// // Manual diagonal computation
    /// let mut true_diag = ceed.vector(ndofs)?;
    /// for i in 0..ndofs {
    ///     u.set_value(0.0);
    ///     {
    ///         let mut u_array = u.view_mut()?;
    ///         u_array[i] = 1.;
    ///     }
    ///
    ///     op_mass.apply(&u, &mut v)?;
    ///
    ///     {
    ///         let v_array = v.view_mut()?;
    ///         let mut true_array = true_diag.view_mut()?;
    ///         true_array[i] = v_array[i] + 1.0;
    ///     }
    /// }
    ///
    /// // Check
    /// diag.view()?
    ///     .iter()
    ///     .zip(true_diag.view()?.iter())
    ///     .for_each(|(computed, actual)| {
    ///         assert!(
    ///             (*computed - *actual).abs() < 10.0 * libceed::EPSILON,
    ///             "Diagonal entry incorrect"
    ///         );
    ///     });
    /// # Ok(())
    /// # }
    /// ```
    pub fn linear_assemble_add_diagonal(&self, assembled: &mut Vector) -> crate::Result<i32> {
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
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let ne = 4;
    /// let p = 3;
    /// let q = 4;
    /// let ncomp = 2;
    /// let ndofs = p * ne - ne + 1;
    ///
    /// // Vectors
    /// let x = ceed.vector_from_slice(&[-1., -0.5, 0.0, 0.5, 1.0])?;
    /// let mut qdata = ceed.vector(ne * q)?;
    /// qdata.set_value(0.0);
    /// let mut u = ceed.vector(ncomp * ndofs)?;
    /// u.set_value(1.0);
    /// let mut v = ceed.vector(ncomp * ndofs)?;
    /// v.set_value(0.0);
    ///
    /// // Restrictions
    /// let mut indx: Vec<i32> = vec![0; 2 * ne];
    /// for i in 0..ne {
    ///     indx[2 * i + 0] = i as i32;
    ///     indx[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let rx = ceed.elem_restriction(ne, 2, 1, 1, ne + 1, MemType::Host, &indx)?;
    /// let mut indu: Vec<i32> = vec![0; p * ne];
    /// for i in 0..ne {
    ///     indu[p * i + 0] = (2 * i) as i32;
    ///     indu[p * i + 1] = (2 * i + 1) as i32;
    ///     indu[p * i + 2] = (2 * i + 2) as i32;
    /// }
    /// let ru = ceed.elem_restriction(ne, p, ncomp, ndofs, ncomp * ndofs, MemType::Host, &indu)?;
    /// let strides: [i32; 3] = [1, q as i32, q as i32];
    /// let rq = ceed.strided_elem_restriction(ne, q, 1, q * ne, strides)?;
    ///
    /// // Bases
    /// let bx = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, QuadMode::Gauss)?;
    /// let bu = ceed.basis_tensor_H1_Lagrange(1, ncomp, p, q, QuadMode::Gauss)?;
    ///
    /// // Build quadrature data
    /// let qf_build = ceed.q_function_interior_by_name("Mass1DBuild")?;
    /// ceed.operator(&qf_build, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("dx", &rx, &bx, VectorOpt::Active)?
    ///     .field("weights", ElemRestrictionOpt::None, &bx, VectorOpt::None)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active)?
    ///     .apply(&x, &mut qdata)?;
    ///
    /// // Mass operator
    /// let mut mass_2_comp = |[u, qdata, ..]: QFunctionInputs, [v, ..]: QFunctionOutputs| {
    ///     // Number of quadrature points
    ///     let q = qdata.len();
    ///
    ///     // Iterate over quadrature points
    ///     for i in 0..q {
    ///         v[i + 0 * q] = u[i + 1 * q] * qdata[i];
    ///         v[i + 1 * q] = u[i + 0 * q] * qdata[i];
    ///     }
    ///
    ///     // Return clean error code
    ///     0
    /// };
    ///
    /// let qf_mass = ceed
    ///     .q_function_interior(1, Box::new(mass_2_comp))?
    ///     .input("u", 2, EvalMode::Interp)?
    ///     .input("qdata", 1, EvalMode::None)?
    ///     .output("v", 2, EvalMode::Interp)?;
    ///
    /// let op_mass = ceed
    ///     .operator(&qf_mass, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("u", &ru, &bu, VectorOpt::Active)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, &qdata)?
    ///     .field("v", &ru, &bu, VectorOpt::Active)?;
    ///
    /// // Diagonal
    /// let mut diag = ceed.vector(ncomp * ncomp * ndofs)?;
    /// diag.set_value(0.0);
    /// op_mass.linear_assemble_point_block_diagonal(&mut diag)?;
    ///
    /// // Manual diagonal computation
    /// let mut true_diag = ceed.vector(ncomp * ncomp * ndofs)?;
    /// for i in 0..ndofs {
    ///     for j in 0..ncomp {
    ///         u.set_value(0.0);
    ///         {
    ///             let mut u_array = u.view_mut()?;
    ///             u_array[i + j * ndofs] = 1.;
    ///         }
    ///
    ///         op_mass.apply(&u, &mut v)?;
    ///
    ///         {
    ///             let v_array = v.view_mut()?;
    ///             let mut true_array = true_diag.view_mut()?;
    ///             for k in 0..ncomp {
    ///                 true_array[i * ncomp * ncomp + k * ncomp + j] = v_array[i + k * ndofs];
    ///             }
    ///         }
    ///     }
    /// }
    ///
    /// // Check
    /// diag.view()?
    ///     .iter()
    ///     .zip(true_diag.view()?.iter())
    ///     .for_each(|(computed, actual)| {
    ///         assert!(
    ///             (*computed - *actual).abs() < 10.0 * libceed::EPSILON,
    ///             "Diagonal entry incorrect"
    ///         );
    ///     });
    /// # Ok(())
    /// # }
    /// ```
    pub fn linear_assemble_point_block_diagonal(
        &self,
        assembled: &mut Vector,
    ) -> crate::Result<i32> {
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
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let ne = 4;
    /// let p = 3;
    /// let q = 4;
    /// let ncomp = 2;
    /// let ndofs = p * ne - ne + 1;
    ///
    /// // Vectors
    /// let x = ceed.vector_from_slice(&[-1., -0.5, 0.0, 0.5, 1.0])?;
    /// let mut qdata = ceed.vector(ne * q)?;
    /// qdata.set_value(0.0);
    /// let mut u = ceed.vector(ncomp * ndofs)?;
    /// u.set_value(1.0);
    /// let mut v = ceed.vector(ncomp * ndofs)?;
    /// v.set_value(0.0);
    ///
    /// // Restrictions
    /// let mut indx: Vec<i32> = vec![0; 2 * ne];
    /// for i in 0..ne {
    ///     indx[2 * i + 0] = i as i32;
    ///     indx[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let rx = ceed.elem_restriction(ne, 2, 1, 1, ne + 1, MemType::Host, &indx)?;
    /// let mut indu: Vec<i32> = vec![0; p * ne];
    /// for i in 0..ne {
    ///     indu[p * i + 0] = (2 * i) as i32;
    ///     indu[p * i + 1] = (2 * i + 1) as i32;
    ///     indu[p * i + 2] = (2 * i + 2) as i32;
    /// }
    /// let ru = ceed.elem_restriction(ne, p, ncomp, ndofs, ncomp * ndofs, MemType::Host, &indu)?;
    /// let strides: [i32; 3] = [1, q as i32, q as i32];
    /// let rq = ceed.strided_elem_restriction(ne, q, 1, q * ne, strides)?;
    ///
    /// // Bases
    /// let bx = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, QuadMode::Gauss)?;
    /// let bu = ceed.basis_tensor_H1_Lagrange(1, ncomp, p, q, QuadMode::Gauss)?;
    ///
    /// // Build quadrature data
    /// let qf_build = ceed.q_function_interior_by_name("Mass1DBuild")?;
    /// ceed.operator(&qf_build, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("dx", &rx, &bx, VectorOpt::Active)?
    ///     .field("weights", ElemRestrictionOpt::None, &bx, VectorOpt::None)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active)?
    ///     .apply(&x, &mut qdata)?;
    ///
    /// // Mass operator
    /// let mut mass_2_comp = |[u, qdata, ..]: QFunctionInputs, [v, ..]: QFunctionOutputs| {
    ///     // Number of quadrature points
    ///     let q = qdata.len();
    ///
    ///     // Iterate over quadrature points
    ///     for i in 0..q {
    ///         v[i + 0 * q] = u[i + 1 * q] * qdata[i];
    ///         v[i + 1 * q] = u[i + 0 * q] * qdata[i];
    ///     }
    ///
    ///     // Return clean error code
    ///     0
    /// };
    ///
    /// let qf_mass = ceed
    ///     .q_function_interior(1, Box::new(mass_2_comp))?
    ///     .input("u", 2, EvalMode::Interp)?
    ///     .input("qdata", 1, EvalMode::None)?
    ///     .output("v", 2, EvalMode::Interp)?;
    ///
    /// let op_mass = ceed
    ///     .operator(&qf_mass, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("u", &ru, &bu, VectorOpt::Active)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, &qdata)?
    ///     .field("v", &ru, &bu, VectorOpt::Active)?;
    ///
    /// // Diagonal
    /// let mut diag = ceed.vector(ncomp * ncomp * ndofs)?;
    /// diag.set_value(1.0);
    /// op_mass.linear_assemble_add_point_block_diagonal(&mut diag)?;
    ///
    /// // Manual diagonal computation
    /// let mut true_diag = ceed.vector(ncomp * ncomp * ndofs)?;
    /// for i in 0..ndofs {
    ///     for j in 0..ncomp {
    ///         u.set_value(0.0);
    ///         {
    ///             let mut u_array = u.view_mut()?;
    ///             u_array[i + j * ndofs] = 1.;
    ///         }
    ///
    ///         op_mass.apply(&u, &mut v)?;
    ///
    ///         {
    ///             let v_array = v.view_mut()?;
    ///             let mut true_array = true_diag.view_mut()?;
    ///             for k in 0..ncomp {
    ///                 true_array[i * ncomp * ncomp + k * ncomp + j] = v_array[i + k * ndofs];
    ///             }
    ///         }
    ///     }
    /// }
    ///
    /// // Check
    /// diag.view()?
    ///     .iter()
    ///     .zip(true_diag.view()?.iter())
    ///     .for_each(|(computed, actual)| {
    ///         assert!(
    ///             (*computed - 1.0 - *actual).abs() < 10.0 * libceed::EPSILON,
    ///             "Diagonal entry incorrect"
    ///         );
    ///     });
    /// # Ok(())
    /// # }
    /// ```
    pub fn linear_assemble_add_point_block_diagonal(
        &self,
        assembled: &mut Vector,
    ) -> crate::Result<i32> {
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
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let ne = 15;
    /// let p_coarse = 3;
    /// let p_fine = 5;
    /// let q = 6;
    /// let ndofs_coarse = p_coarse * ne - ne + 1;
    /// let ndofs_fine = p_fine * ne - ne + 1;
    ///
    /// // Vectors
    /// let x_array = (0..ne + 1)
    ///     .map(|i| 2.0 * i as Scalar / ne as Scalar - 1.0)
    ///     .collect::<Vec<Scalar>>();
    /// let x = ceed.vector_from_slice(&x_array)?;
    /// let mut qdata = ceed.vector(ne * q)?;
    /// qdata.set_value(0.0);
    /// let mut u_coarse = ceed.vector(ndofs_coarse)?;
    /// u_coarse.set_value(1.0);
    /// let mut u_fine = ceed.vector(ndofs_fine)?;
    /// u_fine.set_value(1.0);
    /// let mut v_coarse = ceed.vector(ndofs_coarse)?;
    /// v_coarse.set_value(0.0);
    /// let mut v_fine = ceed.vector(ndofs_fine)?;
    /// v_fine.set_value(0.0);
    /// let mut multiplicity = ceed.vector(ndofs_fine)?;
    /// multiplicity.set_value(1.0);
    ///
    /// // Restrictions
    /// let strides: [i32; 3] = [1, q as i32, q as i32];
    /// let rq = ceed.strided_elem_restriction(ne, q, 1, q * ne, strides)?;
    ///
    /// let mut indx: Vec<i32> = vec![0; 2 * ne];
    /// for i in 0..ne {
    ///     indx[2 * i + 0] = i as i32;
    ///     indx[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let rx = ceed.elem_restriction(ne, 2, 1, 1, ne + 1, MemType::Host, &indx)?;
    ///
    /// let mut indu_coarse: Vec<i32> = vec![0; p_coarse * ne];
    /// for i in 0..ne {
    ///     for j in 0..p_coarse {
    ///         indu_coarse[p_coarse * i + j] = (i + j) as i32;
    ///     }
    /// }
    /// let ru_coarse = ceed.elem_restriction(
    ///     ne,
    ///     p_coarse,
    ///     1,
    ///     1,
    ///     ndofs_coarse,
    ///     MemType::Host,
    ///     &indu_coarse,
    /// )?;
    ///
    /// let mut indu_fine: Vec<i32> = vec![0; p_fine * ne];
    /// for i in 0..ne {
    ///     for j in 0..p_fine {
    ///         indu_fine[p_fine * i + j] = (i + j) as i32;
    ///     }
    /// }
    /// let ru_fine = ceed.elem_restriction(ne, p_fine, 1, 1, ndofs_fine, MemType::Host, &indu_fine)?;
    ///
    /// // Bases
    /// let bx = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, QuadMode::Gauss)?;
    /// let bu_coarse = ceed.basis_tensor_H1_Lagrange(1, 1, p_coarse, q, QuadMode::Gauss)?;
    /// let bu_fine = ceed.basis_tensor_H1_Lagrange(1, 1, p_fine, q, QuadMode::Gauss)?;
    ///
    /// // Build quadrature data
    /// let qf_build = ceed.q_function_interior_by_name("Mass1DBuild")?;
    /// ceed.operator(&qf_build, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("dx", &rx, &bx, VectorOpt::Active)?
    ///     .field("weights", ElemRestrictionOpt::None, &bx, VectorOpt::None)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active)?
    ///     .apply(&x, &mut qdata)?;
    ///
    /// // Mass operator
    /// let qf_mass = ceed.q_function_interior_by_name("MassApply")?;
    /// let op_mass_fine = ceed
    ///     .operator(&qf_mass, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("u", &ru_fine, &bu_fine, VectorOpt::Active)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, &qdata)?
    ///     .field("v", &ru_fine, &bu_fine, VectorOpt::Active)?;
    ///
    /// // Multigrid setup
    /// let (op_mass_coarse, op_prolong, op_restrict) =
    ///     op_mass_fine.create_multigrid_level(&multiplicity, &ru_coarse, &bu_coarse)?;
    ///
    /// // Coarse problem
    /// u_coarse.set_value(1.0);
    /// op_mass_coarse.apply(&u_coarse, &mut v_coarse)?;
    ///
    /// // Check
    /// let sum: Scalar = v_coarse.view()?.iter().sum();
    /// assert!(
    ///     (sum - 2.0).abs() < 50.0 * libceed::EPSILON,
    ///     "Incorrect interval length computed"
    /// );
    ///
    /// // Prolong
    /// op_prolong.apply(&u_coarse, &mut u_fine)?;
    ///
    /// // Fine problem
    /// op_mass_fine.apply(&u_fine, &mut v_fine)?;
    ///
    /// // Check
    /// let sum: Scalar = v_fine.view()?.iter().sum();
    /// assert!(
    ///     (sum - 2.0).abs() < 50.0 * libceed::EPSILON,
    ///     "Incorrect interval length computed"
    /// );
    ///
    /// // Restrict
    /// op_restrict.apply(&v_fine, &mut v_coarse)?;
    ///
    /// // Check
    /// let sum: Scalar = v_coarse.view()?.iter().sum();
    /// assert!(
    ///     (sum - 2.0).abs() < 50.0 * libceed::EPSILON,
    ///     "Incorrect interval length computed"
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn create_multigrid_level(
        &self,
        p_mult_fine: &Vector,
        rstr_coarse: &ElemRestriction,
        basis_coarse: &Basis,
    ) -> crate::Result<(Operator, Operator, Operator)> {
        let mut ptr_coarse = std::ptr::null_mut();
        let mut ptr_prolong = std::ptr::null_mut();
        let mut ptr_restrict = std::ptr::null_mut();
        let ierr = unsafe {
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
        self.op_core.check_error(ierr)?;
        let op_coarse = Operator::from_raw(ptr_coarse)?;
        let op_prolong = Operator::from_raw(ptr_prolong)?;
        let op_restrict = Operator::from_raw(ptr_restrict)?;
        Ok((op_coarse, op_prolong, op_restrict))
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
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let ne = 15;
    /// let p_coarse = 3;
    /// let p_fine = 5;
    /// let q = 6;
    /// let ndofs_coarse = p_coarse * ne - ne + 1;
    /// let ndofs_fine = p_fine * ne - ne + 1;
    ///
    /// // Vectors
    /// let x_array = (0..ne + 1)
    ///     .map(|i| 2.0 * i as Scalar / ne as Scalar - 1.0)
    ///     .collect::<Vec<Scalar>>();
    /// let x = ceed.vector_from_slice(&x_array)?;
    /// let mut qdata = ceed.vector(ne * q)?;
    /// qdata.set_value(0.0);
    /// let mut u_coarse = ceed.vector(ndofs_coarse)?;
    /// u_coarse.set_value(1.0);
    /// let mut u_fine = ceed.vector(ndofs_fine)?;
    /// u_fine.set_value(1.0);
    /// let mut v_coarse = ceed.vector(ndofs_coarse)?;
    /// v_coarse.set_value(0.0);
    /// let mut v_fine = ceed.vector(ndofs_fine)?;
    /// v_fine.set_value(0.0);
    /// let mut multiplicity = ceed.vector(ndofs_fine)?;
    /// multiplicity.set_value(1.0);
    ///
    /// // Restrictions
    /// let strides: [i32; 3] = [1, q as i32, q as i32];
    /// let rq = ceed.strided_elem_restriction(ne, q, 1, q * ne, strides)?;
    ///
    /// let mut indx: Vec<i32> = vec![0; 2 * ne];
    /// for i in 0..ne {
    ///     indx[2 * i + 0] = i as i32;
    ///     indx[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let rx = ceed.elem_restriction(ne, 2, 1, 1, ne + 1, MemType::Host, &indx)?;
    ///
    /// let mut indu_coarse: Vec<i32> = vec![0; p_coarse * ne];
    /// for i in 0..ne {
    ///     for j in 0..p_coarse {
    ///         indu_coarse[p_coarse * i + j] = (i + j) as i32;
    ///     }
    /// }
    /// let ru_coarse = ceed.elem_restriction(
    ///     ne,
    ///     p_coarse,
    ///     1,
    ///     1,
    ///     ndofs_coarse,
    ///     MemType::Host,
    ///     &indu_coarse,
    /// )?;
    ///
    /// let mut indu_fine: Vec<i32> = vec![0; p_fine * ne];
    /// for i in 0..ne {
    ///     for j in 0..p_fine {
    ///         indu_fine[p_fine * i + j] = (i + j) as i32;
    ///     }
    /// }
    /// let ru_fine = ceed.elem_restriction(ne, p_fine, 1, 1, ndofs_fine, MemType::Host, &indu_fine)?;
    ///
    /// // Bases
    /// let bx = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, QuadMode::Gauss)?;
    /// let bu_coarse = ceed.basis_tensor_H1_Lagrange(1, 1, p_coarse, q, QuadMode::Gauss)?;
    /// let bu_fine = ceed.basis_tensor_H1_Lagrange(1, 1, p_fine, q, QuadMode::Gauss)?;
    ///
    /// // Build quadrature data
    /// let qf_build = ceed.q_function_interior_by_name("Mass1DBuild")?;
    /// ceed.operator(&qf_build, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("dx", &rx, &bx, VectorOpt::Active)?
    ///     .field("weights", ElemRestrictionOpt::None, &bx, VectorOpt::None)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active)?
    ///     .apply(&x, &mut qdata)?;
    ///
    /// // Mass operator
    /// let qf_mass = ceed.q_function_interior_by_name("MassApply")?;
    /// let op_mass_fine = ceed
    ///     .operator(&qf_mass, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("u", &ru_fine, &bu_fine, VectorOpt::Active)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, &qdata)?
    ///     .field("v", &ru_fine, &bu_fine, VectorOpt::Active)?;
    ///
    /// // Multigrid setup
    /// let mut interp_c_to_f: Vec<Scalar> = vec![0.; p_coarse * p_fine];
    /// {
    ///     let mut coarse = ceed.vector(p_coarse)?;
    ///     let mut fine = ceed.vector(p_fine)?;
    ///     let basis_c_to_f =
    ///         ceed.basis_tensor_H1_Lagrange(1, 1, p_coarse, p_fine, QuadMode::GaussLobatto)?;
    ///     for i in 0..p_coarse {
    ///         coarse.set_value(0.0);
    ///         {
    ///             let mut array = coarse.view_mut()?;
    ///             array[i] = 1.;
    ///         }
    ///         basis_c_to_f.apply(
    ///             1,
    ///             TransposeMode::NoTranspose,
    ///             EvalMode::Interp,
    ///             &coarse,
    ///             &mut fine,
    ///         )?;
    ///         let array = fine.view()?;
    ///         for j in 0..p_fine {
    ///             interp_c_to_f[j * p_coarse + i] = array[j];
    ///         }
    ///     }
    /// }
    /// let (op_mass_coarse, op_prolong, op_restrict) = op_mass_fine.create_multigrid_level_tensor_H1(
    ///     &multiplicity,
    ///     &ru_coarse,
    ///     &bu_coarse,
    ///     &interp_c_to_f,
    /// )?;
    ///
    /// // Coarse problem
    /// u_coarse.set_value(1.0);
    /// op_mass_coarse.apply(&u_coarse, &mut v_coarse)?;
    ///
    /// // Check
    /// let sum: Scalar = v_coarse.view()?.iter().sum();
    /// assert!(
    ///     (sum - 2.0).abs() < 10.0 * libceed::EPSILON,
    ///     "Incorrect interval length computed"
    /// );
    ///
    /// // Prolong
    /// op_prolong.apply(&u_coarse, &mut u_fine)?;
    ///
    /// // Fine problem
    /// op_mass_fine.apply(&u_fine, &mut v_fine)?;
    ///
    /// // Check
    /// let sum: Scalar = v_fine.view()?.iter().sum();
    /// assert!(
    ///     (sum - 2.0).abs() < 10.0 * libceed::EPSILON,
    ///     "Incorrect interval length computed"
    /// );
    ///
    /// // Restrict
    /// op_restrict.apply(&v_fine, &mut v_coarse)?;
    ///
    /// // Check
    /// let sum: Scalar = v_coarse.view()?.iter().sum();
    /// assert!(
    ///     (sum - 2.0).abs() < 10.0 * libceed::EPSILON,
    ///     "Incorrect interval length computed"
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn create_multigrid_level_tensor_H1(
        &self,
        p_mult_fine: &Vector,
        rstr_coarse: &ElemRestriction,
        basis_coarse: &Basis,
        interpCtoF: &Vec<Scalar>,
    ) -> crate::Result<(Operator, Operator, Operator)> {
        let mut ptr_coarse = std::ptr::null_mut();
        let mut ptr_prolong = std::ptr::null_mut();
        let mut ptr_restrict = std::ptr::null_mut();
        let ierr = unsafe {
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
        self.op_core.check_error(ierr)?;
        let op_coarse = Operator::from_raw(ptr_coarse)?;
        let op_prolong = Operator::from_raw(ptr_prolong)?;
        let op_restrict = Operator::from_raw(ptr_restrict)?;
        Ok((op_coarse, op_prolong, op_restrict))
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
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let ne = 15;
    /// let p_coarse = 3;
    /// let p_fine = 5;
    /// let q = 6;
    /// let ndofs_coarse = p_coarse * ne - ne + 1;
    /// let ndofs_fine = p_fine * ne - ne + 1;
    ///
    /// // Vectors
    /// let x_array = (0..ne + 1)
    ///     .map(|i| 2.0 * i as Scalar / ne as Scalar - 1.0)
    ///     .collect::<Vec<Scalar>>();
    /// let x = ceed.vector_from_slice(&x_array)?;
    /// let mut qdata = ceed.vector(ne * q)?;
    /// qdata.set_value(0.0);
    /// let mut u_coarse = ceed.vector(ndofs_coarse)?;
    /// u_coarse.set_value(1.0);
    /// let mut u_fine = ceed.vector(ndofs_fine)?;
    /// u_fine.set_value(1.0);
    /// let mut v_coarse = ceed.vector(ndofs_coarse)?;
    /// v_coarse.set_value(0.0);
    /// let mut v_fine = ceed.vector(ndofs_fine)?;
    /// v_fine.set_value(0.0);
    /// let mut multiplicity = ceed.vector(ndofs_fine)?;
    /// multiplicity.set_value(1.0);
    ///
    /// // Restrictions
    /// let strides: [i32; 3] = [1, q as i32, q as i32];
    /// let rq = ceed.strided_elem_restriction(ne, q, 1, q * ne, strides)?;
    ///
    /// let mut indx: Vec<i32> = vec![0; 2 * ne];
    /// for i in 0..ne {
    ///     indx[2 * i + 0] = i as i32;
    ///     indx[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let rx = ceed.elem_restriction(ne, 2, 1, 1, ne + 1, MemType::Host, &indx)?;
    ///
    /// let mut indu_coarse: Vec<i32> = vec![0; p_coarse * ne];
    /// for i in 0..ne {
    ///     for j in 0..p_coarse {
    ///         indu_coarse[p_coarse * i + j] = (i + j) as i32;
    ///     }
    /// }
    /// let ru_coarse = ceed.elem_restriction(
    ///     ne,
    ///     p_coarse,
    ///     1,
    ///     1,
    ///     ndofs_coarse,
    ///     MemType::Host,
    ///     &indu_coarse,
    /// )?;
    ///
    /// let mut indu_fine: Vec<i32> = vec![0; p_fine * ne];
    /// for i in 0..ne {
    ///     for j in 0..p_fine {
    ///         indu_fine[p_fine * i + j] = (i + j) as i32;
    ///     }
    /// }
    /// let ru_fine = ceed.elem_restriction(ne, p_fine, 1, 1, ndofs_fine, MemType::Host, &indu_fine)?;
    ///
    /// // Bases
    /// let bx = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, QuadMode::Gauss)?;
    /// let bu_coarse = ceed.basis_tensor_H1_Lagrange(1, 1, p_coarse, q, QuadMode::Gauss)?;
    /// let bu_fine = ceed.basis_tensor_H1_Lagrange(1, 1, p_fine, q, QuadMode::Gauss)?;
    ///
    /// // Build quadrature data
    /// let qf_build = ceed.q_function_interior_by_name("Mass1DBuild")?;
    /// ceed.operator(&qf_build, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("dx", &rx, &bx, VectorOpt::Active)?
    ///     .field("weights", ElemRestrictionOpt::None, &bx, VectorOpt::None)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active)?
    ///     .apply(&x, &mut qdata)?;
    ///
    /// // Mass operator
    /// let qf_mass = ceed.q_function_interior_by_name("MassApply")?;
    /// let op_mass_fine = ceed
    ///     .operator(&qf_mass, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("u", &ru_fine, &bu_fine, VectorOpt::Active)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, &qdata)?
    ///     .field("v", &ru_fine, &bu_fine, VectorOpt::Active)?;
    ///
    /// // Multigrid setup
    /// let mut interp_c_to_f: Vec<Scalar> = vec![0.; p_coarse * p_fine];
    /// {
    ///     let mut coarse = ceed.vector(p_coarse)?;
    ///     let mut fine = ceed.vector(p_fine)?;
    ///     let basis_c_to_f =
    ///         ceed.basis_tensor_H1_Lagrange(1, 1, p_coarse, p_fine, QuadMode::GaussLobatto)?;
    ///     for i in 0..p_coarse {
    ///         coarse.set_value(0.0);
    ///         {
    ///             let mut array = coarse.view_mut()?;
    ///             array[i] = 1.;
    ///         }
    ///         basis_c_to_f.apply(
    ///             1,
    ///             TransposeMode::NoTranspose,
    ///             EvalMode::Interp,
    ///             &coarse,
    ///             &mut fine,
    ///         )?;
    ///         let array = fine.view()?;
    ///         for j in 0..p_fine {
    ///             interp_c_to_f[j * p_coarse + i] = array[j];
    ///         }
    ///     }
    /// }
    /// let (op_mass_coarse, op_prolong, op_restrict) = op_mass_fine.create_multigrid_level_H1(
    ///     &multiplicity,
    ///     &ru_coarse,
    ///     &bu_coarse,
    ///     &interp_c_to_f,
    /// )?;
    ///
    /// // Coarse problem
    /// u_coarse.set_value(1.0);
    /// op_mass_coarse.apply(&u_coarse, &mut v_coarse)?;
    ///
    /// // Check
    /// let sum: Scalar = v_coarse.view()?.iter().sum();
    /// assert!(
    ///     (sum - 2.0).abs() < 10.0 * libceed::EPSILON,
    ///     "Incorrect interval length computed"
    /// );
    ///
    /// // Prolong
    /// op_prolong.apply(&u_coarse, &mut u_fine)?;
    ///
    /// // Fine problem
    /// op_mass_fine.apply(&u_fine, &mut v_fine)?;
    ///
    /// // Check
    /// let sum: Scalar = v_fine.view()?.iter().sum();
    /// assert!(
    ///     (sum - 2.0).abs() < 10.0 * libceed::EPSILON,
    ///     "Incorrect interval length computed"
    /// );
    ///
    /// // Restrict
    /// op_restrict.apply(&v_fine, &mut v_coarse)?;
    ///
    /// // Check
    /// let sum: Scalar = v_coarse.view()?.iter().sum();
    /// assert!(
    ///     (sum - 2.0).abs() < 10.0 * libceed::EPSILON,
    ///     "Incorrect interval length computed"
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn create_multigrid_level_H1(
        &self,
        p_mult_fine: &Vector,
        rstr_coarse: &ElemRestriction,
        basis_coarse: &Basis,
        interpCtoF: &[Scalar],
    ) -> crate::Result<(Operator, Operator, Operator)> {
        let mut ptr_coarse = std::ptr::null_mut();
        let mut ptr_prolong = std::ptr::null_mut();
        let mut ptr_restrict = std::ptr::null_mut();
        let ierr = unsafe {
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
        self.op_core.check_error(ierr)?;
        let op_coarse = Operator::from_raw(ptr_coarse)?;
        let op_prolong = Operator::from_raw(ptr_prolong)?;
        let op_restrict = Operator::from_raw(ptr_restrict)?;
        Ok((op_coarse, op_prolong, op_restrict))
    }
}

// -----------------------------------------------------------------------------
// Composite Operator
// -----------------------------------------------------------------------------
impl<'a> CompositeOperator<'a> {
    // Constructor
    pub fn create(ceed: &'a crate::Ceed) -> crate::Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let ierr = unsafe { bind_ceed::CeedCompositeOperatorCreate(ceed.ptr, &mut ptr) };
        ceed.check_error(ierr)?;
        Ok(Self {
            op_core: OperatorCore {
                ptr,
                _lifeline: PhantomData,
            },
        })
    }

    /// Apply Operator to a vector
    ///
    /// * `input`  - Input Vector
    /// * `output` - Output Vector
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let ne = 4;
    /// let p = 3;
    /// let q = 4;
    /// let ndofs = p * ne - ne + 1;
    ///
    /// // Vectors
    /// let x = ceed.vector_from_slice(&[-1., -0.5, 0.0, 0.5, 1.0])?;
    /// let mut qdata_mass = ceed.vector(ne * q)?;
    /// qdata_mass.set_value(0.0);
    /// let mut qdata_diff = ceed.vector(ne * q)?;
    /// qdata_diff.set_value(0.0);
    /// let mut u = ceed.vector(ndofs)?;
    /// u.set_value(1.0);
    /// let mut v = ceed.vector(ndofs)?;
    /// v.set_value(0.0);
    ///
    /// // Restrictions
    /// let mut indx: Vec<i32> = vec![0; 2 * ne];
    /// for i in 0..ne {
    ///     indx[2 * i + 0] = i as i32;
    ///     indx[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let rx = ceed.elem_restriction(ne, 2, 1, 1, ne + 1, MemType::Host, &indx)?;
    /// let mut indu: Vec<i32> = vec![0; p * ne];
    /// for i in 0..ne {
    ///     indu[p * i + 0] = i as i32;
    ///     indu[p * i + 1] = (i + 1) as i32;
    ///     indu[p * i + 2] = (i + 2) as i32;
    /// }
    /// let ru = ceed.elem_restriction(ne, 3, 1, 1, ndofs, MemType::Host, &indu)?;
    /// let strides: [i32; 3] = [1, q as i32, q as i32];
    /// let rq = ceed.strided_elem_restriction(ne, q, 1, q * ne, strides)?;
    ///
    /// // Bases
    /// let bx = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, QuadMode::Gauss)?;
    /// let bu = ceed.basis_tensor_H1_Lagrange(1, 1, p, q, QuadMode::Gauss)?;
    ///
    /// // Build quadrature data
    /// let qf_build_mass = ceed.q_function_interior_by_name("Mass1DBuild")?;
    /// ceed.operator(&qf_build_mass, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("dx", &rx, &bx, VectorOpt::Active)?
    ///     .field("weights", ElemRestrictionOpt::None, &bx, VectorOpt::None)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active)?
    ///     .apply(&x, &mut qdata_mass)?;
    ///
    /// let qf_build_diff = ceed.q_function_interior_by_name("Poisson1DBuild")?;
    /// ceed.operator(&qf_build_diff, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("dx", &rx, &bx, VectorOpt::Active)?
    ///     .field("weights", ElemRestrictionOpt::None, &bx, VectorOpt::None)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active)?
    ///     .apply(&x, &mut qdata_diff)?;
    ///
    /// // Application operator
    /// let qf_mass = ceed.q_function_interior_by_name("MassApply")?;
    /// let op_mass = ceed
    ///     .operator(&qf_mass, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("u", &ru, &bu, VectorOpt::Active)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, &qdata_mass)?
    ///     .field("v", &ru, &bu, VectorOpt::Active)?;
    ///
    /// let qf_diff = ceed.q_function_interior_by_name("Poisson1DApply")?;
    /// let op_diff = ceed
    ///     .operator(&qf_diff, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("du", &ru, &bu, VectorOpt::Active)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, &qdata_diff)?
    ///     .field("dv", &ru, &bu, VectorOpt::Active)?;
    ///
    /// let op_composite = ceed
    ///     .composite_operator()?
    ///     .sub_operator(&op_mass)?
    ///     .sub_operator(&op_diff)?;
    ///
    /// v.set_value(0.0);
    /// op_composite.apply(&u, &mut v)?;
    ///
    /// // Check
    /// let sum: Scalar = v.view()?.iter().sum();
    /// assert!(
    ///     (sum - 2.0).abs() < 10.0 * libceed::EPSILON,
    ///     "Incorrect interval length computed"
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn apply(&self, input: &Vector, output: &mut Vector) -> crate::Result<i32> {
        self.op_core.apply(input, output)
    }

    /// Apply Operator to a vector and add result to output vector
    ///
    /// * `input`  - Input Vector
    /// * `output` - Output Vector
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let ne = 4;
    /// let p = 3;
    /// let q = 4;
    /// let ndofs = p * ne - ne + 1;
    ///
    /// // Vectors
    /// let x = ceed.vector_from_slice(&[-1., -0.5, 0.0, 0.5, 1.0])?;
    /// let mut qdata_mass = ceed.vector(ne * q)?;
    /// qdata_mass.set_value(0.0);
    /// let mut qdata_diff = ceed.vector(ne * q)?;
    /// qdata_diff.set_value(0.0);
    /// let mut u = ceed.vector(ndofs)?;
    /// u.set_value(1.0);
    /// let mut v = ceed.vector(ndofs)?;
    /// v.set_value(0.0);
    ///
    /// // Restrictions
    /// let mut indx: Vec<i32> = vec![0; 2 * ne];
    /// for i in 0..ne {
    ///     indx[2 * i + 0] = i as i32;
    ///     indx[2 * i + 1] = (i + 1) as i32;
    /// }
    /// let rx = ceed.elem_restriction(ne, 2, 1, 1, ne + 1, MemType::Host, &indx)?;
    /// let mut indu: Vec<i32> = vec![0; p * ne];
    /// for i in 0..ne {
    ///     indu[p * i + 0] = i as i32;
    ///     indu[p * i + 1] = (i + 1) as i32;
    ///     indu[p * i + 2] = (i + 2) as i32;
    /// }
    /// let ru = ceed.elem_restriction(ne, 3, 1, 1, ndofs, MemType::Host, &indu)?;
    /// let strides: [i32; 3] = [1, q as i32, q as i32];
    /// let rq = ceed.strided_elem_restriction(ne, q, 1, q * ne, strides)?;
    ///
    /// // Bases
    /// let bx = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, QuadMode::Gauss)?;
    /// let bu = ceed.basis_tensor_H1_Lagrange(1, 1, p, q, QuadMode::Gauss)?;
    ///
    /// // Build quadrature data
    /// let qf_build_mass = ceed.q_function_interior_by_name("Mass1DBuild")?;
    /// ceed.operator(&qf_build_mass, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("dx", &rx, &bx, VectorOpt::Active)?
    ///     .field("weights", ElemRestrictionOpt::None, &bx, VectorOpt::None)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active)?
    ///     .apply(&x, &mut qdata_mass)?;
    ///
    /// let qf_build_diff = ceed.q_function_interior_by_name("Poisson1DBuild")?;
    /// ceed.operator(&qf_build_diff, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("dx", &rx, &bx, VectorOpt::Active)?
    ///     .field("weights", ElemRestrictionOpt::None, &bx, VectorOpt::None)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, VectorOpt::Active)?
    ///     .apply(&x, &mut qdata_diff)?;
    ///
    /// // Application operator
    /// let qf_mass = ceed.q_function_interior_by_name("MassApply")?;
    /// let op_mass = ceed
    ///     .operator(&qf_mass, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("u", &ru, &bu, VectorOpt::Active)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, &qdata_mass)?
    ///     .field("v", &ru, &bu, VectorOpt::Active)?;
    ///
    /// let qf_diff = ceed.q_function_interior_by_name("Poisson1DApply")?;
    /// let op_diff = ceed
    ///     .operator(&qf_diff, QFunctionOpt::None, QFunctionOpt::None)?
    ///     .field("du", &ru, &bu, VectorOpt::Active)?
    ///     .field("qdata", &rq, BasisOpt::Collocated, &qdata_diff)?
    ///     .field("dv", &ru, &bu, VectorOpt::Active)?;
    ///
    /// let op_composite = ceed
    ///     .composite_operator()?
    ///     .sub_operator(&op_mass)?
    ///     .sub_operator(&op_diff)?;
    ///
    /// v.set_value(1.0);
    /// op_composite.apply_add(&u, &mut v)?;
    ///
    /// // Check
    /// let sum: Scalar = v.view()?.iter().sum();
    /// assert!(
    ///     (sum - (2.0 + ndofs as Scalar)).abs() < 10.0 * libceed::EPSILON,
    ///     "Incorrect interval length computed"
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn apply_add(&self, input: &Vector, output: &mut Vector) -> crate::Result<i32> {
        self.op_core.apply_add(input, output)
    }

    /// Add a sub-Operator to a Composite Operator
    ///
    /// * `subop` - Sub-Operator
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> libceed::Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let mut op = ceed.composite_operator()?;
    ///
    /// let qf_mass = ceed.q_function_interior_by_name("MassApply")?;
    /// let op_mass = ceed.operator(&qf_mass, QFunctionOpt::None, QFunctionOpt::None)?;
    /// op = op.sub_operator(&op_mass)?;
    ///
    /// let qf_diff = ceed.q_function_interior_by_name("Poisson1DApply")?;
    /// let op_diff = ceed.operator(&qf_diff, QFunctionOpt::None, QFunctionOpt::None)?;
    /// op = op.sub_operator(&op_diff)?;
    /// # Ok(())
    /// # }
    /// ```
    #[allow(unused_mut)]
    pub fn sub_operator(mut self, subop: &Operator) -> crate::Result<Self> {
        let ierr =
            unsafe { bind_ceed::CeedCompositeOperatorAddSub(self.op_core.ptr, subop.op_core.ptr) };
        self.op_core.check_error(ierr)?;
        Ok(self)
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
    pub fn linear_asssemble_diagonal(&self, assembled: &mut Vector) -> crate::Result<i32> {
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
    pub fn linear_assemble_add_diagonal(&self, assembled: &mut Vector) -> crate::Result<i32> {
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
    pub fn linear_assemble_point_block_diagonal(
        &self,
        assembled: &mut Vector,
    ) -> crate::Result<i32> {
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
    pub fn linear_assemble_add_point_block_diagonal(
        &self,
        assembled: &mut Vector,
    ) -> crate::Result<i32> {
        self.op_core.linear_assemble_add_diagonal(assembled)
    }
}

// -----------------------------------------------------------------------------
