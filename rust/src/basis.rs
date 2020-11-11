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
// CeedBasis option
// -----------------------------------------------------------------------------
#[derive(Clone, Copy)]
pub enum BasisOpt<'a> {
    Some(&'a Basis),
    Collocated,
}
/// Contruct a BasisOpt reference from a Basis reference
impl<'a> From<&'a Basis> for BasisOpt<'a> {
    fn from(basis: &'a Basis) -> Self {
        Self::Some(basis)
    }
}
impl<'a> BasisOpt<'a> {
    /// Transform a Rust libCEED Basis into C libCEED CeedBasis
    pub(crate) fn to_raw(self) -> bind_ceed::CeedBasis {
        match self {
            Self::Some(basis) => basis.ptr,
            Self::Collocated => unsafe { bind_ceed::CEED_BASIS_COLLOCATED },
        }
    }
}

// -----------------------------------------------------------------------------
// CeedBasis context wrapper
// -----------------------------------------------------------------------------
pub struct Basis {
    pub(crate) ptr: bind_ceed::CeedBasis,
}

// -----------------------------------------------------------------------------
// Destructor
// -----------------------------------------------------------------------------
impl Drop for Basis {
    fn drop(&mut self) {
        unsafe {
            if self.ptr != bind_ceed::CEED_BASIS_COLLOCATED {
                bind_ceed::CeedBasisDestroy(&mut self.ptr);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Display
// -----------------------------------------------------------------------------
impl fmt::Display for Basis {
    /// View a Basis
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let b = ceed.basis_tensor_H1_Lagrange(1, 2, 3, 4, QuadMode::Gauss);
    /// println!("{}", b);
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut ptr = std::ptr::null_mut();
        let mut sizeloc = crate::MAX_BUFFER_LENGTH;
        let cstring = unsafe {
            let file = bind_ceed::open_memstream(&mut ptr, &mut sizeloc);
            bind_ceed::CeedBasisView(self.ptr, file);
            bind_ceed::fclose(file);
            CString::from_raw(ptr)
        };
        cstring.to_string_lossy().fmt(f)
    }
}

// -----------------------------------------------------------------------------
// Implementations
// -----------------------------------------------------------------------------
impl Basis {
    // Constructors
    pub fn create_tensor_H1(
        ceed: &crate::Ceed,
        dim: usize,
        ncomp: usize,
        P1d: usize,
        Q1d: usize,
        interp1d: &[f64],
        grad1d: &[f64],
        qref1d: &[f64],
        qweight1d: &[f64],
    ) -> Self {
        let mut ptr = std::ptr::null_mut();
        let (dim, ncomp, P1d, Q1d) = (dim as i32, ncomp as i32, P1d as i32, Q1d as i32);
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
        Self { ptr }
    }

    pub fn create_tensor_H1_Lagrange(
        ceed: &crate::Ceed,
        dim: usize,
        ncomp: usize,
        P: usize,
        Q: usize,
        qmode: crate::QuadMode,
    ) -> Self {
        let mut ptr = std::ptr::null_mut();
        let (dim, ncomp, P, Q) = (dim as i32, ncomp as i32, P as i32, Q as i32);
        let qmode = qmode as bind_ceed::CeedQuadMode;
        unsafe {
            bind_ceed::CeedBasisCreateTensorH1Lagrange(ceed.ptr, dim, ncomp, P, Q, qmode, &mut ptr);
        }
        Self { ptr }
    }

    pub fn create_H1(
        ceed: &crate::Ceed,
        topo: crate::ElemTopology,
        ncomp: usize,
        nnodes: usize,
        nqpts: usize,
        interp: &[f64],
        grad: &[f64],
        qref: &[f64],
        qweight: &[f64],
    ) -> Self {
        let mut ptr = std::ptr::null_mut();
        let (topo, ncomp, nnodes, nqpts) = (
            topo as bind_ceed::CeedElemTopology,
            ncomp as i32,
            nnodes as i32,
            nqpts as i32,
        );
        unsafe {
            bind_ceed::CeedBasisCreateH1(
                ceed.ptr,
                topo,
                ncomp,
                nnodes,
                nqpts,
                interp.as_ptr(),
                grad.as_ptr(),
                qref.as_ptr(),
                qweight.as_ptr(),
                &mut ptr,
            )
        };
        Self { ptr }
    }

    /// Apply basis evaluation from nodes to quadrature points or vice versa
    ///
    /// * `nelem` - The number of elements to apply the basis evaluation to
    /// * `tmode` - `TrasposeMode::NoTranspose` to evaluate from nodes to
    ///               quadrature points, `TransposeMode::Transpose` to apply the
    ///               transpose, mapping from quadrature points to nodes
    /// * `emode` - `EvalMode::None` to use values directly, `EvalMode::Interp`
    ///               to use interpolated values, `EvalMode::Grad` to use
    ///               gradients, `EvalMode::Weight` to use quadrature weights
    /// * `u`     - Input Vector
    /// * `v`     - Output Vector
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// const q: usize = 6;
    /// let bu = ceed.basis_tensor_H1_Lagrange(1, 1, q, q, QuadMode::GaussLobatto);
    /// let bx = ceed.basis_tensor_H1_Lagrange(1, 1, 2, q, QuadMode::Gauss);
    ///
    /// let x_corners = ceed.vector_from_slice(&[-1., 1.]);
    /// let mut x_qpts = ceed.vector(q);
    /// let mut x_nodes = ceed.vector(q);
    /// bx.apply(1, TransposeMode::NoTranspose, EvalMode::Interp,
    ///          &x_corners, &mut x_nodes);
    /// bu.apply(1, TransposeMode::NoTranspose, EvalMode::Interp,
    ///          &x_nodes, &mut x_qpts);
    ///
    /// // Create function x^3 + 1 on Gauss Lobatto points
    /// let mut u_arr = [0.; q];
    /// let x_nodes_arr = x_nodes.view();
    /// for i in 0..q {
    ///   u_arr[i] = x_nodes_arr[i]*x_nodes_arr[i]*x_nodes_arr[i] + 1.;
    /// }
    /// let u = ceed.vector_from_slice(&u_arr);
    ///
    /// // Map function to Gauss points
    /// let mut v = ceed.vector(q);
    /// v.set_value(0.);
    /// bu.apply(1, TransposeMode::NoTranspose, EvalMode::Interp, &u, &mut v);
    ///
    /// // Verify results
    /// let v_arr = v.view();
    /// let x_qpts_arr = x_qpts.view();
    /// for i in 0..q {
    ///   let true_value = x_qpts_arr[i]*x_qpts_arr[i]*x_qpts_arr[i] + 1.;
    ///   assert_eq!(v_arr[i], true_value, "Incorrect basis application");
    /// }
    /// ```
    pub fn apply(
        &self,
        nelem: i32,
        tmode: TransposeMode,
        emode: EvalMode,
        u: &Vector,
        v: &mut Vector,
    ) {
        let (tmode, emode) = (
            tmode as bind_ceed::CeedTransposeMode,
            emode as bind_ceed::CeedEvalMode,
        );
        unsafe { bind_ceed::CeedBasisApply(self.ptr, nelem, tmode, emode, u.ptr, v.ptr) };
    }

    /// Returns the dimension for given CeedBasis
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let dim = 2;
    /// let b = ceed.basis_tensor_H1_Lagrange(dim, 1, 3, 4, QuadMode::Gauss);
    ///
    /// let d = b.get_dimension();
    /// assert_eq!(d, dim as i32, "Incorrect dimension");
    /// ```
    pub fn get_dimension(&self) -> i32 {
        let mut dim = 0;
        unsafe { bind_ceed::CeedBasisGetDimension(self.ptr, &mut dim) };
        dim
    }

    /// Returns number of components for given CeedBasis
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let ncomp = 2;
    /// let b = ceed.basis_tensor_H1_Lagrange(1, ncomp, 3, 4, QuadMode::Gauss);
    ///
    /// let n = b.get_num_components();
    /// assert_eq!(n, ncomp as i32, "Incorrect number of components");
    /// ```
    pub fn get_num_components(&self) -> i32 {
        let mut ncomp = 0;
        unsafe { bind_ceed::CeedBasisGetNumComponents(self.ptr, &mut ncomp) };
        ncomp
    }

    /// Returns total number of nodes (in dim dimensions) of a CeedBasis
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let p = 3;
    /// let b = ceed.basis_tensor_H1_Lagrange(2, 1, p, 4, QuadMode::Gauss);
    ///
    /// let nnodes = b.get_num_nodes();
    /// assert_eq!(nnodes, (p*p) as i32, "Incorrect number of nodes");
    /// ```
    pub fn get_num_nodes(&self) -> i32 {
        let mut nnodes = 0;
        unsafe { bind_ceed::CeedBasisGetNumNodes(self.ptr, &mut nnodes) };
        nnodes
    }

    /// Returns total number of quadrature points (in dim dimensions) of a
    /// CeedBasis
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let q = 4;
    /// let b = ceed.basis_tensor_H1_Lagrange(2, 1, 3, q, QuadMode::Gauss);
    ///
    /// let nqpts = b.get_num_quadrature_points();
    /// assert_eq!(nqpts, (q*q) as i32, "Incorrect number of quadrature points");
    /// ```
    pub fn get_num_quadrature_points(&self) -> i32 {
        let mut Q = 0;
        unsafe {
            bind_ceed::CeedBasisGetNumQuadraturePoints(self.ptr, &mut Q);
        }
        Q
    }
}

// -----------------------------------------------------------------------------
