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

//! A Ceed Basis defines the discrete finite element basis and associated
//! quadrature rule.

use crate::prelude::*;

// -----------------------------------------------------------------------------
// CeedBasis option
// -----------------------------------------------------------------------------
#[derive(Debug)]
pub enum BasisOpt<'a> {
    Some(&'a Basis<'a>),
    Collocated,
}
/// Construct a BasisOpt reference from a Basis reference
impl<'a> From<&'a Basis<'_>> for BasisOpt<'a> {
    fn from(basis: &'a Basis) -> Self {
        debug_assert!(basis.ptr != unsafe { bind_ceed::CEED_BASIS_COLLOCATED });
        Self::Some(basis)
    }
}
impl<'a> BasisOpt<'a> {
    /// Transform a Rust libCEED BasisOpt into C libCEED CeedBasis
    pub(crate) fn to_raw(self) -> bind_ceed::CeedBasis {
        match self {
            Self::Some(basis) => basis.ptr,
            Self::Collocated => unsafe { bind_ceed::CEED_BASIS_COLLOCATED },
        }
    }

    /// Check if a BasisOpt is Some
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let b = ceed.basis_tensor_H1_Lagrange(1, 2, 3, 4, QuadMode::Gauss)?;
    /// let b_opt = BasisOpt::from(&b);
    /// assert!(b_opt.is_some(), "Incorrect BasisOpt");
    ///
    /// let b_opt = BasisOpt::Collocated;
    /// assert!(!b_opt.is_some(), "Incorrect BasisOpt");
    /// # Ok(())
    /// # }
    /// ```
    pub fn is_some(&self) -> bool {
        match self {
            Self::Some(_) => true,
            Self::Collocated => false,
        }
    }

    /// Check if a BasisOpt is Collocated
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let b = ceed.basis_tensor_H1_Lagrange(1, 2, 3, 4, QuadMode::Gauss)?;
    /// let b_opt = BasisOpt::from(&b);
    /// assert!(!b_opt.is_collocated(), "Incorrect BasisOpt");
    ///
    /// let b_opt = BasisOpt::Collocated;
    /// assert!(b_opt.is_collocated(), "Incorrect BasisOpt");
    /// # Ok(())
    /// # }
    /// ```
    pub fn is_collocated(&self) -> bool {
        match self {
            Self::Some(_) => false,
            Self::Collocated => true,
        }
    }
}

// -----------------------------------------------------------------------------
// CeedBasis context wrapper
// -----------------------------------------------------------------------------
#[derive(Debug)]
pub struct Basis<'a> {
    pub(crate) ptr: bind_ceed::CeedBasis,
    _lifeline: PhantomData<&'a ()>,
}

// -----------------------------------------------------------------------------
// Destructor
// -----------------------------------------------------------------------------
impl<'a> Drop for Basis<'a> {
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
impl<'a> fmt::Display for Basis<'a> {
    /// View a Basis
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let b = ceed.basis_tensor_H1_Lagrange(1, 2, 3, 4, QuadMode::Gauss)?;
    /// println!("{}", b);
    /// # Ok(())
    /// # }
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
impl<'a> Basis<'a> {
    // Constructors
    pub fn create_tensor_H1(
        ceed: &'a crate::Ceed,
        dim: usize,
        ncomp: usize,
        P1d: usize,
        Q1d: usize,
        interp1d: &[crate::Scalar],
        grad1d: &[crate::Scalar],
        qref1d: &[crate::Scalar],
        qweight1d: &[crate::Scalar],
    ) -> crate::Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let (dim, ncomp, P1d, Q1d) = (
            i32::try_from(dim).unwrap(),
            i32::try_from(ncomp).unwrap(),
            i32::try_from(P1d).unwrap(),
            i32::try_from(Q1d).unwrap(),
        );
        let ierr = unsafe {
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
        ceed.check_error(ierr)?;
        Ok(Self {
            ptr,
            _lifeline: PhantomData,
        })
    }

    pub fn create_tensor_H1_Lagrange(
        ceed: &'a crate::Ceed,
        dim: usize,
        ncomp: usize,
        P: usize,
        Q: usize,
        qmode: crate::QuadMode,
    ) -> crate::Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let (dim, ncomp, P, Q, qmode) = (
            i32::try_from(dim).unwrap(),
            i32::try_from(ncomp).unwrap(),
            i32::try_from(P).unwrap(),
            i32::try_from(Q).unwrap(),
            qmode as bind_ceed::CeedQuadMode,
        );
        let ierr = unsafe {
            bind_ceed::CeedBasisCreateTensorH1Lagrange(ceed.ptr, dim, ncomp, P, Q, qmode, &mut ptr)
        };
        ceed.check_error(ierr)?;
        Ok(Self {
            ptr,
            _lifeline: PhantomData,
        })
    }

    pub fn create_H1(
        ceed: &'a crate::Ceed,
        topo: crate::ElemTopology,
        ncomp: usize,
        nnodes: usize,
        nqpts: usize,
        interp: &[crate::Scalar],
        grad: &[crate::Scalar],
        qref: &[crate::Scalar],
        qweight: &[crate::Scalar],
    ) -> crate::Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let (topo, ncomp, nnodes, nqpts) = (
            topo as bind_ceed::CeedElemTopology,
            i32::try_from(ncomp).unwrap(),
            i32::try_from(nnodes).unwrap(),
            i32::try_from(nqpts).unwrap(),
        );
        let ierr = unsafe {
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
        ceed.check_error(ierr)?;
        Ok(Self {
            ptr,
            _lifeline: PhantomData,
        })
    }

    // Error handling
    #[doc(hidden)]
    fn check_error(&self, ierr: i32) -> crate::Result<i32> {
        let mut ptr = std::ptr::null_mut();
        unsafe {
            bind_ceed::CeedBasisGetCeed(self.ptr, &mut ptr);
        }
        crate::check_error(ptr, ierr)
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
    /// # fn main() -> Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// const Q: usize = 6;
    /// let bu = ceed.basis_tensor_H1_Lagrange(1, 1, Q, Q, QuadMode::GaussLobatto)?;
    /// let bx = ceed.basis_tensor_H1_Lagrange(1, 1, 2, Q, QuadMode::Gauss)?;
    ///
    /// let x_corners = ceed.vector_from_slice(&[-1., 1.])?;
    /// let mut x_qpts = ceed.vector(Q)?;
    /// let mut x_nodes = ceed.vector(Q)?;
    /// bx.apply(
    ///     1,
    ///     TransposeMode::NoTranspose,
    ///     EvalMode::Interp,
    ///     &x_corners,
    ///     &mut x_nodes,
    /// )?;
    /// bu.apply(
    ///     1,
    ///     TransposeMode::NoTranspose,
    ///     EvalMode::Interp,
    ///     &x_nodes,
    ///     &mut x_qpts,
    /// )?;
    ///
    /// // Create function x^3 + 1 on Gauss Lobatto points
    /// let mut u_arr = [0.; Q];
    /// u_arr
    ///     .iter_mut()
    ///     .zip(x_nodes.view()?.iter())
    ///     .for_each(|(u, x)| *u = x * x * x + 1.);
    /// let u = ceed.vector_from_slice(&u_arr)?;
    ///
    /// // Map function to Gauss points
    /// let mut v = ceed.vector(Q)?;
    /// v.set_value(0.);
    /// bu.apply(1, TransposeMode::NoTranspose, EvalMode::Interp, &u, &mut v)?;
    ///
    /// // Verify results
    /// v.view()?
    ///     .iter()
    ///     .zip(x_qpts.view()?.iter())
    ///     .for_each(|(v, x)| {
    ///         let true_value = x * x * x + 1.;
    ///         assert!(
    ///             (*v - true_value).abs() < 10.0 * libceed::EPSILON,
    ///             "Incorrect basis application"
    ///         );
    ///     });
    /// # Ok(())
    /// # }
    /// ```
    pub fn apply(
        &self,
        nelem: usize,
        tmode: TransposeMode,
        emode: EvalMode,
        u: &Vector,
        v: &mut Vector,
    ) -> crate::Result<i32> {
        let (nelem, tmode, emode) = (
            i32::try_from(nelem).unwrap(),
            tmode as bind_ceed::CeedTransposeMode,
            emode as bind_ceed::CeedEvalMode,
        );
        let ierr =
            unsafe { bind_ceed::CeedBasisApply(self.ptr, nelem, tmode, emode, u.ptr, v.ptr) };
        self.check_error(ierr)
    }

    /// Returns the dimension for given CeedBasis
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let dim = 2;
    /// let b = ceed.basis_tensor_H1_Lagrange(dim, 1, 3, 4, QuadMode::Gauss)?;
    ///
    /// let d = b.dimension();
    /// assert_eq!(d, dim, "Incorrect dimension");
    /// # Ok(())
    /// # }
    /// ```
    pub fn dimension(&self) -> usize {
        let mut dim = 0;
        unsafe { bind_ceed::CeedBasisGetDimension(self.ptr, &mut dim) };
        usize::try_from(dim).unwrap()
    }

    /// Returns number of components for given CeedBasis
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let ncomp = 2;
    /// let b = ceed.basis_tensor_H1_Lagrange(1, ncomp, 3, 4, QuadMode::Gauss)?;
    ///
    /// let n = b.num_components();
    /// assert_eq!(n, ncomp, "Incorrect number of components");
    /// # Ok(())
    /// # }
    /// ```
    pub fn num_components(&self) -> usize {
        let mut ncomp = 0;
        unsafe { bind_ceed::CeedBasisGetNumComponents(self.ptr, &mut ncomp) };
        usize::try_from(ncomp).unwrap()
    }

    /// Returns total number of nodes (in dim dimensions) of a CeedBasis
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let p = 3;
    /// let b = ceed.basis_tensor_H1_Lagrange(2, 1, p, 4, QuadMode::Gauss)?;
    ///
    /// let nnodes = b.num_nodes();
    /// assert_eq!(nnodes, p * p, "Incorrect number of nodes");
    /// # Ok(())
    /// # }
    /// ```
    pub fn num_nodes(&self) -> usize {
        let mut nnodes = 0;
        unsafe { bind_ceed::CeedBasisGetNumNodes(self.ptr, &mut nnodes) };
        usize::try_from(nnodes).unwrap()
    }

    /// Returns total number of quadrature points (in dim dimensions) of a
    /// CeedBasis
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # fn main() -> Result<()> {
    /// # let ceed = libceed::Ceed::default_init();
    /// let q = 4;
    /// let b = ceed.basis_tensor_H1_Lagrange(2, 1, 3, q, QuadMode::Gauss)?;
    ///
    /// let nqpts = b.num_quadrature_points();
    /// assert_eq!(nqpts, q * q, "Incorrect number of quadrature points");
    /// # Ok(())
    /// # }
    /// ```
    pub fn num_quadrature_points(&self) -> usize {
        let mut Q = 0;
        unsafe {
            bind_ceed::CeedBasisGetNumQuadraturePoints(self.ptr, &mut Q);
        }
        usize::try_from(Q).unwrap()
    }
}

// -----------------------------------------------------------------------------
