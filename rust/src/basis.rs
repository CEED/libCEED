use crate::prelude::*;

/// CeedBasis context wrapper
pub struct Basis<'a> {
    ceed: &'a crate::Ceed,
    pub ptr: bind_ceed::CeedBasis,
}

impl<'a> Basis<'a> {
    /// Constructors
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
        let mut ptr = std::ptr::null_mut();
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
        qmode: crate::QuadMode,
    ) -> Self {
        let mut ptr = std::ptr::null_mut();
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
        topo: crate::ElemTopology,
        ncomp: i32,
        nnodes: i32,
        nqpts: i32,
        interp: &Vec<f64>,
        grad: &Vec<f64>,
        qref: &Vec<f64>,
        qweight: &Vec<f64>,
    ) -> Self {
        let mut ptr = std::ptr::null_mut();
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
                &mut ptr,
            )
        };
        Self { ceed, ptr }
    }

    pub fn apply(
        &self,
        nelem: i32,
        tmode: crate::TransposeMode,
        emode: crate::EvalMode,
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

    /// Returns the dimension for given CeedBasis
    ///
    /// ```
    /// # let ceed = ceed::Ceed::init("/cpu/self/ref/serial");
    /// let b = ceed.basis_tensor_H1_Lagrange(2, 1, 3, 4, ceed::QuadMode::Gauss);
    /// let dim = b.get_dimension();
    /// assert!(dim == 2);
    /// ```
    pub fn get_dimension(&self) -> i32 {
        let mut dim = 0;
        unsafe { bind_ceed::CeedBasisGetDimension(self.ptr, &mut dim) };
        dim
    }

    pub fn get_topology(&self) -> crate::ElemTopology {
        let mut topo = crate::ElemTopology::Line;
        unsafe {
            // bind_ceed::CeedBasisGetTopology(
            //     self.ptr,
            //     &mut topo as &mut bind_ceed::CeedElemTopology,
            // )
        };
        topo
    }

    /// Returns number of components for given CeedBasis
    ///
    /// ```
    /// # let ceed = ceed::Ceed::init("/cpu/self/ref/serial");
    /// let b = ceed.basis_tensor_H1_Lagrange(1, 2, 3, 4, ceed::QuadMode::Gauss);
    /// let ncomp = b.get_num_components();
    /// assert!(ncomp == 2);
    /// ```
    pub fn get_num_components(&self) -> i32 {
        let mut ncomp = 0;
        unsafe { bind_ceed::CeedBasisGetNumComponents(self.ptr, &mut ncomp) };
        ncomp
    }

    /// Returns total number of nodes (in dim dimensions) of a CeedBasis
    ///
    /// ```
    /// # let ceed = ceed::Ceed::init("/cpu/self/ref/serial");
    /// let b = ceed.basis_tensor_H1_Lagrange(2, 1, 3, 4, ceed::QuadMode::Gauss);
    /// let nqpts = b.get_num_nodes();
    /// assert!(nqpts == 3*3);
    /// ```
    pub fn get_num_nodes(&self) -> i32 {
        let mut nnodes = 0;
        unsafe { bind_ceed::CeedBasisGetNumNodes(self.ptr, &mut nnodes) };
        nnodes
    }

    /// Returns total number of quadrature points (in dim dimensions) of a CeedBasis
    ///
    /// ```
    /// # let ceed = ceed::Ceed::init("/cpu/self/ref/serial");
    /// let b = ceed.basis_tensor_H1_Lagrange(2, 1, 3, 4, ceed::QuadMode::Gauss);
    /// let ncomp = b.get_num_quadrature_points();
    /// assert!(ncomp == 4*4);
    /// ```
    pub fn get_num_quadrature_points(&self) -> i32 {
        let mut Q = 0;
        unsafe {
            bind_ceed::CeedBasisGetNumQuadraturePoints(self.ptr, &mut Q);
        }
        Q
    }
}

/// Destructor
impl<'a> Drop for Basis<'a> {
    fn drop(&mut self) {
        unsafe {
            bind_ceed::CeedBasisDestroy(&mut self.ptr);
        }
    }
}
