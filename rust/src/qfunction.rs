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
// CeedQFunction option
// -----------------------------------------------------------------------------
#[derive(Clone, Copy)]
pub enum QFunctionOpt<'a> {
    Some(&'a QFunction),
    None,
}
/// Contruct a QFunctionOpt reference from a QFunction reference
impl<'a> From<&'a QFunction> for QFunctionOpt<'a> {
    fn from(qfunc: &'a QFunction) -> Self {
        Self::Some(qfunc)
    }
}
impl<'a> QFunctionOpt<'a> {
    /// Transform a Rust libCEED QFunction into C libCEED CeedQFunction
    pub(crate) fn to_raw(self) -> bind_ceed::CeedQFunction {
        match self {
            Self::Some(qfunc) => qfunc.ptr,
            Self::None => unsafe { bind_ceed::CEED_QFUNCTION_NONE },
        }
    }
}

// -----------------------------------------------------------------------------
// CeedQFunction context wrapper
// -----------------------------------------------------------------------------
pub struct QFunction {
    pub(crate) ptr: bind_ceed::CeedQFunction,
}

// -----------------------------------------------------------------------------
// Destructor
// -----------------------------------------------------------------------------
impl Drop for QFunction {
    fn drop(&mut self) {
        unsafe {
            if self.ptr != bind_ceed::CEED_QFUNCTION_NONE {
                bind_ceed::CeedQFunctionDestroy(&mut self.ptr);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Display
// -----------------------------------------------------------------------------
impl fmt::Display for QFunction {
    /// View a QFunction
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// let qf = ceed.q_function_interior_by_name("Mass1DBuild".to_string());
    /// println!("{}", qf);
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut ptr = std::ptr::null_mut();
        let mut sizeloc = crate::MAX_BUFFER_LENGTH;
        let file = unsafe { bind_ceed::open_memstream(&mut ptr, &mut sizeloc) };
        unsafe { bind_ceed::CeedQFunctionView(self.ptr, file) };
        unsafe { bind_ceed::fclose(file) };
        let cstring = unsafe { CString::from_raw(ptr) };
        cstring.to_string_lossy().fmt(f)
    }
}

// -----------------------------------------------------------------------------
// QFunction
// -----------------------------------------------------------------------------
impl QFunction {
    // Constructors
    pub fn create(
        ceed: &crate::Ceed,
        vlength: i32,
        f: bind_ceed::CeedQFunctionUser,
        source: impl Into<String>,
    ) -> Self {
        let source_c = CString::new(source.into()).expect("CString::new failed");
        let mut ptr = std::ptr::null_mut();
        unsafe {
            bind_ceed::CeedQFunctionCreateInterior(
                ceed.ptr,
                vlength,
                f,
                source_c.as_ptr(),
                &mut ptr,
            )
        };
        Self { ptr }
    }

    pub fn create_by_name(ceed: &crate::Ceed, name: impl Into<String>) -> Self {
        let name_c = CString::new(name.into()).expect("CString::new failed");
        let mut ptr = std::ptr::null_mut();
        unsafe {
            bind_ceed::CeedQFunctionCreateInteriorByName(ceed.ptr, name_c.as_ptr(), &mut ptr)
        };
        Self { ptr }
    }

    /// Apply the action of a QFunction
    ///
    /// * 'Q'      - The number of quadrature points
    /// * 'input'  - Array of input Vectors
    /// * 'output' - Array of output Vectors
    ///
    /// ```
    /// # let ceed = ceed::Ceed::default_init();
    /// const Q : usize = 8;
    /// let qf_build = ceed.q_function_interior_by_name("Mass1DBuild".to_string());
    /// let qf_mass = ceed.q_function_interior_by_name("MassApply".to_string());
    ///
    /// let mut j = [0.; Q];
    /// let mut w = [0.; Q];
    /// let mut u = [0.; Q];
    /// let mut v = [0.; Q];
    ///
    /// for i in 0..Q as usize {
    ///   let x = 2.*(i as f64)/((Q as f64) - 1.) - 1.;
    ///   j[i] = 1.;
    ///   w[i] = 1. - x*x;
    ///   u[i] = 2. + 3.*x + 5.*x*x;
    ///   v[i] = w[i] * u[i];
    /// }
    ///
    /// let J = ceed.vector_from_slice(&j);
    /// let W = ceed.vector_from_slice(&w);
    /// let U = ceed.vector_from_slice(&u);
    /// let mut V = ceed.vector(Q);
    /// V.set_value(0.0);
    /// let mut Qdata = ceed.vector(Q);
    /// Qdata.set_value(0.0);
    ///
    /// {
    ///   let mut input = vec![J, W];
    ///   let mut output = vec![Qdata];
    ///   qf_build.apply(Q as i32, &input, &output);
    ///   Qdata = output.remove(0);
    /// }
    ///
    /// {
    ///   let mut input = vec![Qdata, U];
    ///   let mut output = vec![V];
    ///   qf_mass.apply(Q as i32, &input, &output);
    ///   V = output.remove(0);
    /// }
    ///
    /// let array = V.view();
    /// for i in 0..Q {
    ///   assert_eq!(array[i], v[i], "Incorrect value in QFunction application");
    /// }
    /// ```
    pub fn apply(&self, Q: i32, u: &Vec<crate::vector::Vector>, v: &Vec<crate::vector::Vector>) {
        unsafe {
            let mut u_c = [std::ptr::null_mut(); 16];
            for i in 0..std::cmp::min(16, u.len()) {
                u_c[i] = u[i].ptr;
            }
            let mut v_c = [std::ptr::null_mut(); 16];
            for i in 0..std::cmp::min(16, v.len()) {
                v_c[i] = v[i].ptr;
            }
            bind_ceed::CeedQFunctionApply(self.ptr, Q, &mut u_c[0], &mut v_c[0]);
        }
    }

    /// Add a QFunction input
    ///
    /// * 'fieldname' - Name of QFunction field
    /// * 'size'      - Size of QFunction field, (ncomp * dim) of Grad or (ncomp
    ///   * 1) for None and Interp
    /// * 'emode'     - EvalMode::None to use values directly, EvalMode::Interp
    ///   to use interpolated values, EvalMode::Grad to use gradients,
    ///   EvalMode::Weight to use quadrature weights
    pub fn add_input(&self, fieldname: String, size: i32, emode: crate::EvalMode) {
        let name_c = CString::new(fieldname).expect("CString::new failed");
        unsafe {
            bind_ceed::CeedQFunctionAddInput(
                self.ptr,
                name_c.as_ptr(),
                size,
                emode as bind_ceed::CeedEvalMode,
            );
        }
    }

    /// Add a QFunction output
    ///
    /// * 'fieldname' - Name of QFunction field
    /// * 'size'      - Size of QFunction field, (ncomp * dim) of Grad or (ncomp
    ///   * 1) for None and Interp
    /// * 'emode'     - EvalMode::None to use values directly, EvalMode::Interp
    ///   to use interpolated values, EvalMode::Grad to use gradients
    pub fn add_output(&self, fieldname: String, size: i32, emode: crate::EvalMode) {
        let name_c = CString::new(fieldname).expect("CString::new failed");
        unsafe {
            bind_ceed::CeedQFunctionAddOutput(
                self.ptr,
                name_c.as_ptr(),
                size,
                emode as bind_ceed::CeedEvalMode,
            );
        }
    }
}

// -----------------------------------------------------------------------------
