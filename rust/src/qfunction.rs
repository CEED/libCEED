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
// CeedQFunction context wrapper
// -----------------------------------------------------------------------------
pub struct QFunctionCore<'a> {
    pub(crate) ceed: &'a crate::Ceed,
    pub(crate) ptr: bind_ceed::CeedQFunction,
}

pub struct QFunction<'a> {
    pub(crate) qf_core: QFunctionCore<'a>,
}

pub struct QFunctionByName<'a> {
    pub(crate) qf_core: QFunctionCore<'a>,
}

// -----------------------------------------------------------------------------
// Destructor
// -----------------------------------------------------------------------------
impl<'a> Drop for QFunctionCore<'a> {
    fn drop(&mut self) {
        unsafe {
            bind_ceed::CeedQFunctionDestroy(&mut self.ptr);
        }
    }
}

// -----------------------------------------------------------------------------
// Display
// -----------------------------------------------------------------------------
impl<'a> fmt::Display for QFunctionCore<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut ptr = std::ptr::null_mut();
        let mut sizeloc = 202020;
        unsafe {
            let file = bind_ceed::open_memstream(&mut ptr, &mut sizeloc);
            bind_ceed::CeedQFunctionView(self.ptr, file);
            bind_ceed::fclose(file);
            let cstring = CString::from_raw(ptr);
            let s = cstring.to_string_lossy().into_owned();
            write!(f, "{}", s)
        }
    }
}

impl<'a> fmt::Display for QFunction<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.qf_core.fmt(f)
    }
}

/// View a QFunction created by name
///
/// ```
/// # let ceed = ceed::Ceed::default_init();
/// let qf = ceed.q_function_interior_by_name("Mass1DBuild".to_string());
/// println!("{}", qf);
/// ```
impl<'a> fmt::Display for QFunctionByName<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.qf_core.fmt(f)
    }
}

// -----------------------------------------------------------------------------
// Core functionality
// -----------------------------------------------------------------------------
impl<'a> QFunctionCore<'a> {
    // Constructor
    pub fn new(ceed: &'a crate::Ceed, ptr: bind_ceed::CeedQFunction) -> Self {
        Self { ceed, ptr }
    }

    // Common implementation
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
}

// -----------------------------------------------------------------------------
// QFunction
// -----------------------------------------------------------------------------
impl<'a> QFunction<'a> {
    /// Constructor
    //pub fn create(ceed: &'a crate::Ceed, n: usize) -> Self {
    //
    //}

    /// Apply the action of a QFunction
    ///
    /// * 'Q'      - The number of quadrature points
    /// * 'input'  - Array of input Vectors
    /// * 'output' - Array of output Vectors
    ///
    pub fn apply(&self, Q: i32, u: &Vec<crate::vector::Vector>, v: &Vec<crate::vector::Vector>) {
        self.qf_core.apply(Q, u, v);
    }

    /// Add a QFunction input
    ///
    /// * 'fieldname' - Name of QFunction field
    /// * 'size'      - Size of QFunction field, (ncomp * dim) of Grad or
    ///                   (ncomp * 1) for None and Interp
    /// * 'emode'     - EvalMode::None to use values directly,
    ///                   EvalMode::Interp to use interpolated values,
    ///                   EvalMode::Grad to use gradients,
    ///                   EvalMode::Weight to use quadrature weights
    ///
    pub fn add_input(&self, fieldname: String, size: i32, emode: crate::EvalMode) {
        let name_c = CString::new(fieldname).expect("CString::new failed");
        unsafe {
            bind_ceed::CeedQFunctionAddInput(
                self.qf_core.ptr,
                name_c.as_ptr(),
                size,
                emode as bind_ceed::CeedEvalMode,
            );
        }
    }

    /// Add a QFunction output
    ///
    /// * 'fieldname' - Name of QFunction field
    /// * 'size'      - Size of QFunction field, (ncomp * dim) of Grad or
    ///                   (ncomp * 1) for None and Interp
    /// * 'emode'     - EvalMode::None to use values directly,
    ///                   EvalMode::Interp to use interpolated values,
    ///                   EvalMode::Grad to use gradients
    ///
    pub fn add_output(&self, fieldname: String, size: i32, emode: crate::EvalMode) {
        let name_c = CString::new(fieldname).expect("CString::new failed");
        unsafe {
            bind_ceed::CeedQFunctionAddOutput(
                self.qf_core.ptr,
                name_c.as_ptr(),
                size,
                emode as bind_ceed::CeedEvalMode,
            );
        }
    }

    /// Set global context for a QFunction
    ///
    /// * 'ctx' - Context data to set
    ///
    pub fn set_context(&self, ctx: crate::qfunction_context::QFunctionContext) {
        unsafe {
            bind_ceed::CeedQFunctionSetContext(self.qf_core.ptr, ctx.ptr);
        }
    }
}

// -----------------------------------------------------------------------------
// QFunction by Name
// -----------------------------------------------------------------------------
impl<'a> QFunctionByName<'a> {
    // Constructor
    pub fn create(ceed: &'a crate::Ceed, name: String) -> Self {
        let name_c = CString::new(name).expect("CString::new failed");
        let mut ptr = std::ptr::null_mut();
        unsafe {
            bind_ceed::CeedQFunctionCreateInteriorByName(ceed.ptr, name_c.as_ptr(), &mut ptr)
        };
        let qf_core = QFunctionCore::new(ceed, ptr);
        Self { qf_core }
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
    /// let array = V.get_array(ceed::MemType::Host);
    /// for i in 0..Q {
    ///   assert_eq!(array[i], v[i]);
    /// }
    /// V.restore_array(array);
    /// ```
    pub fn apply(&self, Q: i32, u: &Vec<crate::vector::Vector>, v: &Vec<crate::vector::Vector>) {
        self.qf_core.apply(Q, u, v);
    }
}

// -----------------------------------------------------------------------------
