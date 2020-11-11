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
    SomeQFunction(&'a QFunction),
    SomeQFunctionByName(&'a QFunctionByName),
    None,
}

/// Contruct a QFunctionOpt reference from a QFunction reference
impl<'a> From<&'a QFunction> for QFunctionOpt<'a> {
    fn from(qfunc: &'a QFunction) -> Self {
        Self::SomeQFunction(qfunc)
    }
}

/// Contruct a QFunctionOpt reference from a QFunction by Name reference
impl<'a> From<&'a QFunctionByName> for QFunctionOpt<'a> {
    fn from(qfunc: &'a QFunctionByName) -> Self {
        Self::SomeQFunctionByName(qfunc)
    }
}

impl<'a> QFunctionOpt<'a> {
    /// Transform a Rust libCEED QFunction into C libCEED CeedQFunction
    pub(crate) fn to_raw(self) -> bind_ceed::CeedQFunction {
        match self {
            Self::SomeQFunction(qfunc) => qfunc.qf_core.ptr,
            Self::SomeQFunctionByName(qfunc) => qfunc.qf_core.ptr,
            Self::None => unsafe { bind_ceed::CEED_QFUNCTION_NONE },
        }
    }
}

// -----------------------------------------------------------------------------
// CeedQFunction context wrapper
// -----------------------------------------------------------------------------
pub(crate) struct QFunctionCore {
    ptr: bind_ceed::CeedQFunction,
}

struct QFunctionTrampolineData {
    number_inputs: usize,
    number_outputs: usize,
    input_sizes: [i32; crate::MAX_QFUNCTION_FIELDS],
    output_sizes: [i32; crate::MAX_QFUNCTION_FIELDS],
}

pub struct QFunction {
    qf_core: QFunctionCore,
    qf_ctx_ptr: bind_ceed::CeedQFunctionContext,
    trampoline_data: QFunctionTrampolineData,
    user_f: Box<QFunctionUserClosure>,
}

pub struct QFunctionByName {
    qf_core: QFunctionCore,
}

// -----------------------------------------------------------------------------
// Destructor
// -----------------------------------------------------------------------------
impl Drop for QFunctionCore {
    fn drop(&mut self) {
        unsafe {
            if self.ptr != bind_ceed::CEED_QFUNCTION_NONE {
                bind_ceed::CeedQFunctionDestroy(&mut self.ptr);
            }
        }
    }
}

impl Drop for QFunction {
    fn drop(&mut self) {
        unsafe {
            bind_ceed::CeedQFunctionContextDestroy(&mut self.qf_ctx_ptr);
        }
        drop(&mut self.qf_core);
    }
}

// -----------------------------------------------------------------------------
// Display
// -----------------------------------------------------------------------------
impl fmt::Display for QFunctionCore {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut ptr = std::ptr::null_mut();
        let mut sizeloc = crate::MAX_BUFFER_LENGTH;
        let cstring = unsafe {
            let file = bind_ceed::open_memstream(&mut ptr, &mut sizeloc);
            bind_ceed::CeedQFunctionView(self.ptr, file);
            bind_ceed::fclose(file);
            CString::from_raw(ptr)
        };
        cstring.to_string_lossy().fmt(f)
    }
}
/// View a QFunction
///
/// ```
/// # use libceed::prelude::*;
/// # let ceed = libceed::Ceed::default_init();
/// let mut user_f = |
///   q: usize,
///   inputs: &[&[f64]],
///   outputs: &mut [&mut [f64]],
/// | -> i32
/// {
///   let u = &inputs[0];
///   let weights = &inputs[1];
///
///   let v = &mut outputs[0];
///
///   for i in 0..q {
///       v[i] = u[i] * weights[i];
///   }
///
///   return 0
/// };
///
/// let mut qf = ceed.q_function_interior(1, Box::new(user_f));
///
/// qf.add_input("u", 1, EvalMode::Interp);
/// qf.add_input("weights", 1, EvalMode::Weight);
///
/// qf.add_output("v", 1, EvalMode::Interp);
///
/// println!("{}", qf);
/// ```
impl fmt::Display for QFunction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.qf_core.fmt(f)
    }
}

/// View a QFunction by Name
///
/// ```
/// # use libceed::prelude::*;
/// # let ceed = libceed::Ceed::default_init();
/// let qf = ceed.q_function_interior_by_name("Mass1DBuild");
/// println!("{}", qf);
/// ```
impl fmt::Display for QFunctionByName {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.qf_core.fmt(f)
    }
}

// -----------------------------------------------------------------------------
// Core functionality
// -----------------------------------------------------------------------------
impl QFunctionCore {
    // Common implementation
    pub fn apply(&self, Q: i32, u: &[Vector], v: &[Vector]) {
        let mut u_c = [std::ptr::null_mut(); 16];
        for i in 0..std::cmp::min(16, u.len()) {
            u_c[i] = u[i].ptr;
        }
        let mut v_c = [std::ptr::null_mut(); 16];
        for i in 0..std::cmp::min(16, v.len()) {
            v_c[i] = v[i].ptr;
        }
        unsafe {
            bind_ceed::CeedQFunctionApply(self.ptr, Q, u_c.as_mut_ptr(), v_c.as_mut_ptr());
        }
    }
}

// -----------------------------------------------------------------------------
// User QFunction Closure
// -----------------------------------------------------------------------------
pub type QFunctionUserClosure = dyn FnMut(usize, &[&[f64]], &mut [&mut [f64]]) -> i32;

unsafe extern "C" fn trampoline(
    ctx: *mut ::std::os::raw::c_void,
    q: bind_ceed::CeedInt,
    inputs: *const *const bind_ceed::CeedScalar,
    outputs: *const *mut bind_ceed::CeedScalar,
) -> ::std::os::raw::c_int {
    let context = &mut *(ctx as *mut QFunction);
    let trampoline_data = &context.trampoline_data;

    // Inputs
    let inputs_slice: &[*const bind_ceed::CeedScalar] =
        std::slice::from_raw_parts(inputs, crate::MAX_QFUNCTION_FIELDS);
    let mut inputs_vec: Vec<&[f64]> = Vec::new();
    for i in 0..trampoline_data.number_inputs {
        inputs_vec.push(&std::slice::from_raw_parts(
            inputs_slice[i],
            (trampoline_data.input_sizes[i] * q) as usize,
        ) as &[f64]);
    }

    // Outputs
    let outputs_slice: &[*mut bind_ceed::CeedScalar] =
        std::slice::from_raw_parts(outputs, crate::MAX_QFUNCTION_FIELDS);
    let mut outputs_vec: Vec<&mut [f64]> = Vec::new();
    for i in 0..trampoline_data.number_outputs {
        outputs_vec.push(std::slice::from_raw_parts_mut(
            outputs_slice[i],
            (trampoline_data.output_sizes[i] * q) as usize,
        ));
    }

    // User closure
    (context.user_f)(q as usize, &inputs_vec, &mut outputs_vec)
}

// -----------------------------------------------------------------------------
// QFunction
// -----------------------------------------------------------------------------
impl QFunction {
    // Constructor
    pub fn create(ceed: &crate::Ceed, vlength: i32, user_f: Box<QFunctionUserClosure>) -> Self {
        let source_c = CString::new("").expect("CString::new failed");
        let mut ptr = std::ptr::null_mut();

        // Context for closure
        let number_inputs = 0;
        let number_outputs = 0;
        let input_sizes = [0; crate::MAX_QFUNCTION_FIELDS];
        let output_sizes = [0; crate::MAX_QFUNCTION_FIELDS];
        let trampoline_data = QFunctionTrampolineData {
            number_inputs,
            number_outputs,
            input_sizes,
            output_sizes,
        };

        // Create QF
        unsafe {
            bind_ceed::CeedQFunctionCreateInterior(
                ceed.ptr,
                vlength,
                Some(trampoline),
                source_c.as_ptr(),
                &mut ptr,
            )
        };

        // Create QF contetx
        let qf_ctx_ptr = std::ptr::null_mut();

        // Create object
        let qf_core = QFunctionCore { ptr };
        let mut qf_self = Self {
            qf_core,
            qf_ctx_ptr,
            trampoline_data,
            user_f,
        };

        // Set closure
        let mut qf_ctx_ptr = std::ptr::null_mut();
        unsafe {
            bind_ceed::CeedQFunctionContextCreate(ceed.ptr, &mut qf_ctx_ptr);
            bind_ceed::CeedQFunctionContextSetData(
                qf_ctx_ptr,
                crate::MemType::Host as bind_ceed::CeedMemType,
                crate::CopyMode::UsePointer as bind_ceed::CeedCopyMode,
                10, /* Note: size not relevant - CPU only approach */
                &mut qf_self as *mut _ as *mut ::std::os::raw::c_void,
            );
            bind_ceed::CeedQFunctionSetContext(qf_self.qf_core.ptr, qf_ctx_ptr);
        }
        qf_self.qf_ctx_ptr = qf_ctx_ptr;
        qf_self
    }

    /// Apply the action of a QFunction
    ///
    /// * `Q`      - The number of quadrature points
    /// * `input`  - Array of input Vectors
    /// * `output` - Array of output Vectors
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let mut user_f = |
    ///   q: usize,
    ///   inputs: &[&[f64]],
    ///   outputs: &mut [&mut [f64]],
    /// | -> i32
    /// {
    ///   let u = &inputs[0];
    ///   let weights = &inputs[1];
    ///
    ///   let v = &mut outputs[0];
    ///
    ///   for i in 0..q {
    ///     v[i] = u[i] * weights[i];
    ///   }
    ///
    ///   return 0
    /// };
    ///
    /// let mut qf = ceed.q_function_interior(1, Box::new(user_f));
    ///
    /// qf.add_input("u", 1, EvalMode::Interp);
    /// qf.add_input("weights", 1, EvalMode::Weight);
    ///
    /// qf.add_output("v", 1, EvalMode::Interp);
    ///
    /// const Q : usize = 8;
    /// let mut w = [0.; Q];
    /// let mut u = [0.; Q];
    /// let mut v = [0.; Q];
    ///
    /// for i in 0..Q as usize {
    ///   let x = 2.*(i as f64)/((Q as f64) - 1.) - 1.;
    ///   u[i] = 2. + 3.*x + 5.*x*x;
    ///   w[i] = 1. - x*x;
    ///   v[i] = u[i] * w[i];
    /// }
    ///
    /// let U = ceed.vector_from_slice(&u);
    /// let W = ceed.vector_from_slice(&w);
    /// let mut V = ceed.vector(Q);
    /// V.set_value(0.0);
    /// {
    ///   let input = vec![U, W];
    ///   let mut output = vec![V];
    ///   qf.apply(Q as i32, &input, &output);
    ///   V = output.remove(0);
    /// }
    ///
    /// let array = V.view();
    /// for i in 0..Q {
    ///   assert_eq!(array[i], v[i], "Incorrect value in QFunction application");
    /// }
    /// ```
    pub fn apply(&mut self, Q: i32, u: &[Vector], v: &[Vector]) {
        self.qf_core.apply(Q, u, v)
    }

    /// Add a QFunction input
    ///
    /// * `fieldname` - Name of QFunction field
    /// * `size`      - Size of QFunction field, `(ncomp * dim)` for `Grad` or
    ///                   `(ncomp * 1)` for `None`, `Interp`, and `Weight`
    /// * `emode`     - `EvalMode::None` to use values directly, `EvalMode::Interp`
    ///                   to use interpolated values, `EvalMode::Grad` to use
    ///                   gradients, `EvalMode::Weight` to use quadrature weights
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let mut user_f = |
    ///   q: usize,
    ///   inputs: &[&[f64]],
    ///   outputs: &mut [&mut [f64]],
    /// | -> i32
    /// {
    ///   let u = &inputs[0];
    ///   let weights = &inputs[1];
    ///
    ///   let v = &mut outputs[0];
    ///
    ///   for i in 0..q {
    ///     v[i] = u[i] * weights[i];
    ///   }
    ///
    ///   return 0
    /// };
    ///
    /// let mut qf = ceed.q_function_interior(1, Box::new(user_f));
    ///
    /// qf.add_input("u", 1, EvalMode::Interp);
    /// qf.add_input("weights", 1, EvalMode::Weight);
    /// ```
    pub fn add_input(&mut self, fieldname: &str, size: i32, emode: crate::EvalMode) {
        let name_c = CString::new(fieldname).expect("CString::new failed");
        self.trampoline_data.input_sizes[self.trampoline_data.number_inputs] = size;
        self.trampoline_data.number_inputs += 1;
        let emode = emode as bind_ceed::CeedEvalMode;
        unsafe {
            bind_ceed::CeedQFunctionAddInput(self.qf_core.ptr, name_c.as_ptr(), size, emode);
        }
    }

    /// Add a QFunction output
    ///
    /// * `fieldname` - Name of QFunction field
    /// * `size`      - Size of QFunction field, `(ncomp * dim)` for `Grad` or
    ///                   `(ncomp * 1)` for `None` and `Interp`
    /// * `emode`     - `EvalMode::None` to use values directly, `EvalMode::Interp`
    ///                   to use interpolated values, `EvalMode::Grad` to use
    ///                   gradients
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// let mut user_f = |
    ///     q: usize,
    ///     inputs: &[&[f64]],
    ///     outputs: &mut [&mut [f64]],
    /// | -> i32
    /// {
    ///     let u = &inputs[0];
    ///     let weights = &inputs[1];
    ///
    ///     let v = &mut outputs[0];
    ///
    ///     for i in 0..q {
    ///         v[i] = u[i] * weights[i];
    ///     }
    ///
    ///     return 0
    /// };
    ///
    /// let mut qf = ceed.q_function_interior(1, Box::new(user_f));
    ///
    /// qf.add_output("v", 1, EvalMode::Interp);
    /// ```
    pub fn add_output(&mut self, fieldname: &str, size: i32, emode: crate::EvalMode) {
        let name_c = CString::new(fieldname).expect("CString::new failed");
        self.trampoline_data.output_sizes[self.trampoline_data.number_outputs] = size;
        self.trampoline_data.number_outputs += 1;
        let emode = emode as bind_ceed::CeedEvalMode;
        unsafe {
            bind_ceed::CeedQFunctionAddOutput(self.qf_core.ptr, name_c.as_ptr(), size, emode);
        }
    }
}

// -----------------------------------------------------------------------------
// QFunction
// -----------------------------------------------------------------------------
impl QFunctionByName {
    // Constructor
    pub fn create(ceed: &crate::Ceed, name: &str) -> Self {
        let name_c = CString::new(name).expect("CString::new failed");
        let mut ptr = std::ptr::null_mut();
        unsafe {
            bind_ceed::CeedQFunctionCreateInteriorByName(ceed.ptr, name_c.as_ptr(), &mut ptr)
        };
        let qf_core = QFunctionCore { ptr };
        Self { qf_core }
    }

    /// Apply the action of a QFunction
    ///
    /// * `Q`      - The number of quadrature points
    /// * `input`  - Array of input Vectors
    /// * `output` - Array of output Vectors
    ///
    /// ```
    /// # use libceed::prelude::*;
    /// # let ceed = libceed::Ceed::default_init();
    /// const Q : usize = 8;
    /// let qf_build = ceed.q_function_interior_by_name("Mass1DBuild");
    /// let qf_mass = ceed.q_function_interior_by_name("MassApply");
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
    pub fn apply(&self, Q: i32, u: &[Vector], v: &[Vector]) {
        self.qf_core.apply(Q, u, v)
    }
}

// -----------------------------------------------------------------------------
