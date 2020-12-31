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

//! A Ceed QFunction represents the spatial terms of the point-wise functions
//! describing the physics at the quadrature points.

use crate::prelude::*;

pub type QFunctionInputs<'a> = [&'a [f64]; MAX_QFUNCTION_FIELDS];
pub type QFunctionOutputs<'a> = [&'a mut [f64]; MAX_QFUNCTION_FIELDS];

// -----------------------------------------------------------------------------
// CeedQFunction option
// -----------------------------------------------------------------------------
#[derive(Clone, Copy)]
pub enum QFunctionOpt<'a> {
    SomeQFunction(&'a QFunction),
    SomeQFunctionByName(&'a QFunctionByName),
    None,
}

/// Construct a QFunctionOpt reference from a QFunction reference
impl<'a> From<&'a QFunction> for QFunctionOpt<'a> {
    fn from(qfunc: &'a QFunction) -> Self {
        debug_assert!(qfunc.qf_core.ptr != unsafe { bind_ceed::CEED_QFUNCTION_NONE });
        Self::SomeQFunction(qfunc)
    }
}

/// Construct a QFunctionOpt reference from a QFunction by Name reference
impl<'a> From<&'a QFunctionByName> for QFunctionOpt<'a> {
    fn from(qfunc: &'a QFunctionByName) -> Self {
        debug_assert!(qfunc.qf_core.ptr != unsafe { bind_ceed::CEED_QFUNCTION_NONE });
        Self::SomeQFunctionByName(qfunc)
    }
}

impl<'a> QFunctionOpt<'a> {
    /// Transform a Rust libCEED QFunctionOpt into C libCEED CeedQFunction
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

#[derive(Copy, Clone)]
struct QFunctionTrampolineData {
    number_inputs: usize,
    number_outputs: usize,
    input_sizes: [i32; MAX_QFUNCTION_FIELDS],
    output_sizes: [i32; MAX_QFUNCTION_FIELDS],
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
///   [u, weights, ..]: QFunctionInputs,
///   [v, ..]: QFunctionOutputs,
/// |
/// {
///   // Iterate over quadrature points
///   v
///     .iter_mut()
///     .zip(u.iter().zip(weights.iter()))
///     .for_each(|(v, (u, w))| *v = u * w);
///
///   // Return clean error code
///   0
/// };
///
/// let qf = ceed.q_function_interior(1, Box::new(user_f))
///     .input("u", 1, EvalMode::Interp)
///     .input("weights", 1, EvalMode::Weight)
///     .output("v", 1, EvalMode::Interp)
///     .build();
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
        let mut u_c = [std::ptr::null_mut(); MAX_QFUNCTION_FIELDS];
        for i in 0..std::cmp::min(MAX_QFUNCTION_FIELDS, u.len()) {
            u_c[i] = u[i].ptr;
        }
        let mut v_c = [std::ptr::null_mut(); MAX_QFUNCTION_FIELDS];
        for i in 0..std::cmp::min(MAX_QFUNCTION_FIELDS, v.len()) {
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
pub type QFunctionUserClosure =
    dyn FnMut([&[f64]; MAX_QFUNCTION_FIELDS], [&mut [f64]; MAX_QFUNCTION_FIELDS]) -> i32;

macro_rules! mut_max_fields {
    ($e:expr) => {
        [
            $e, $e, $e, $e, $e, $e, $e, $e, $e, $e, $e, $e, $e, $e, $e, $e,
        ]
    };
}
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
        std::slice::from_raw_parts(inputs, MAX_QFUNCTION_FIELDS);
    let mut inputs_array: [&[f64]; MAX_QFUNCTION_FIELDS] = [&[0.0]; MAX_QFUNCTION_FIELDS];
    inputs_slice
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            std::slice::from_raw_parts(x, (trampoline_data.input_sizes[i] * q) as usize) as &[f64]
        })
        .zip(inputs_array.iter_mut())
        .for_each(|(x, a)| *a = x);

    // Outputs
    let outputs_slice: &[*mut bind_ceed::CeedScalar] =
        std::slice::from_raw_parts(outputs, MAX_QFUNCTION_FIELDS);
    let mut outputs_array: [&mut [f64]; MAX_QFUNCTION_FIELDS] = mut_max_fields!(&mut [0.0]);
    outputs_slice
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            std::slice::from_raw_parts_mut(x, (trampoline_data.output_sizes[i] * q) as usize)
                as &mut [f64]
        })
        .zip(outputs_array.iter_mut())
        .for_each(|(x, a)| *a = x);

    // User closure
    (context.user_f)(inputs_array, outputs_array)
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
        let input_sizes = [0; MAX_QFUNCTION_FIELDS];
        let output_sizes = [0; MAX_QFUNCTION_FIELDS];
        let trampoline_data = QFunctionTrampolineData {
            number_inputs,
            number_outputs,
            input_sizes,
            output_sizes,
        };

        // Create QFunction
        unsafe {
            bind_ceed::CeedQFunctionCreateInterior(
                ceed.ptr,
                vlength,
                Some(trampoline),
                source_c.as_ptr(),
                &mut ptr,
            )
        };

        // Create QFunction context
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

    pub fn build(&mut self) -> Self {
        let dummy_f = |_: QFunctionInputs, _: QFunctionOutputs| 1;
        let mut user_f = std::mem::replace(&mut self.user_f, Box::new(dummy_f));
        std::mem::swap(&mut self.user_f, &mut user_f);
        Self {
            qf_core: QFunctionCore {
                ptr: std::mem::replace(&mut self.qf_core.ptr, std::ptr::null_mut()),
            },
            qf_ctx_ptr: std::mem::replace(&mut self.qf_ctx_ptr, std::ptr::null_mut()),
            trampoline_data: self.trampoline_data.clone(),
            user_f: user_f,
        }
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
    ///   [u, weights, ..]: QFunctionInputs,
    ///   [v, ..]: QFunctionOutputs,
    /// |
    /// {
    ///   // Iterate over quadrature points
    ///   v
    ///     .iter_mut()
    ///     .zip(u.iter().zip(weights.iter()))
    ///     .for_each(|(v, (u, w))| *v = u * w);
    ///
    ///   // Return clean error code
    ///   0
    /// };
    ///
    /// let qf = ceed.q_function_interior(1, Box::new(user_f))
    ///     .input("u", 1, EvalMode::Interp)
    ///     .input("weights", 1, EvalMode::Weight)
    ///     .output("v", 1, EvalMode::Interp)
    ///     .build();
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
    /// let uu = ceed.vector_from_slice(&u);
    /// let ww = ceed.vector_from_slice(&w);
    /// let mut vv = ceed.vector(Q);
    /// vv.set_value(0.0);
    /// {
    ///   let input = vec![uu, ww];
    ///   let mut output = vec![vv];
    ///   qf.apply(Q as i32, &input, &output);
    ///   vv = output.remove(0);
    /// }
    ///
    /// let array = vv.view();
    /// for i in 0..Q {
    ///   assert_eq!(array[i], v[i], "Incorrect value in QFunction application");
    /// }
    /// ```
    pub fn apply(&self, Q: i32, u: &[Vector], v: &[Vector]) {
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
    ///   [u, weights, ..]: QFunctionInputs,
    ///   [v, ..]: QFunctionOutputs,
    /// |
    /// {
    ///   // Iterate over quadrature points
    ///   v
    ///     .iter_mut()
    ///     .zip(u.iter().zip(weights.iter()))
    ///     .for_each(|(v, (u, w))| *v = u * w);
    ///
    ///   // Return clean error code
    ///   0
    /// };
    ///
    /// let mut qf = ceed.q_function_interior(1, Box::new(user_f));
    ///
    /// qf.input("u", 1, EvalMode::Interp);
    /// qf.input("weights", 1, EvalMode::Weight);
    /// ```
    pub fn input<'a>(
        &'a mut self,
        fieldname: &str,
        size: i32,
        emode: crate::EvalMode,
    ) -> &'a mut Self {
        let name_c = CString::new(fieldname).expect("CString::new failed");
        self.trampoline_data.input_sizes[self.trampoline_data.number_inputs] = size;
        self.trampoline_data.number_inputs += 1;
        let emode = emode as bind_ceed::CeedEvalMode;
        unsafe {
            bind_ceed::CeedQFunctionAddInput(self.qf_core.ptr, name_c.as_ptr(), size, emode);
        }
        self
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
    ///   [u, weights, ..]: QFunctionInputs,
    ///   [v, ..]: QFunctionOutputs,
    /// |
    /// {
    ///   // Iterate over quadrature points
    ///   v
    ///     .iter_mut()
    ///     .zip(u.iter().zip(weights.iter()))
    ///     .for_each(|(v, (u, w))| *v = u * w);
    ///
    ///   // Return clean error code
    ///   0
    /// };
    ///
    /// let mut qf = ceed.q_function_interior(1, Box::new(user_f));
    ///
    /// qf.output("v", 1, EvalMode::Interp);
    /// ```
    pub fn output<'a>(
        &'a mut self,
        fieldname: &str,
        size: i32,
        emode: crate::EvalMode,
    ) -> &'a mut Self {
        let name_c = CString::new(fieldname).expect("CString::new failed");
        self.trampoline_data.output_sizes[self.trampoline_data.number_outputs] = size;
        self.trampoline_data.number_outputs += 1;
        let emode = emode as bind_ceed::CeedEvalMode;
        unsafe {
            bind_ceed::CeedQFunctionAddOutput(self.qf_core.ptr, name_c.as_ptr(), size, emode);
        }
        self
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
    /// let jj = ceed.vector_from_slice(&j);
    /// let ww = ceed.vector_from_slice(&w);
    /// let uu = ceed.vector_from_slice(&u);
    /// let mut vv = ceed.vector(Q);
    /// vv.set_value(0.0);
    /// let mut qdata = ceed.vector(Q);
    /// qdata.set_value(0.0);
    ///
    /// {
    ///   let mut input = vec![jj, ww];
    ///   let mut output = vec![qdata];
    ///   qf_build.apply(Q as i32, &input, &output);
    ///   qdata = output.remove(0);
    /// }
    ///
    /// {
    ///   let mut input = vec![qdata, uu];
    ///   let mut output = vec![vv];
    ///   qf_mass.apply(Q as i32, &input, &output);
    ///   vv = output.remove(0);
    /// }
    ///
    /// let array = vv.view();
    /// for i in 0..Q {
    ///   assert_eq!(array[i], v[i], "Incorrect value in QFunction application");
    /// }
    /// ```
    pub fn apply(&self, Q: i32, u: &[Vector], v: &[Vector]) {
        self.qf_core.apply(Q, u, v)
    }
}

// -----------------------------------------------------------------------------
