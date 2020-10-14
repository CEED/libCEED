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
        let file = unsafe { bind_ceed::open_memstream(&mut ptr, &mut sizeloc) };
        unsafe { bind_ceed::CeedQFunctionView(self.ptr, file) };
        unsafe { bind_ceed::fclose(file) };
        let cstring = unsafe { CString::from_raw(ptr) };
        cstring.to_string_lossy().fmt(f)
    }
}

impl fmt::Display for QFunction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.qf_core.fmt(f)
    }
}

/// View a QFunction by Name
///
/// ```
/// # let ceed = ceed::Ceed::default_init();
/// let qf = ceed.q_function_interior_by_name("Mass1DBuild".to_string());
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
// User QFunction Closure
// -----------------------------------------------------------------------------
pub type QFunctionUserClosure = dyn FnMut(usize, &Vec<&[f64]>, &mut Vec<&mut [f64]>) -> i32;

unsafe extern "C" fn trampoline<F>(
    ctx: *mut ::std::os::raw::c_void,
    q: bind_ceed::CeedInt,
    inputs: *const *const bind_ceed::CeedScalar,
    outputs: *const *mut bind_ceed::CeedScalar,
) -> ::std::os::raw::c_int
where
    F: FnMut(
        bind_ceed::CeedInt,
        *const *const bind_ceed::CeedScalar,
        *const *mut bind_ceed::CeedScalar,
    ) -> i32,
{
    let inner_function = &mut *(ctx as *mut F);
    inner_function(q, inputs, outputs)
}

pub fn get_trampoline<F>(_closure: &F) -> bind_ceed::CeedQFunctionUser
where
    F: FnMut(
        bind_ceed::CeedInt,
        *const *const bind_ceed::CeedScalar,
        *const *mut bind_ceed::CeedScalar,
    ) -> i32,
{
    Some(trampoline::<F>)
}

// -----------------------------------------------------------------------------
// QFunction
// -----------------------------------------------------------------------------
impl QFunction {
    // Constructor
    pub fn create(
        ceed: &crate::Ceed,
        vlength: i32,
        mut f: Box<&mut QFunctionUserClosure>,
        source: impl Into<String>,
    ) -> Self {
        let source_c = CString::new(source.into()).expect("CString::new failed");
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

        // Closure around user closure
        let mut qf_closure = |q: bind_ceed::CeedInt,
                              inputs: *const *const bind_ceed::CeedScalar,
                              outputs: *const *mut bind_ceed::CeedScalar|
         -> i32 {
            // Inputs
            let mut inputs_vec = Vec::new();
            for i in 0..trampoline_data.number_inputs {
                inputs_vec.push(unsafe {
                    std::slice::from_raw_parts(
                        inputs.offset(i as isize) as *const f64,
                        (input_sizes[i] * q) as usize,
                    )
                } as &[f64]);
            }

            // Outputs
            let mut outputs_vec = Vec::new();
            for i in 0..trampoline_data.number_outputs {
                outputs_vec.push(unsafe {
                    std::slice::from_raw_parts_mut(
                        outputs.offset(i as isize) as *mut f64,
                        (output_sizes[i] * q) as usize,
                    )
                } as &mut [f64]);
            }

            // User closure
            f(q as usize, &inputs_vec, &mut outputs_vec)
        };

        // Create QF
        let f_trampoline = get_trampoline(&qf_closure);
        unsafe {
            bind_ceed::CeedQFunctionCreateInterior(
                ceed.ptr,
                vlength,
                f_trampoline,
                source_c.as_ptr(),
                &mut ptr,
            )
        };

        // Create QF contetx
        let mut qf_ctx_ptr = std::ptr::null_mut();
        unsafe {
            bind_ceed::CeedQFunctionContextCreate(ceed.ptr, &mut qf_ctx_ptr);
            bind_ceed::CeedQFunctionContextSetData(
                qf_ctx_ptr,
                crate::MemType::Host as bind_ceed::CeedMemType,
                crate::CopyMode::UsePointer as bind_ceed::CeedCopyMode,
                10,
                &mut qf_closure as *mut _ as *mut ::std::os::raw::c_void,
            );
            bind_ceed::CeedQFunctionSetContext(ptr, qf_ctx_ptr);
        }

        // Create object
        let qf_core = QFunctionCore { ptr };
        Self {
            qf_core,
            qf_ctx_ptr,
            trampoline_data,
        }
    }

    /// Apply the action of a QFunction
    ///
    /// * 'Q'      - The number of quadrature points
    /// * 'input'  - Array of input Vectors
    /// * 'output' - Array of output Vectors
    pub fn apply(&self, Q: i32, u: &Vec<crate::vector::Vector>, v: &Vec<crate::vector::Vector>) {
        self.qf_core.apply(Q, u, v)
    }

    /// Add a QFunction input
    ///
    /// * 'fieldname' - Name of QFunction field
    /// * 'size'      - Size of QFunction field, (ncomp * dim) of Grad or
    ///                   (ncomp * 1) for None and Interp
    /// * 'emode'     - EvalMode::None to use values directly, EvalMode::Interp
    ///                   to use interpolated values, EvalMode::Grad to use
    ///                   gradients, EvalMode::Weight to use quadrature weights
    pub fn add_input(&mut self, fieldname: String, size: i32, emode: crate::EvalMode) {
        let name_c = CString::new(fieldname).expect("CString::new failed");
        self.trampoline_data.input_sizes[self.trampoline_data.number_inputs] = size;
        self.trampoline_data.number_inputs += 1;
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
    /// * 'emode'     - EvalMode::None to use values directly, EvalMode::Interp
    ///                   to use interpolated values, EvalMode::Grad to use
    ///                   gradients
    pub fn add_output(&mut self, fieldname: String, size: i32, emode: crate::EvalMode) {
        let name_c = CString::new(fieldname).expect("CString::new failed");
        self.trampoline_data.output_sizes[self.trampoline_data.number_outputs] = size;
        self.trampoline_data.number_outputs += 1;
        unsafe {
            bind_ceed::CeedQFunctionAddOutput(
                self.qf_core.ptr,
                name_c.as_ptr(),
                size,
                emode as bind_ceed::CeedEvalMode,
            );
        }
    }
}

// -----------------------------------------------------------------------------
// QFunction
// -----------------------------------------------------------------------------
impl QFunctionByName {
    // Constructor
    pub fn create(ceed: &crate::Ceed, name: impl Into<String>) -> Self {
        let name_c = CString::new(name.into()).expect("CString::new failed");
        let mut ptr = std::ptr::null_mut();
        unsafe {
            bind_ceed::CeedQFunctionCreateInteriorByName(ceed.ptr, name_c.as_ptr(), &mut ptr)
        };
        let qf_core = QFunctionCore { ptr };
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
    /// let array = V.view();
    /// for i in 0..Q {
    ///   assert_eq!(array[i], v[i], "Incorrect value in QFunction application");
    /// }
    /// ```
    pub fn apply(&self, Q: i32, u: &Vec<crate::vector::Vector>, v: &Vec<crate::vector::Vector>) {
        self.qf_core.apply(Q, u, v)
    }
}

// -----------------------------------------------------------------------------
