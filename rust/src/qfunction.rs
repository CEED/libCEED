use crate::prelude::*;
use std::ffi::CString;
use std::fmt;

/// CeedQFunction context wrapper
pub struct QFunctionCore<'a> {
    ceed: &'a crate::Ceed,
    pub ptr: bind_ceed::CeedQFunction,
}
pub struct QFunction<'a> {
    pub qf_core: QFunctionCore<'a>,
}
pub struct QFunctionByName<'a> {
    pub qf_core: QFunctionCore<'a>,
}

/// Display
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

/// Core functionality
impl<'a> QFunctionCore<'a> {
    pub fn new(ceed: &'a crate::Ceed, ptr: bind_ceed::CeedQFunction) -> Self {
        Self { ceed, ptr }
    }

    pub fn apply(
        &self,
        Q: i32,
        input: [crate::vector::Vector; 16],
        output: [crate::vector::Vector; 16],
    ) {
        unsafe {
            let mut input_c = [std::ptr::null_mut(); 16];
            for i in 0..std::cmp::min(16, input.len()) {
                input_c[i] = input[i].ptr;
            }
            let mut output_c = [std::ptr::null_mut(); 16];
            for i in 0..std::cmp::min(16, output.len()) {
                output_c[i] = output[i].ptr;
            }
            bind_ceed::CeedQFunctionApply(self.ptr, Q, &mut input_c[0], &mut output_c[0]);
        }
    }
}

/// QFunction
impl<'a> QFunction<'a> {
    /// Constructor
    //pub fn create(ceed: &'a crate::Ceed, n: usize) -> Self {
    //
    //}

    pub fn apply(
        &self,
        Q: i32,
        input: [crate::vector::Vector; 16],
        output: [crate::vector::Vector; 16],
    ) {
        self.qf_core.apply(Q, input, output);
    }
}

/// QFunction by Name
impl<'a> QFunctionByName<'a> {
    /// Constructor
    pub fn create(ceed: &'a crate::Ceed, name: String) -> Self {
        let name_c = CString::new(name).expect("CString::new failed");
        let mut ptr = std::ptr::null_mut();
        unsafe {
            bind_ceed::CeedQFunctionCreateInteriorByName(ceed.ptr, name_c.as_ptr(), &mut ptr)
        };
        let qf_core = QFunctionCore::new(ceed, ptr);
        Self { qf_core }
    }

    pub fn apply(
        &self,
        Q: i32,
        input: [crate::vector::Vector; 16],
        output: [crate::vector::Vector; 16],
    ) {
        self.qf_core.apply(Q, input, output);
    }
}

/// Destructor
impl<'a> Drop for QFunctionCore<'a> {
    fn drop(&mut self) {
        unsafe {
            bind_ceed::CeedQFunctionDestroy(&mut self.ptr);
        }
    }
}
