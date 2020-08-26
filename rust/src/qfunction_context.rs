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
// CeedQFunctionContext context wrapper
// -----------------------------------------------------------------------------
pub struct QFunctionContext<'a> {
    ceed: &'a crate::Ceed,
    pub ptr: bind_ceed::CeedQFunctionContext,
}

// -----------------------------------------------------------------------------
// Destructor
// -----------------------------------------------------------------------------
impl<'a> Drop for QFunctionContext<'a> {
    fn drop(&mut self) {
        unsafe {
            bind_ceed::CeedQFunctionContextDestroy(&mut self.ptr);
        }
    }
}

// -----------------------------------------------------------------------------
// Display
// -----------------------------------------------------------------------------
impl<'a> fmt::Display for QFunctionContext<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut ptr = std::ptr::null_mut();
        let mut sizeloc = 202020;
        unsafe {
            let file = bind_ceed::open_memstream(&mut ptr, &mut sizeloc);
            bind_ceed::CeedQFunctionContextView(self.ptr, file);
            bind_ceed::fclose(file);
            let cstring = CString::from_raw(ptr);
            let s = cstring.to_string_lossy().into_owned();
            write!(f, "{}", s)
        }
    }
}

// -----------------------------------------------------------------------------
// Implementations
// -----------------------------------------------------------------------------
impl<'a> QFunctionContext<'a> {
    // Constructor
    pub fn create(ceed: &'a crate::Ceed) -> Self {
        let mut ptr = std::ptr::null_mut();
        unsafe { bind_ceed::CeedQFunctionContextCreate(ceed.ptr, &mut ptr) };
        Self { ceed, ptr }
    }
}

// -----------------------------------------------------------------------------
