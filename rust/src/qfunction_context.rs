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
// CeedQFunctionContext context wrapper
// -----------------------------------------------------------------------------
pub struct QFunctionContext {
    pub(crate) ptr: bind_ceed::CeedQFunctionContext,
    pub(crate) ceed: crate::Ceed,
}

// -----------------------------------------------------------------------------
// Destructor
// -----------------------------------------------------------------------------
impl Drop for QFunctionContext {
    fn drop(&mut self) {
        unsafe {
            bind_ceed::CeedQFunctionContextDestroy(&mut self.ptr);
        }
    }
}

// -----------------------------------------------------------------------------
// Display
// -----------------------------------------------------------------------------
impl fmt::Display for QFunctionContext {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut ptr = std::ptr::null_mut();
        let mut sizeloc = crate::MAX_BUFFER_LENGTH;
        let file = unsafe { bind_ceed::open_memstream(&mut ptr, &mut sizeloc) };
        unsafe { bind_ceed::CeedQFunctionContextView(self.ptr, file) };
        unsafe { bind_ceed::fclose(file) };
        let cstring = unsafe { CString::from_raw(ptr) };
        cstring.to_string_lossy().fmt(f)
    }
}

// -----------------------------------------------------------------------------
// Implementations
// -----------------------------------------------------------------------------
impl QFunctionContext {
    // Constructor
    pub fn create(ceed: & crate::Ceed) -> Self {
        let mut ptr = std::ptr::null_mut();
        unsafe { bind_ceed::CeedQFunctionContextCreate(ceed.core.ptr, &mut ptr) };
        Self { ptr, ceed: ceed.clone() }
    }
}

// -----------------------------------------------------------------------------
