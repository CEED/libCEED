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

//! # libCEED Rust Interface
//!
//! This is the documentation for the low level Rust bindings to the libCEED C interface.
//! See the full libCEED user manual [here](https://libceed.readthedocs.io).
//!
//! libCEED is a low-level API for for the efficient high-order discretization methods
//! developed by the ECP co-design Center for Efficient Exascale Discretizations (CEED).
//! While our focus is on high-order finite elements, the approach is mostly algebraic
//! and thus applicable to other discretizations in factored form.
//!
//! ## Usage
//!
//! To use low level libCEED bindings in a Rust package, the following `Cargo.toml`
//! can be used.
//! ```toml
//! [dependencies]
//! libceed-sys = { git = "https://github.com/CEED/libCEED", branch = "main" }
//! ```
//!
//! Supported features:
//! * `static` (default): link to static libceed.a
//! * `system`: use libceed from a system directory (otherwise, install from source)
//!
//! ## Development
//!
//! To develop libCEED, use `cargo build` in the `rust/libceed-sys` directory to
//! install a local copy and build the bindings. If you need custom flags for the
//! C project, we recommend using `make configure` to cache arguments. If you
//! disable the `static` feature, then you'll need to set `LD_LIBRARY_PATH` for
//! doctests to be able to find it. You can do this in `$CEED_DIR/lib` and set
//! `PKG_CONFIG_PATH`.
//!
//! Note: the `LD_LIBRARY_PATH` workarounds will become unnecessary if [this
//! issue](https://github.com/rust-lang/cargo/issues/1592) is resolved -- it's
//! currently closed, but the problem still exists.

pub mod bind_ceed {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(dead_code)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
