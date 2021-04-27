// Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
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
// testbed platforms, in support of the nation's exascale computing imperative.

use structopt::StructOpt;

// ----------------------------------------------------------------------------
// Command line arguments
// ----------------------------------------------------------------------------
#[derive(Debug, StructOpt)]
#[structopt(
    name = "libCEED Rust Example 2 - Surface Area",
    about = "This example illustrates a simple usage of libCEED to compute the surface area of a body using matrix-free application of a diffusion operator."
)]
#[cfg(not(tarpaulin_include))]
pub(crate) struct Opt {
    /// libCEED backend resource to use
    #[structopt(name = "ceed", short = "c", long = "ceed", default_value = "/cpu/self")]
    pub(crate) ceed_spec: String,
    /// Mesh dimension
    #[structopt(
        name = "dimension",
        short = "d",
        long = "dimension",
        default_value = "3"
    )]
    pub(crate) dim: usize,
    /// Polynomial degree for the mesh
    #[structopt(
        name = "mesh degree",
        short = "m",
        long = "mesh_degree",
        default_value = "4"
    )]
    pub(crate) mesh_degree: usize,
    /// Polynomial degree for the solution
    #[structopt(
        name = "solution degree",
        short = "p",
        long = "solution_degree",
        default_value = "4"
    )]
    pub(crate) solution_degree: usize,
    /// Number of quadrature points in 1D
    #[structopt(
        name = "number of quadrature points",
        short = "q",
        long = "num_qpts",
        default_value = "6"
    )]
    pub(crate) num_qpts: usize,
    /// Approximate problem size
    #[structopt(
        name = "problem size",
        short = "s",
        long = "problem_size",
        default_value = "-1"
    )]
    pub(crate) problem_size_requested: i64,
    /// Test mode
    #[structopt(name = "test mode", short = "t", long = "test")]
    pub(crate) test: bool,
    /// Quiet mode
    #[structopt(name = "quiet mode", short = "q", long = "quiet")]
    pub(crate) quiet: bool,
    /// Gallery QFunctions
    #[structopt(name = "gallery QFunctions", short = "g", long = "gallery")]
    pub(crate) gallery: bool,
}

// ----------------------------------------------------------------------------
