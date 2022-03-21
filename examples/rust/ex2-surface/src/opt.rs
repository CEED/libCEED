// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

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
    #[structopt(name = "quiet mode", short = "x", long = "quiet")]
    pub(crate) quiet: bool,
    /// Gallery QFunctions
    #[structopt(name = "gallery QFunctions", short = "g", long = "gallery")]
    pub(crate) gallery: bool,
}

// ----------------------------------------------------------------------------
