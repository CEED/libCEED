// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

use clap::Parser;

// ----------------------------------------------------------------------------
// Command line arguments
// ----------------------------------------------------------------------------
#[derive(Debug, Parser)]
#[command(
    name = "libCEED Rust Example 2 - Surface Area",
    about = "This example illustrates a simple usage of libCEED to compute the surface area of a body using matrix-free application of a diffusion operator."
)]
#[cfg(not(tarpaulin_include))]
pub(crate) struct Opt {
    /// libCEED backend resource to use
    #[arg(name = "CEED", short, long = "ceed", default_value = "/cpu/self")]
    pub(crate) ceed_spec: String,
    /// Mesh dimension
    #[arg(short, long = "dimension", default_value = "3")]
    pub(crate) dim: usize,
    /// Polynomial degree for the mesh
    #[arg(short, long, default_value = "4")]
    pub(crate) mesh_degree: usize,
    /// Polynomial degree for the solution
    #[arg(short = 'p', long, default_value = "4")]
    pub(crate) solution_degree: usize,
    /// Number of quadrature points in 1D
    #[arg(short = 'q', long, default_value = "6")]
    pub(crate) num_qpts: usize,
    /// Approximate problem size
    #[arg(name = "DoF", short = 's', long = "problem_size", default_value = "-1")]
    pub(crate) problem_size_requested: i64,
    /// Test mode
    #[arg(short, long)]
    pub(crate) test: bool,
    /// Quiet mode
    #[arg(short = 'x', long)]
    pub(crate) quiet: bool,
    /// Use QFunctions from the Gallery instead of example
    #[arg(short, long)]
    pub(crate) gallery: bool,
}

// ----------------------------------------------------------------------------
