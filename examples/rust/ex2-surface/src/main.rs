// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
//
//                             libCEED Example 2
//
// This example illustrates a simple usage of libCEED to compute the surface
// area of a 3D body using matrix-free application of a diffusion operator.
// Arbitrary mesh and solution degrees in 1D, 2D and 3D are supported from the
// same code.
//
// The example has no dependencies, and is designed to be self-contained. For
// additional examples that use external discretization libraries (MFEM, PETSc,
// etc.) see the subdirectories in libceed/examples.
//
// All libCEED objects use a Ceed device object constructed based on a command
// line argument (-ceed).

use clap::Parser;
use libceed::{
    BasisOpt, Ceed, ElemRestrictionOpt, QFunctionInputs, QFunctionOpt, QFunctionOutputs, VectorOpt,
};
mod opt;
mod transform;

// ----------------------------------------------------------------------------
// Example 2
// ----------------------------------------------------------------------------
fn main() -> libceed::Result<()> {
    let options = opt::Opt::parse();
    example_2(options)
}

#[allow(clippy::erasing_op)]
#[allow(clippy::identity_op)]
fn example_2(options: opt::Opt) -> libceed::Result<()> {
    // Process command line arguments
    let opt::Opt {
        ceed_spec,
        dim,
        mesh_degree,
        solution_degree,
        num_qpts,
        problem_size_requested,
        test,
        quiet,
        gallery,
    } = options;
    assert!((0..=3).contains(&dim));
    assert!(mesh_degree >= 1);
    assert!(solution_degree >= 1);
    assert!(num_qpts >= 1);
    let ncomp_x = dim;
    let problem_size: i64 = if problem_size_requested < 0 {
        if test {
            16 * 16 * (dim * dim) as i64
        } else {
            256 * 1024
        }
    } else {
        problem_size_requested
    };

    // Summary output
    if !quiet {
        println!("Selected options: [command line option] : <current value>");
        println!("    Ceed specification [-c] : {}", ceed_spec);
        println!("    Mesh dimension     [-d] : {}", dim);
        println!("    Mesh degree        [-m] : {}", mesh_degree);
        println!("    Solution degree    [-p] : {}", solution_degree);
        println!("    Num. 1D quadr. pts [-q] : {}", num_qpts);
        println!("    Approx. # unknowns [-s] : {}", problem_size);
        println!(
            "    QFunction source   [-g] : {}",
            if gallery { "gallery" } else { "user closure" }
        );
    }

    // Initalize ceed context
    let ceed = Ceed::init(&ceed_spec);

    // Mesh and solution bases
    let basis_mesh = ceed.basis_tensor_H1_Lagrange(
        dim,
        ncomp_x,
        mesh_degree + 1,
        num_qpts,
        libceed::QuadMode::Gauss,
    )?;
    let basis_solution = ceed.basis_tensor_H1_Lagrange(
        dim,
        1,
        solution_degree + 1,
        num_qpts,
        libceed::QuadMode::Gauss,
    )?;

    // Determine mesh size from approximate problem size
    let num_xyz = mesh::cartesian_mesh_size(dim, solution_degree, problem_size);
    if !quiet {
        print!("\nMesh size                   : nx = {}", num_xyz[0]);
        if dim > 1 {
            print!(", ny = {}", num_xyz[1]);
        }
        if dim > 2 {
            print!(", nz = {}", num_xyz[2]);
        }
        println!();
    }

    // Build ElemRestriction objects describing the mesh and solution discrete
    // representations
    let (rstr_mesh, _) =
        mesh::build_cartesian_restriction(&ceed, dim, num_xyz, mesh_degree, ncomp_x, num_qpts)?;
    let (_, rstr_qdata) = mesh::build_cartesian_restriction(
        &ceed,
        dim,
        num_xyz,
        solution_degree,
        dim * (dim + 1) / 2,
        num_qpts,
    )?;
    let (rstr_solution, _) =
        mesh::build_cartesian_restriction(&ceed, dim, num_xyz, solution_degree, 1, num_qpts)?;
    let mesh_size = rstr_mesh.lvector_size();
    let solution_size = rstr_solution.lvector_size();
    if !quiet {
        println!("Number of mesh nodes        : {}", mesh_size / dim);
        println!("Number of solution nodes    : {}", solution_size);
    }

    // Create a Vector with the mesh coordinates
    let mut mesh_coords = mesh::cartesian_mesh_coords(&ceed, dim, num_xyz, mesh_degree, mesh_size)?;

    // Apply a transformation to the mesh coordinates
    let exact_area = transform::transform_mesh_coordinates(dim, &mut mesh_coords)?;

    // QFunction that builds the quadrature data for the diff operator
    // -- QFunction from user closure
    let build_diff = move |[jacobian, weights, ..]: QFunctionInputs,
                           [qdata, ..]: QFunctionOutputs| {
        // Build quadrature data
        match dim {
            1 => qdata
                .iter_mut()
                .zip(jacobian.iter().zip(weights.iter()))
                .for_each(|(qdata, (j, weight))| *qdata = weight / j),
            2 => {
                let q = qdata.len() / 3;
                for i in 0..q {
                    let j11 = jacobian[i + q * 0];
                    let j21 = jacobian[i + q * 1];
                    let j12 = jacobian[i + q * 2];
                    let j22 = jacobian[i + q * 3];
                    let qw = weights[i] / (j11 * j22 - j21 * j12);
                    qdata[i + q * 0] = qw * (j12 * j12 + j22 * j22);
                    qdata[i + q * 1] = qw * (j11 * j11 + j21 * j21);
                    qdata[i + q * 2] = -qw * (j11 * j12 + j21 * j22);
                }
            }
            3 => {
                let q = qdata.len() / 6;
                for i in 0..q {
                    let mut a = [0.0; 9];
                    for j in 0..3 {
                        for k in 0..3 {
                            a[k * 3 + j] = jacobian[i + q * ((j + 1) % 3 + 3 * ((k + 1) % 3))]
                                * jacobian[i + q * ((j + 2) % 3 + 3 * ((k + 2) % 3))]
                                - jacobian[i + q * ((j + 1) % 3 + 3 * ((k + 2) % 3))]
                                    * jacobian[i + q * ((j + 2) % 3 + 3 * ((k + 1) % 3))];
                        }
                    }
                    let qw = weights[i]
                        / (jacobian[i + q * 0] * a[0 * 3 + 0]
                            + jacobian[i + q * 1] * a[0 * 3 + 1]
                            + jacobian[i + q * 2] * a[0 * 3 + 2]);
                    qdata[i + q * 0] = qw
                        * (a[0 * 3 + 0] * a[0 * 3 + 0]
                            + a[0 * 3 + 1] * a[0 * 3 + 1]
                            + a[0 * 3 + 2] * a[0 * 3 + 2]);
                    qdata[i + q * 1] = qw
                        * (a[1 * 3 + 0] * a[1 * 3 + 0]
                            + a[1 * 3 + 1] * a[1 * 3 + 1]
                            + a[1 * 3 + 2] * a[1 * 3 + 2]);
                    qdata[i + q * 2] = qw
                        * (a[2 * 3 + 0] * a[2 * 3 + 0]
                            + a[2 * 3 + 1] * a[2 * 3 + 1]
                            + a[2 * 3 + 2] * a[2 * 3 + 2]);
                    qdata[i + q * 3] = qw
                        * (a[1 * 3 + 0] * a[2 * 3 + 0]
                            + a[1 * 3 + 1] * a[2 * 3 + 1]
                            + a[1 * 3 + 2] * a[2 * 3 + 2]);
                    qdata[i + q * 4] = qw
                        * (a[0 * 3 + 0] * a[2 * 3 + 0]
                            + a[0 * 3 + 1] * a[2 * 3 + 1]
                            + a[0 * 3 + 2] * a[2 * 3 + 2]);
                    qdata[i + q * 5] = qw
                        * (a[0 * 3 + 0] * a[1 * 3 + 0]
                            + a[0 * 3 + 1] * a[1 * 3 + 1]
                            + a[0 * 3 + 2] * a[1 * 3 + 2]);
                }
            }
            _ => unreachable!(),
        };

        // Return clean error code
        0
    };
    let qf_build_closure = ceed
        .q_function_interior(1, Box::new(build_diff))?
        .input("dx", ncomp_x * dim, libceed::EvalMode::Grad)?
        .input("weights", 1, libceed::EvalMode::Weight)?
        .output("qdata", dim * (dim + 1) / 2, libceed::EvalMode::None)?;
    // -- QFunction from gallery
    let qf_build_named = {
        let name = format!("Poisson{}DBuild", dim);
        ceed.q_function_interior_by_name(&name)?
    };
    // -- QFunction for use with Operator
    let qf_build = if gallery {
        QFunctionOpt::SomeQFunctionByName(&qf_build_named)
    } else {
        QFunctionOpt::SomeQFunction(&qf_build_closure)
    };

    // Operator that build the quadrature data for the diff operator
    let op_build = ceed
        .operator(qf_build, QFunctionOpt::None, QFunctionOpt::None)?
        .name("build qdata")?
        .field("dx", &rstr_mesh, &basis_mesh, VectorOpt::Active)?
        .field(
            "weights",
            ElemRestrictionOpt::None,
            &basis_mesh,
            VectorOpt::None,
        )?
        .field("qdata", &rstr_qdata, BasisOpt::None, VectorOpt::Active)?
        .check()?;

    // Compute the quadrature data for the diff operator
    let elem_qpts = num_qpts.pow(dim as u32);
    let num_elem: usize = num_xyz.iter().take(dim).product();
    let mut qdata = ceed.vector(num_elem * elem_qpts * dim * (dim + 1) / 2)?;
    op_build.apply(&mesh_coords, &mut qdata)?;

    // QFunction that applies the diff operator
    // -- QFunction from user closure
    let apply_diff = move |[ug, qdata, ..]: QFunctionInputs, [vg, ..]: QFunctionOutputs| {
        // Apply diffusion operator
        match dim {
            1 => vg
                .iter_mut()
                .zip(ug.iter().zip(qdata.iter()))
                .for_each(|(vg, (ug, w))| *vg = ug * w),
            2 => {
                let q = qdata.len() / 3;
                for i in 0..q {
                    let du = [ug[i + q * 0], ug[i + q * 1]];
                    let dxdxdxdx_t = [
                        [qdata[i + 0 * q], qdata[i + 2 * q]],
                        [qdata[i + 2 * q], qdata[i + 1 * q]],
                    ];
                    for j in 0..2 {
                        vg[i + j * q] = du[0] * dxdxdxdx_t[0][j] + du[1] * dxdxdxdx_t[1][j];
                    }
                }
            }
            3 => {
                let q = qdata.len() / 6;
                for i in 0..q {
                    let du = [ug[i + q * 0], ug[i + q * 1], ug[i + q * 2]];
                    let dxdxdxdx_t = [
                        [qdata[i + 0 * q], qdata[i + 5 * q], qdata[i + 4 * q]],
                        [qdata[i + 5 * q], qdata[i + 1 * q], qdata[i + 3 * q]],
                        [qdata[i + 4 * q], qdata[i + 3 * q], qdata[i + 2 * q]],
                    ];
                    for j in 0..3 {
                        vg[i + j * q] = du[0] * dxdxdxdx_t[0][j]
                            + du[1] * dxdxdxdx_t[1][j]
                            + du[2] * dxdxdxdx_t[2][j];
                    }
                }
            }
            _ => unreachable!(),
        };

        // Return clean error code
        0
    };
    let qf_diff_closure = ceed
        .q_function_interior(1, Box::new(apply_diff))?
        .input("du", dim, libceed::EvalMode::Grad)?
        .input("qdata", dim * (dim + 1) / 2, libceed::EvalMode::None)?
        .output("dv", dim, libceed::EvalMode::Grad)?;
    // -- QFunction from gallery
    let qf_diff_named = {
        let name = format!("Poisson{}DApply", dim);
        ceed.q_function_interior_by_name(&name)?
    };
    // -- QFunction for use with Operator
    let qf_diff = if gallery {
        QFunctionOpt::SomeQFunctionByName(&qf_diff_named)
    } else {
        QFunctionOpt::SomeQFunction(&qf_diff_closure)
    };

    // Diff Operator
    let op_diff = ceed
        .operator(qf_diff, QFunctionOpt::None, QFunctionOpt::None)?
        .name("Poisson")?
        .field("du", &rstr_solution, &basis_solution, VectorOpt::Active)?
        .field("qdata", &rstr_qdata, BasisOpt::None, &qdata)?
        .field("dv", &rstr_solution, &basis_solution, VectorOpt::Active)?
        .check()?;

    // Solution vectors
    let mut u = ceed.vector(solution_size)?;
    let mut v = ceed.vector(solution_size)?;

    // Initialize u with sum of node coordinates
    let coords = mesh_coords.view()?;
    u.set_value(0.0)?;
    for (i, u) in u.view_mut()?.iter_mut().enumerate() {
        *u = (0..dim).map(|d| coords[i + d * solution_size]).sum();
    }

    // Apply the diff operator
    op_diff.apply(&u, &mut v)?;

    // Compute the mesh surface area
    let area: libceed::Scalar = v.view()?.iter().map(|v| (*v).abs()).sum();

    // Output results
    if !quiet {
        println!("Exact mesh surface area     : {:.12}", exact_area);
        println!("Computed mesh surface_area  : {:.12}", area);
        println!("Surface area error          : {:.12e}", area - exact_area);
    }
    let tolerance = match dim {
        1 => 10000.0 * libceed::EPSILON,
        _ => 1E-1,
    };
    let error = (area - exact_area).abs();
    if error > tolerance {
        println!("Volume error too large: {:.12e}", error);
        return Err(libceed::Error {
            message: format!(
                "Volume error too large - expected: {:.12e}, actual: {:.12e}",
                tolerance, error
            ),
        });
    }
    Ok(())
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_2_1d() {
        let options = opt::Opt {
            ceed_spec: "/cpu/self/ref/serial".to_string(),
            dim: 1,
            mesh_degree: 4,
            solution_degree: 4,
            num_qpts: 6,
            problem_size_requested: -1,
            test: true,
            quiet: true,
            gallery: false,
        };
        assert!(example_2(options).is_ok());
    }

    #[test]
    fn example_2_2d() {
        let options = opt::Opt {
            ceed_spec: "/cpu/self/ref/serial".to_string(),
            dim: 2,
            mesh_degree: 4,
            solution_degree: 4,
            num_qpts: 6,
            problem_size_requested: -1,
            test: true,
            quiet: true,
            gallery: false,
        };
        assert!(example_2(options).is_ok());
    }

    #[test]
    fn example_2_3d() {
        let options = opt::Opt {
            ceed_spec: "/cpu/self/ref/serial".to_string(),
            dim: 3,
            mesh_degree: 4,
            solution_degree: 4,
            num_qpts: 6,
            problem_size_requested: -1,
            test: true,
            quiet: false,
            gallery: false,
        };
        assert!(example_2(options).is_ok());
    }

    #[test]
    fn example_2_1d_gallery() {
        let options = opt::Opt {
            ceed_spec: "/cpu/self/ref/serial".to_string(),
            dim: 1,
            mesh_degree: 4,
            solution_degree: 4,
            num_qpts: 6,
            problem_size_requested: -1,
            test: true,
            quiet: true,
            gallery: true,
        };
        assert!(example_2(options).is_ok());
    }

    #[test]
    fn example_2_2d_gallery() {
        let options = opt::Opt {
            ceed_spec: "/cpu/self/ref/serial".to_string(),
            dim: 2,
            mesh_degree: 4,
            solution_degree: 4,
            num_qpts: 6,
            problem_size_requested: -1,
            test: true,
            quiet: true,
            gallery: true,
        };
        assert!(example_2(options).is_ok());
    }

    #[test]
    fn example_2_3d_gallery() {
        let options = opt::Opt {
            ceed_spec: "/cpu/self/ref/serial".to_string(),
            dim: 3,
            mesh_degree: 4,
            solution_degree: 4,
            num_qpts: 6,
            problem_size_requested: -1,
            test: true,
            quiet: true,
            gallery: true,
        };
        assert!(example_2(options).is_ok());
    }
}

// ----------------------------------------------------------------------------
