// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
//
//                             libCEED Example 1
//
// This example illustrates a simple usage of libCEED to compute the volume of a
// 3D body using matrix-free application of a mass + diff operator.  Arbitrary
// mesh and solution orders in 1D, 2D and 3D are supported from the same code.
// This calculation is executed in triplicate with a 3 component vector system.
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
// Example 1
// ----------------------------------------------------------------------------
fn main() -> libceed::Result<()> {
    let options = opt::Opt::parse();
    example_3_vector(options)
}

#[allow(clippy::erasing_op)]
#[allow(clippy::identity_op)]
fn example_3_vector(options: opt::Opt) -> libceed::Result<()> {
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
    } = options;
    assert!((1..=3).contains(&dim));
    assert!(mesh_degree >= 1);
    assert!(solution_degree >= 1);
    assert!(num_qpts >= 1);
    let ncomp_x = dim;
    let problem_size: i64 = if problem_size_requested < 0 {
        if test {
            8 * 16
        } else {
            256 * 1024
        }
    } else {
        problem_size_requested
    };
    let ncomp_u = 3;

    // Summary output
    if !quiet {
        println!("Selected options: [command line option] : <current value>");
        println!("    Ceed specification [-c] : {}", ceed_spec);
        println!("    Mesh dimension     [-d] : {}", dim);
        println!("    Mesh degree        [-m] : {}", mesh_degree);
        println!("    Solution degree    [-p] : {}", solution_degree);
        println!("    Num. 1D quadr. pts [-q] : {}", num_qpts);
        println!("    Approx. # unknowns [-s] : {}", problem_size);
        println!("    QFunction source        : user closure");
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
        ncomp_u,
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
        1 + dim * (dim + 1) / 2,
        num_qpts,
    )?;
    let (rstr_solution, _) =
        mesh::build_cartesian_restriction(&ceed, dim, num_xyz, solution_degree, ncomp_u, num_qpts)?;
    let mesh_size = rstr_mesh.lvector_size();
    let solution_size = rstr_solution.lvector_size();
    if !quiet {
        println!("Number of mesh nodes        : {}", mesh_size / dim);
        println!("Number of solution nodes    : {}", solution_size);
    }

    // Create a Vector with the mesh coordinates
    let mut mesh_coords = mesh::cartesian_mesh_coords(&ceed, dim, num_xyz, mesh_degree, mesh_size)?;

    // Apply a transformation to the mesh coordinates
    let exact_volume = transform::transform_mesh_coordinates(dim, mesh_size, &mut mesh_coords)?;

    // QFunction that builds the quadrature data for the mass + diff operator
    // -- QFunction from user closure
    let build_mass_diff = move |[jacobian, weights, ..]: QFunctionInputs,
                                [qdata, ..]: QFunctionOutputs| {
        // Build quadrature data
        match dim {
            1 => {
                let q = qdata.len() / 2;
                for i in 0..q {
                    // Mass
                    qdata[i + q * 0] = weights[i] * jacobian[i];
                    // Diff
                    qdata[i + q * 1] = weights[i] / jacobian[i];
                }
            }
            2 => {
                let q = qdata.len() / 4;
                for i in 0..q {
                    let j11 = jacobian[i + q * 0];
                    let j21 = jacobian[i + q * 1];
                    let j12 = jacobian[i + q * 2];
                    let j22 = jacobian[i + q * 3];
                    // Mass
                    qdata[i + q * 0] = weights[i] * (j11 * j22 - j21 * j12);
                    // Diff
                    let qw = weights[i] / (j11 * j22 - j21 * j12);
                    qdata[i + q * 1] = qw * (j12 * j12 + j22 * j22);
                    qdata[i + q * 2] = qw * (j11 * j11 + j21 * j21);
                    qdata[i + q * 3] = -qw * (j11 * j12 + j21 * j22);
                }
            }
            3 => {
                let q = qdata.len() / 7;
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
                    // Mass
                    qdata[i + q * 0] = weights[i]
                        * (jacobian[i + q * 0] * a[0 * 3 + 0]
                            + jacobian[i + q * 1] * a[0 * 3 + 1]
                            + jacobian[i + q * 2] * a[0 * 3 + 2]);
                    let qw = weights[i]
                        / (jacobian[i + q * 0] * a[0 * 3 + 0]
                            + jacobian[i + q * 1] * a[0 * 3 + 1]
                            + jacobian[i + q * 2] * a[0 * 3 + 2]);
                    // Diff
                    qdata[i + q * 1] = qw
                        * (a[0 * 3 + 0] * a[0 * 3 + 0]
                            + a[0 * 3 + 1] * a[0 * 3 + 1]
                            + a[0 * 3 + 2] * a[0 * 3 + 2]);
                    qdata[i + q * 2] = qw
                        * (a[1 * 3 + 0] * a[1 * 3 + 0]
                            + a[1 * 3 + 1] * a[1 * 3 + 1]
                            + a[1 * 3 + 2] * a[1 * 3 + 2]);
                    qdata[i + q * 3] = qw
                        * (a[2 * 3 + 0] * a[2 * 3 + 0]
                            + a[2 * 3 + 1] * a[2 * 3 + 1]
                            + a[2 * 3 + 2] * a[2 * 3 + 2]);
                    qdata[i + q * 4] = qw
                        * (a[1 * 3 + 0] * a[2 * 3 + 0]
                            + a[1 * 3 + 1] * a[2 * 3 + 1]
                            + a[1 * 3 + 2] * a[2 * 3 + 2]);
                    qdata[i + q * 5] = qw
                        * (a[0 * 3 + 0] * a[2 * 3 + 0]
                            + a[0 * 3 + 1] * a[2 * 3 + 1]
                            + a[0 * 3 + 2] * a[2 * 3 + 2]);
                    qdata[i + q * 6] = qw
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
        .q_function_interior(1, Box::new(build_mass_diff))?
        .input("dx", ncomp_x * dim, libceed::EvalMode::Grad)?
        .input("weights", 1, libceed::EvalMode::Weight)?
        .output("qdata", 1 + dim * (dim + 1) / 2, libceed::EvalMode::None)?;
    // -- QFunction for use with Operator
    let qf_build = QFunctionOpt::SomeQFunction(&qf_build_closure);

    // Operator that build the quadrature data for the mass + diff operator
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

    // Compute the quadrature data for the mass + diff operator
    let elem_qpts = num_qpts.pow(dim as u32);
    let num_elem: usize = num_xyz.iter().take(dim).product();
    let mut qdata = ceed.vector(num_elem * elem_qpts * (1 + dim * (dim + 1) / 2))?;
    op_build.apply(&mesh_coords, &mut qdata)?;

    // QFunction that applies the mass + diff operator
    // -- QFunction from user closure
    let apply_mass_diff = move |[u, ug, qdata, ..]: QFunctionInputs,
                                [v, vg, ..]: QFunctionOutputs| {
        // Apply diffusion operator
        match dim {
            1 => {
                let q = qdata.len() / 2;
                for i in 0..q {
                    for c in 0..ncomp_u {
                        // Mass
                        v[i + c * q] = u[i + c * q] * qdata[i + 0 * q];
                        // Diff
                        vg[i + c * q] = ug[i + c * q] * qdata[i + 1 * q];
                    }
                }
            }
            2 => {
                let q = qdata.len() / 4;
                for i in 0..q {
                    let dxdxdxdx_t = [
                        [qdata[i + 1 * q], qdata[i + 3 * q]],
                        [qdata[i + 3 * q], qdata[i + 2 * q]],
                    ];
                    for c in 0..ncomp_u {
                        // Mass
                        v[i + c * q] = u[i + c * q] * qdata[i + 0 * q];
                        // Diff
                        let du = [ug[i + (c + 0 * ncomp_u) * q], ug[i + (c + 1 * ncomp_u) * q]];
                        for j in 0..2 {
                            vg[i + (j + j * ncomp_u) * q] =
                                du[0] * dxdxdxdx_t[0][j] + du[1] * dxdxdxdx_t[1][j];
                        }
                    }
                }
            }
            3 => {
                let q = qdata.len() / 7;
                for i in 0..q {
                    let dxdxdxdx_t = [
                        [qdata[i + 1 * q], qdata[i + 6 * q], qdata[i + 5 * q]],
                        [qdata[i + 6 * q], qdata[i + 2 * q], qdata[i + 4 * q]],
                        [qdata[i + 5 * q], qdata[i + 4 * q], qdata[i + 3 * q]],
                    ];
                    for c in 0..ncomp_u {
                        // Mass
                        v[i + c * q] = u[i + c * q] * qdata[i + 0 * q];
                        // Diff
                        let du = [
                            ug[i + (c + 0 * ncomp_u) * q],
                            ug[i + (c + 1 * ncomp_u) * q],
                            ug[i + (c + 2 * ncomp_u) * q],
                        ];
                        for j in 0..3 {
                            vg[i + (c + j * ncomp_u) * q] = du[0] * dxdxdxdx_t[0][j]
                                + du[1] * dxdxdxdx_t[1][j]
                                + du[2] * dxdxdxdx_t[2][j];
                        }
                    }
                }
            }
            _ => unreachable!(),
        };

        // Return clean error code
        0
    };
    let qf_mass_diff_closure = ceed
        .q_function_interior(1, Box::new(apply_mass_diff))?
        .input("u", ncomp_u, libceed::EvalMode::Interp)?
        .input("du", dim * ncomp_u, libceed::EvalMode::Grad)?
        .input("qdata", 1 + dim * (dim + 1) / 2, libceed::EvalMode::None)?
        .output("v", ncomp_u, libceed::EvalMode::Interp)?
        .output("dv", dim * ncomp_u, libceed::EvalMode::Grad)?;
    // -- QFunction for use with Operator
    let qf_mass_diff = QFunctionOpt::SomeQFunction(&qf_mass_diff_closure);

    // Mass + diff Operator
    let op_mass_diff = ceed
        .operator(qf_mass_diff, QFunctionOpt::None, QFunctionOpt::None)?
        .name("mass diff")?
        .field("u", &rstr_solution, &basis_solution, VectorOpt::Active)?
        .field("du", &rstr_solution, &basis_solution, VectorOpt::Active)?
        .field("qdata", &rstr_qdata, BasisOpt::None, &qdata)?
        .field("v", &rstr_solution, &basis_solution, VectorOpt::Active)?
        .field("dv", &rstr_solution, &basis_solution, VectorOpt::Active)?
        .check()?;

    // Solution vectors
    let mut u = ceed.vector(solution_size)?;
    let mut v = ceed.vector(solution_size)?;

    // Initialize u with component index
    u.set_value(0.0)?;
    for c in 0..ncomp_u {
        let q = solution_size / ncomp_u;
        u.view_mut()?.iter_mut().skip(c * q).take(q).for_each(|u| {
            *u = (c + 1) as libceed::Scalar;
        });
    }

    // Apply the mass + diff operator
    op_mass_diff.apply(&u, &mut v)?;

    // Compute the mesh volume
    let volume: libceed::Scalar = v.view()?.iter().sum::<libceed::Scalar>()
        / ((ncomp_u * (ncomp_u + 1)) / 2) as libceed::Scalar;

    // Output results
    if !quiet {
        println!("Exact mesh volume           : {:.12}", exact_volume);
        println!("Computed mesh volume        : {:.12}", volume);
        println!(
            "Volume error                : {:.12e}",
            volume - exact_volume
        );
    }
    let tolerance = match dim {
        1 => 200.0 * libceed::EPSILON,
        _ => 1E-5,
    };
    let error = (volume - exact_volume).abs();
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
    fn example_3_vector_1d() {
        let options = opt::Opt {
            ceed_spec: "/cpu/self/ref/serial".to_string(),
            dim: 1,
            mesh_degree: 4,
            solution_degree: 4,
            num_qpts: 6,
            problem_size_requested: -1,
            test: true,
            quiet: true,
        };
        assert!(example_3_vector(options).is_ok());
    }

    #[test]
    fn example_3_vector_2d() {
        let options = opt::Opt {
            ceed_spec: "/cpu/self/ref/serial".to_string(),
            dim: 2,
            mesh_degree: 4,
            solution_degree: 4,
            num_qpts: 6,
            problem_size_requested: -1,
            test: true,
            quiet: true,
        };
        assert!(example_3_vector(options).is_ok());
    }

    #[test]
    fn example_3_vector_vector_3d() {
        let options = opt::Opt {
            ceed_spec: "/cpu/self/ref/serial".to_string(),
            dim: 3,
            mesh_degree: 4,
            solution_degree: 4,
            num_qpts: 6,
            problem_size_requested: -1,
            test: true,
            quiet: false,
        };
        assert!(example_3_vector(options).is_ok());
    }
}

// ----------------------------------------------------------------------------
