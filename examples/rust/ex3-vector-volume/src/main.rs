// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
//
//                             libCEED Example 3
//
// This example illustrates a simple usage of libCEED to compute the volume of a
// 3D body using matrix-free application of a mass operator.  Arbitrary mesh and
// solution orders in 1D, 2D and 3D are supported from the same code. This
// calculation is executed in triplicate with a 3 component vector system.
//
// The example has no dependencies, and is designed to be self-contained. For
// additional examples that use external discretization libraries (MFEM, PETSc,
// etc.) see the subdirectories in libceed/examples.
//
// All libCEED objects use a Ceed device object constructed based on a command
// line argument (-ceed).

use clap::Parser;
use libceed::{prelude::*, Ceed};
mod opt;
mod transform;

// ----------------------------------------------------------------------------
// Example 3
// ----------------------------------------------------------------------------
#[cfg(not(tarpaulin_include))]
fn main() -> libceed::Result<()> {
    let options = opt::Opt::parse();
    example_3(options)
}

fn example_3(options: opt::Opt) -> libceed::Result<()> {
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
    assert!(dim >= 1 && dim <= 3);
    assert!(mesh_degree >= 1);
    assert!(solution_degree >= 1);
    assert!(num_qpts >= 1);
    let ncomp_x = dim;
    let problem_size: i64;
    if problem_size_requested < 0 {
        problem_size = if test { 8 * 16 } else { 256 * 1024 };
    } else {
        problem_size = problem_size_requested;
    }
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
        println!(
            "    QFunction source   [-g] : {}",
            if gallery { "gallery" } else { "user closure" }
        );
    }

    // Initalize ceed context
    let ceed = Ceed::init(&ceed_spec);

    // Mesh and solution bases
    let basis_mesh =
        ceed.basis_tensor_H1_Lagrange(dim, ncomp_x, mesh_degree + 1, num_qpts, QuadMode::Gauss)?;
    let basis_solution = ceed.basis_tensor_H1_Lagrange(
        dim,
        ncomp_u,
        solution_degree + 1,
        num_qpts,
        QuadMode::Gauss,
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
        print!("\n");
    }

    // Build ElemRestriction objects describing the mesh and solution discrete
    // representations
    let (rstr_mesh, _) =
        mesh::build_cartesian_restriction(&ceed, dim, num_xyz, mesh_degree, ncomp_x, num_qpts)?;
    let (_, rstr_qdata) =
        mesh::build_cartesian_restriction(&ceed, dim, num_xyz, solution_degree, 1, num_qpts)?;
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

    // QFunction that builds the quadrature data for the mass operator
    // -- QFunction from user closure
    let build_mass = move |[jacobian, weights, ..]: QFunctionInputs,
                           [qdata, ..]: QFunctionOutputs| {
        // Build quadrature data
        match dim {
            1 => qdata
                .iter_mut()
                .zip(jacobian.iter().zip(weights.iter()))
                .for_each(|(qdata, (j, weight))| *qdata = j * weight),
            2 => {
                let q = qdata.len();
                qdata.iter_mut().zip(weights.iter()).enumerate().for_each(
                    |(i, (qdata, weight))| {
                        *qdata = (jacobian[i + q * 0] * jacobian[i + q * 3]
                            - jacobian[i + q * 1] * jacobian[i + q * 2])
                            * weight
                    },
                );
            }
            3 => {
                let q = qdata.len();
                qdata.iter_mut().zip(weights.iter()).enumerate().for_each(
                    |(i, (qdata, weight))| {
                        *qdata = (jacobian[i + q * 0]
                            * (jacobian[i + q * 4] * jacobian[i + q * 8]
                                - jacobian[i + q * 5] * jacobian[i + q * 7])
                            - jacobian[i + q * 1]
                                * (jacobian[i + q * 3] * jacobian[i + q * 8]
                                    - jacobian[i + q * 5] * jacobian[i + q * 6])
                            + jacobian[i + q * 2]
                                * (jacobian[i + q * 3] * jacobian[i + q * 7]
                                    - jacobian[i + q * 4] * jacobian[i + q * 6]))
                            * weight
                    },
                );
            }
            _ => unreachable!(),
        };

        // Return clean error code
        0
    };
    let qf_build_closure = ceed
        .q_function_interior(1, Box::new(build_mass))?
        .input("dx", ncomp_x * dim, EvalMode::Grad)?
        .input("weights", 1, EvalMode::Weight)?
        .output("qdata", 1, EvalMode::None)?;
    // -- QFunction from gallery
    let qf_build_named = {
        let name = format!("Mass{}DBuild", dim);
        ceed.q_function_interior_by_name(&name)?
    };
    // -- QFunction for use with Operator
    let qf_build = if gallery {
        QFunctionOpt::SomeQFunctionByName(&qf_build_named)
    } else {
        QFunctionOpt::SomeQFunction(&qf_build_closure)
    };

    // Operator that build the quadrature data for the mass operator
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

    // Compute the quadrature data for the mass operator
    let elem_qpts = num_qpts.pow(dim as u32);
    let num_elem: usize = num_xyz.iter().take(dim).product();
    let mut qdata = ceed.vector(num_elem * elem_qpts)?;
    op_build.apply(&mesh_coords, &mut qdata)?;

    // QFunction that applies the mass operator
    // -- QFunction from user closure
    let apply_mass = move |[u, qdata, ..]: QFunctionInputs, [v, ..]: QFunctionOutputs| {
        let q = qdata.len();
        // Apply mass operator
        for c in 0..ncomp_u {
            v.iter_mut()
                .skip(c * q)
                .zip(u.iter().skip(c * q).zip(qdata.iter()))
                .for_each(|(v, (u, w))| *v = u * w);
        }
        // Return clean error code
        0
    };
    let qf_mass_closure = ceed
        .q_function_interior(1, Box::new(apply_mass))?
        .input("u", ncomp_u, EvalMode::Interp)?
        .input("qdata", 1, EvalMode::None)?
        .output("v", ncomp_u, EvalMode::Interp)?;
    // -- QFunction from gallery
    let qf_mass_named = ceed.q_function_interior_by_name("Vector3MassApply")?;
    // -- QFunction for use with Operator
    let qf_mass = if gallery {
        QFunctionOpt::SomeQFunctionByName(&qf_mass_named)
    } else {
        QFunctionOpt::SomeQFunction(&qf_mass_closure)
    };

    // Mass Operator
    let op_mass = ceed
        .operator(qf_mass, QFunctionOpt::None, QFunctionOpt::None)?
        .name("mass")?
        .field("u", &rstr_solution, &basis_solution, VectorOpt::Active)?
        .field("qdata", &rstr_qdata, BasisOpt::None, &qdata)?
        .field("v", &rstr_solution, &basis_solution, VectorOpt::Active)?
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

    // Apply the mass operator
    op_mass.apply(&u, &mut v)?;

    // Compute the mesh volume
    let volume: Scalar = v.view()?.iter().sum::<libceed::Scalar>()
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
        1 => 100.0 * libceed::EPSILON,
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
    fn example_3_1d() {
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
        assert!(example_3(options).is_ok());
    }

    #[test]
    fn example_3_2d() {
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
        assert!(example_3(options).is_ok());
    }

    #[test]
    fn example_3_3d() {
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
        assert!(example_3(options).is_ok());
    }

    #[test]
    fn example_3_1d_gallery() {
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
        assert!(example_3(options).is_ok());
    }

    #[test]
    fn example_3_2d_gallery() {
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
        assert!(example_3(options).is_ok());
    }

    #[test]
    fn example_3_3d_gallery() {
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
        assert!(example_3(options).is_ok());
    }
}

// ----------------------------------------------------------------------------
