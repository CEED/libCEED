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

//                             libCEED Example 1
//
// This example illustrates a simple usage of libCEED to compute the volume of a
// 3D body using matrix-free application of a mass operator.  Arbitrary mesh and
// solution orders in 1D, 2D and 3D are supported from the same code.
//
// The example has no dependencies, and is designed to be self-contained. For
// additional examples that use external discretization libraries (MFEM, PETSc,
// etc.) see the subdirectories in libceed/examples.
//
// All libCEED objects use a Ceed device object constructed based on a command
// line argument (-ceed).

use libceed::{prelude::*, Ceed};
use structopt::StructOpt;

// ----------------------------------------------------------------------------
// Command line arguments
// ----------------------------------------------------------------------------
#[derive(Debug, StructOpt)]
#[structopt(
    name = "libCEED Rust Example 1 - Volume",
    about = "This example uses the mass matrix to compute the length, area, or volume of a region, depending upon runtime parameters."
)]
struct Opt {
    /// Set libCEED backend resource to use
    #[structopt(short = "c", long = "ceed", default_value = "/cpu/self")]
    ceed_spec: String,
    /// Set mesh dimension
    #[structopt(short = "d", long = "dimension", default_value = "3")]
    dim: usize,
    /// Set polynomial degree for the mesh
    #[structopt(short = "m", long = "mesh_degree", default_value = "4")]
    mesh_degree: usize,
    /// Set polynomial degree for the solution
    #[structopt(short = "p", long = "solution_degree", default_value = "4")]
    solution_degree: usize,
    /// Set number of quadrature points in 1D
    #[structopt(short = "q", long = "num_qpts", default_value = "6")]
    num_qpts: usize,
    /// Set approximate problem size
    #[structopt(short = "s", long = "problem_size", default_value = "-1")]
    problem_size_requested: i64,
    /// Enable test mode
    #[structopt(short = "t", long = "test")]
    test: bool,
    /// Use QFunctions from gallery
    #[structopt(short = "g", long = "gallery")]
    gallery: bool,
}

// ----------------------------------------------------------------------------
// Example 1
// ----------------------------------------------------------------------------
fn main() -> Result<(), String> {
    let opt = Opt::from_args();
    example_1(opt)
}

fn example_1(opt: Opt) -> Result<(), String> {
    // Process command line arguments
    let Opt {
        ceed_spec,
        dim,
        mesh_degree,
        solution_degree,
        num_qpts,
        problem_size_requested,
        test,
        gallery,
    } = opt;
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

    // Summary output
    if !test {
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
        ceed.basis_tensor_H1_Lagrange(dim, ncomp_x, mesh_degree + 1, num_qpts, QuadMode::Gauss);
    let basis_solution =
        ceed.basis_tensor_H1_Lagrange(dim, 1, solution_degree + 1, num_qpts, QuadMode::Gauss);

    // Determine mesh size from approximate problem size
    let num_xyz = get_cartesian_mesh_size(dim, solution_degree, problem_size);
    if !test {
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
    let (restr_mesh, _) =
        build_cartesian_restriction(&ceed, dim, num_xyz, mesh_degree, ncomp_x, num_qpts);
    let (restr_solution, restr_qdata) =
        build_cartesian_restriction(&ceed, dim, num_xyz, solution_degree, 1, num_qpts);
    let mesh_size = restr_mesh.get_lvector_size() as usize;
    let solution_size = restr_solution.get_lvector_size() as usize;
    if !test {
        println!("Number of mesh nodes        : {}", mesh_size / dim);
        println!("Number of solution nodes    : {}", solution_size);
    }

    // Create a Vector with the mesh coordinates
    let mut mesh_coords = set_cartesian_mesh_coords(&ceed, dim, num_xyz, mesh_degree, mesh_size);

    // Apply a transformation to the mesh coordinates
    let exact_volume = transform_mesh_coordinates(dim, mesh_size, &mut mesh_coords);

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
    let qf_build_closure = {
        let mut qfunction = ceed.q_function_interior(1, Box::new(build_mass));
        qfunction.add_input("dx", (ncomp_x * dim) as i32, EvalMode::Grad);
        qfunction.add_input("weights", 1, EvalMode::Weight);
        qfunction.add_output("qdata", 1, EvalMode::None);
        qfunction
    };

    // -- QFunction from gallery
    let qf_build_named = {
        let name = format!("Mass{}DBuild", dim);
        ceed.q_function_interior_by_name(&name)
    };
    // -- QFunction for use with Operator
    let qf_build = if gallery {
        QFunctionOpt::SomeQFunctionByName(&qf_build_named)
    } else {
        QFunctionOpt::SomeQFunction(&qf_build_closure)
    };

    // Operator that build the quadrature data for the mass operator
    let op_build = {
        let mut op = ceed.operator(qf_build, QFunctionOpt::None, QFunctionOpt::None);
        op.set_field("dx", &restr_mesh, &basis_mesh, VectorOpt::Active);
        op.set_field(
            "weights",
            ElemRestrictionOpt::None,
            &basis_mesh,
            VectorOpt::None,
        );
        op.set_field(
            "qdata",
            &restr_qdata,
            BasisOpt::Collocated,
            VectorOpt::Active,
        );
        op
    };

    // Compute the quadrature data for the mass operator
    let elem_qpts = num_qpts.pow(dim as u32);
    let num_elem: usize = num_xyz.iter().take(dim).product();
    let mut qdata = ceed.vector(num_elem * elem_qpts);
    op_build.apply(&mesh_coords, &mut qdata);

    // QFunction that applies the mass operator
    // -- QFunction from user closure
    let apply_mass = |[u, qdata, ..]: QFunctionInputs, [v, ..]: QFunctionOutputs| {
        // Apply mass operator
        v.iter_mut()
            .zip(u.iter().zip(qdata.iter()))
            .for_each(|(v, (u, w))| *v = u * w);
        // Return clean error code
        0
    };
    let qf_mass_closure = {
        let mut qfunction = ceed.q_function_interior(1, Box::new(apply_mass));
        qfunction.add_input("u", 1, EvalMode::Interp);
        qfunction.add_input("qdata", 1, EvalMode::None);
        qfunction.add_output("v", 1, EvalMode::Interp);
        qfunction
    };
    // -- QFunction from gallery
    let qf_mass_named = { ceed.q_function_interior_by_name("MassApply") };
    // -- QFunction for use with Operator
    let qf_mass = if gallery {
        QFunctionOpt::SomeQFunctionByName(&qf_mass_named)
    } else {
        QFunctionOpt::SomeQFunction(&qf_mass_closure)
    };

    // Mass Operator
    let op_mass = {
        let mut op = ceed.operator(qf_mass, QFunctionOpt::None, QFunctionOpt::None);
        op.set_field("u", &restr_solution, &basis_solution, VectorOpt::Active);
        op.set_field("qdata", &restr_qdata, BasisOpt::Collocated, &qdata);
        op.set_field("v", &restr_solution, &basis_solution, VectorOpt::Active);
        op
    };

    // Solution vectors
    let u = ceed.vector_from_slice(&vec![1.0; solution_size]);
    let mut v = ceed.vector(solution_size);

    // Apply the mass operator
    op_mass.apply(&u, &mut v);

    // Compute the mesh volume
    let volume: f64 = v.view().iter().sum();

    // Output results
    if !test {
        println!("Exact mesh volume           : {}", exact_volume);
        println!("Computed mesh volume        : {}", volume);
        println!("Volume error                : {}", volume - exact_volume);
    } else {
        let tolerance = if dim == 1 {
            1E-14
        } else if dim == 2 {
            1E-7
        } else {
            1E-5
        };
        let error = (volume - exact_volume).abs();
        if error > tolerance {
            println!("Volume error: {}", error);
            return Err(format!(
                "Volume error too large - expected: {}, actual: {}",
                tolerance, error
            ));
        }
    }
    Ok(())
}

// ----------------------------------------------------------------------------
// Determine problem size in each dimension from size and dimenison
// ----------------------------------------------------------------------------
fn get_cartesian_mesh_size(dim: usize, solution_degree: usize, problem_size: i64) -> [usize; 3] {
    // Use the approximate formula:
    //    prob_size ~ num_elem * degree^dim
    let mut num_elem = problem_size / solution_degree.pow(dim as u32) as i64;
    let mut s = 0; // find s: num_elem / 2 < 2^s <= num_elem

    while num_elem > 1 {
        num_elem /= 2;
        s += 1;
    }

    let mut r = s % dim;
    let mut num_xyz = [0; 3];
    for d in 0..dim {
        let mut sd = s / dim;
        if r > 0 {
            sd += 1;
            r -= 1;
        }
        num_xyz[d] = 1 << sd;
    }
    num_xyz
}

// ----------------------------------------------------------------------------
// Build element restriction objects for the mesh
// ----------------------------------------------------------------------------
fn build_cartesian_restriction(
    ceed: &Ceed,
    dim: usize,
    num_xyz: [usize; 3],
    degree: usize,
    num_comp: usize,
    num_qpts: usize,
) -> (ElemRestriction, ElemRestriction) {
    let p = degree + 1;
    let num_nodes = p.pow(dim as u32); // number of nodes per element
    let elem_qpts = num_qpts.pow(dim as u32); // number of quadrature pts per element

    let mut num_d = [0; 3];
    let mut num_elem = 1;
    let mut scalar_size = 1;
    for d in 0..dim {
        num_elem *= num_xyz[d];
        num_d[d] = num_xyz[d] * (p - 1) + 1;
        scalar_size *= num_d[d];
    }

    // elem:          0             1                 n-1
    //         |---*-...-*---|---*-...-*---|- ... -|--...--|
    // nnodes: 0   1    p-1  p  p+1       2*p             n*p
    let mut elem_nodes = vec![0; num_elem * num_nodes];
    for e in 0..num_elem {
        let mut e_xyz = [1; 3];
        let mut re = e;
        for d in 0..dim {
            e_xyz[d] = re % num_xyz[d];
            re /= num_xyz[d];
        }
        let loc_offset = e * num_nodes;
        for loc_nodes in 0..num_nodes {
            let mut global_nodes = 0;
            let mut global_nodes_stride = 1;
            let mut r_nodes = loc_nodes;
            for d in 0..dim {
                global_nodes += (e_xyz[d] * (p - 1) + r_nodes % p) * global_nodes_stride;
                global_nodes_stride *= num_d[d];
                r_nodes /= p;
            }
            elem_nodes[loc_offset + loc_nodes] = global_nodes as i32;
        }
    }

    // Mesh/solution data restriction
    let restr = ceed.elem_restriction(
        num_elem,
        num_nodes,
        num_comp,
        scalar_size,
        num_comp * scalar_size,
        MemType::Host,
        &elem_nodes,
    );
    // Quadratue data restriction
    let strides: [i32; 3] = [1, elem_qpts as i32, elem_qpts as i32];
    let restr_qdata = ceed.strided_elem_restriction(
        num_elem,
        elem_qpts,
        num_comp,
        num_comp * elem_qpts * num_elem,
        strides,
    );
    (restr, restr_qdata)
}

// ----------------------------------------------------------------------------
// Set mesh coordinates
// ----------------------------------------------------------------------------
fn set_cartesian_mesh_coords(
    ceed: &Ceed,
    dim: usize,
    num_xyz: [usize; 3],
    mesh_degree: usize,
    mesh_size: usize,
) -> Vector {
    let p = mesh_degree + 1;
    let mut num_d = [0; 3];
    let mut scalar_size = 1;
    for d in 0..dim {
        num_d[d] = num_xyz[d] * (p - 1) + 1;
        scalar_size *= num_d[d];
    }

    // Lobatto points
    let lobatto_basis = ceed.basis_tensor_H1_Lagrange(1, 1, 2, p, QuadMode::GaussLobatto);
    let nodes_corners = ceed.vector_from_slice(&[0.0, 1.0]);
    let mut nodes_full = ceed.vector(p);
    lobatto_basis.apply(
        1,
        TransposeMode::NoTranspose,
        EvalMode::Interp,
        &nodes_corners,
        &mut nodes_full,
    );

    let mut mesh_coords = ceed.vector(mesh_size);
    {
        let mut coords = mesh_coords.view_mut();
        let nodes = nodes_full.view();
        for gs_nodes in 0..scalar_size {
            let mut r_nodes = gs_nodes;
            for d in 0..dim {
                let d_1d = r_nodes % num_d[d];
                coords[gs_nodes + scalar_size * d] =
                    ((d_1d / (p - 1)) as f64 + nodes[d_1d % (p - 1)]) / num_xyz[d] as f64;
                r_nodes /= num_d[d];
            }
        }
    }
    mesh_coords
}

// ----------------------------------------------------------------------------
// Transform mesh coordinates
// ----------------------------------------------------------------------------
fn transform_mesh_coordinates(dim: usize, mesh_size: usize, mesh_coords: &mut Vector) -> f64 {
    let exact_volume = if dim == 1 {
        1.0
    } else {
        3.0 / 4.0 * std::f64::consts::PI
    };
    let mut coords = mesh_coords.view_mut();

    if dim == 1 {
        for i in 0..mesh_size {
            // map [0,1] to [0,1] varying the mesh density
            coords[i] = 0.5
                + 1.0 / (3.0_f64).sqrt()
                    * ((2.0 / 3.0) * std::f64::consts::PI * (coords[i] - 0.5)).sin();
        }
    } else {
        let num_nodes = mesh_size / dim;
        for i in 0..num_nodes {
            // map (x,y) from [0,1]x[0,1] to the quarter annulus with polar
            // coordinates, (r,phi) in [1,2]x[0,pi/2] with area = 3/4*pi
            let u = 1.0 + coords[i];
            let v = std::f64::consts::PI / 2.0 * coords[i + num_nodes];
            coords[i] = u * v.cos();
            coords[i + num_nodes] = u * v.sin();
        }
    }

    exact_volume
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_1_1d() {
        let ceed_spec = "/cpu/self/ref/serial".to_string();
        let dim = 1;
        let mesh_degree = 4;
        let solution_degree = 4;
        let num_qpts = 6;
        let problem_size_requested = -1;
        let test = true;
        let gallery = false;
        let opt = Opt {
            ceed_spec,
            dim,
            mesh_degree,
            solution_degree,
            num_qpts,
            problem_size_requested,
            test,
            gallery,
        };
        assert!(example_1(opt).is_ok());
    }

    #[test]
    fn test_example_1_2d() {
        let ceed_spec = "/cpu/self/ref/serial".to_string();
        let dim = 2;
        let mesh_degree = 4;
        let solution_degree = 4;
        let num_qpts = 6;
        let problem_size_requested = -1;
        let test = true;
        let gallery = false;
        let opt = Opt {
            ceed_spec,
            dim,
            mesh_degree,
            solution_degree,
            num_qpts,
            problem_size_requested,
            test,
            gallery,
        };
        assert!(example_1(opt).is_ok());
    }

    #[test]
    fn test_example_1_3d() {
        let ceed_spec = "/cpu/self/ref/serial".to_string();
        let dim = 3;
        let mesh_degree = 4;
        let solution_degree = 4;
        let num_qpts = 6;
        let problem_size_requested = -1;
        let test = true;
        let gallery = false;
        let opt = Opt {
            ceed_spec,
            dim,
            mesh_degree,
            solution_degree,
            num_qpts,
            problem_size_requested,
            test,
            gallery,
        };
        assert!(example_1(opt).is_ok());
    }

    #[test]
    fn test_example_1_2d_gallery() {
        let ceed_spec = "/cpu/self/ref/serial".to_string();
        let dim = 2;
        let mesh_degree = 4;
        let solution_degree = 4;
        let num_qpts = 6;
        let problem_size_requested = -1;
        let test = true;
        let gallery = true;
        let opt = Opt {
            ceed_spec,
            dim,
            mesh_degree,
            solution_degree,
            num_qpts,
            problem_size_requested,
            test,
            gallery,
        };
        assert!(example_1(opt).is_ok());
    }
}

// ----------------------------------------------------------------------------
