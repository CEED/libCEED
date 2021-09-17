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

use libceed::{prelude::*, Ceed};

// ----------------------------------------------------------------------------
// Determine problem size in each dimension from size and dimenison
// ----------------------------------------------------------------------------
pub fn cartesian_mesh_size(dim: usize, solution_degree: usize, problem_size: i64) -> [usize; 3] {
    // Use the approximate formula:
    //    prob_size ~ num_elem * degree^dim
    let mut num_elem = problem_size / solution_degree.pow(dim as u32) as i64;
    let mut s = 0; // find s: num_elem / 2 < 2^s <= num_elem
    while num_elem > 1 {
        num_elem /= 2;
        s += 1;
    }

    // Size per dimension
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
pub fn build_cartesian_restriction(
    ceed: &Ceed,
    dim: usize,
    num_xyz: [usize; 3],
    degree: usize,
    num_comp: usize,
    num_qpts: usize,
) -> libceed::Result<(ElemRestriction, ElemRestriction)> {
    let p = degree + 1;
    let num_nodes = p.pow(dim as u32); // number of nodes per element
    let elem_qpts = num_qpts.pow(dim as u32); // number of quadrature pts per element

    // Problem dimensions
    let mut num_d = [0; 3];
    let mut num_elem = 1;
    let mut scalar_size = 1;
    for d in 0..dim {
        num_elem *= num_xyz[d];
        num_d[d] = num_xyz[d] * (p - 1) + 1;
        scalar_size *= num_d[d];
    }

    // elem:         0             1                 n-1
    //        |---*-...-*---|---*-...-*---|- ... -|--...--|
    // nodes: 0   1    p-1  p  p+1       2*p             n*p
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
    )?;

    // Quadratue data restriction
    let restr_qdata = ceed.strided_elem_restriction(
        num_elem,
        elem_qpts,
        num_comp,
        num_comp * elem_qpts * num_elem,
        CEED_STRIDES_BACKEND,
    )?;
    Ok((restr, restr_qdata))
}

// ----------------------------------------------------------------------------
// Set mesh coordinates
// ----------------------------------------------------------------------------
pub fn cartesian_mesh_coords(
    ceed: &Ceed,
    dim: usize,
    num_xyz: [usize; 3],
    mesh_degree: usize,
    mesh_size: usize,
) -> libceed::Result<Vector> {
    let p = mesh_degree + 1;
    let mut num_d = [0; 3];
    let mut scalar_size = 1;
    for d in 0..dim {
        num_d[d] = num_xyz[d] * (p - 1) + 1;
        scalar_size *= num_d[d];
    }

    // Lobatto points
    let lobatto_basis = ceed.basis_tensor_H1_Lagrange(1, 1, 2, p, QuadMode::GaussLobatto)?;
    let nodes_corners = ceed.vector_from_slice(&[0.0, 1.0])?;
    let mut nodes_full = ceed.vector(p)?;
    lobatto_basis.apply(
        1,
        TransposeMode::NoTranspose,
        EvalMode::Interp,
        &nodes_corners,
        &mut nodes_full,
    )?;

    // Coordinates for mesh
    let mut mesh_coords = ceed.vector(mesh_size)?;
    {
        let mut coords = mesh_coords.view_mut()?;
        let nodes = nodes_full.view()?;
        for gs_nodes in 0..scalar_size {
            let mut r_nodes = gs_nodes;
            for d in 0..dim {
                let d_1d = r_nodes % num_d[d];
                coords[gs_nodes + scalar_size * d] =
                    ((d_1d / (p - 1)) as Scalar + nodes[d_1d % (p - 1)]) / num_xyz[d] as Scalar;
                r_nodes /= num_d[d];
            }
        }
    }
    Ok(mesh_coords)
}

// ----------------------------------------------------------------------------
