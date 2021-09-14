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

use libceed::prelude::*;

// ----------------------------------------------------------------------------
// Transform mesh coordinates
// ----------------------------------------------------------------------------
pub(crate) fn transform_mesh_coordinates(
    dim: usize,
    mesh_size: usize,
    mesh_coords: &mut Vector,
) -> libceed::Result<Scalar> {
    // Transform coordinates
    if dim == 1 {
        mesh_coords.view_mut()?.iter_mut().for_each(|coord| {
            // map [0,1] to [0,1] varying the mesh density
            *coord = 0.5
                + 1.0 / (3.0 as Scalar).sqrt()
                    * ((2.0 / 3.0) * std::f64::consts::PI as Scalar * (*coord - 0.5)).sin()
        });
    } else {
        let mut coords = mesh_coords.view_mut()?;
        let num_nodes = mesh_size / dim;
        for i in 0..num_nodes {
            // map (x,y) from [0,1]x[0,1] to the quarter annulus with polar
            // coordinates, (r,phi) in [1,2]x[0,pi/2] with area = 3/4*pi
            let u = 1.0 + coords[i];
            let v = std::f64::consts::PI as Scalar / 2.0 * coords[i + num_nodes];
            coords[i] = u * v.cos();
            coords[i + num_nodes] = u * v.sin();
        }
    }

    // Exact volume of transformed region
    let exact_volume = match dim {
        1 => 1.0,
        _ => 3.0 / 4.0 * std::f64::consts::PI as Scalar,
    };
    Ok(exact_volume)
}

// ----------------------------------------------------------------------------
