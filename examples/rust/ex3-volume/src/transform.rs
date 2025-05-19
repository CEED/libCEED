// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

// ----------------------------------------------------------------------------
// Transform mesh coordinates
// ----------------------------------------------------------------------------
pub(crate) fn transform_mesh_coordinates(
    dim: usize,
    mesh_size: usize,
    mesh_coords: &mut libceed::Vector,
) -> libceed::Result<libceed::Scalar> {
    // Transform coordinates
    match dim {
        1 => {
            for coord in mesh_coords.view_mut()?.iter_mut() {
                // map [0,1] to [0,1] varying the mesh density
                *coord = 0.5
                    + 1.0 / (3.0 as libceed::Scalar).sqrt()
                        * ((2.0 / 3.0) * std::f64::consts::PI as libceed::Scalar * (*coord - 0.5))
                            .sin()
            }
        }
        _ => {
            let num_nodes = mesh_size / dim;
            let mut coords = mesh_coords.view_mut()?;
            for i in 0..num_nodes {
                // map (x,y) from [0,1]x[0,1] to the quarter annulus with polar
                // coordinates, (r,phi) in [1,2]x[0,pi/2] with area = 3/4*pi
                let u = coords[i] + 1.;
                let v = coords[i + num_nodes] * std::f64::consts::PI / 2.;
                coords[i] = u * v.cos();
                coords[i + num_nodes] = u * v.sin();
            }
        }
    }

    // Exact volume of transformed region
    let exact_volume = match dim {
        1 => 1.,
        2 | 3 => 3. / 4. * std::f64::consts::PI,
        _ => unreachable!(),
    };
    Ok(exact_volume)
}

// ----------------------------------------------------------------------------
