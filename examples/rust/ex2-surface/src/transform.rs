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
    mesh_coords: &mut libceed::Vector,
) -> libceed::Result<libceed::Scalar> {
    // Transform coordinates
    for coord in mesh_coords.view_mut()?.iter_mut() {
        // map [0,1] to [0,1] varying the mesh density
        *coord = 0.5
            + 1.0 / (3.0 as libceed::Scalar).sqrt()
                * ((2.0 / 3.0) * std::f64::consts::PI as libceed::Scalar * (*coord - 0.5)).sin()
    }

    // Exact surface area of transformed region
    let exact_area = match dim {
        1 => 2.0,
        2 => 4.0,
        3 => 6.0,
        _ => unreachable!(),
    };
    Ok(exact_area)
}

// ----------------------------------------------------------------------------
