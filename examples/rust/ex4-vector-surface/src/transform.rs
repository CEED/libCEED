// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

use libceed::prelude::*;

// ----------------------------------------------------------------------------
// Transform mesh coordinates
// ----------------------------------------------------------------------------
pub(crate) fn transform_mesh_coordinates(
    dim: usize,
    mesh_coords: &mut Vector,
) -> libceed::Result<Scalar> {
    // Transform coordinates
    for coord in mesh_coords.view_mut()?.iter_mut() {
        // map [0,1] to [0,1] varying the mesh density
        *coord = 0.5
            + 1.0 / (3.0 as Scalar).sqrt()
                * ((2.0 / 3.0) * std::f64::consts::PI as Scalar * (*coord - 0.5)).sin()
    }

    // Exact surface area of transformed region
    let exact_area = match dim {
        1 => 2.0,
        2 => 4.0,
        _ => 6.0,
    };
    Ok(exact_area)
}

// ----------------------------------------------------------------------------
