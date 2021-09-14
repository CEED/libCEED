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
    mesh_coords: &mut Vector,
) -> libceed::Result<Scalar> {
    // Transform coordinates
    mesh_coords.view_mut()?.iter_mut().for_each(|coord| {
        // map [0,1] to [0,1] varying the mesh density
        *coord = 0.5
            + 1.0 / (3.0 as Scalar).sqrt()
                * ((2.0 / 3.0) * std::f64::consts::PI as Scalar * (*coord - 0.5)).sin()
    });

    // Exact surface area of transformed region
    let exact_area = match dim {
        1 => 2.0,
        2 => 4.0,
        _ => 6.0,
    };
    Ok(exact_area)
}

// ----------------------------------------------------------------------------
