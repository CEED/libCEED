// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/types.h>

/// A structure used to pass additional data to f_build_mass
struct BuildContext {
  CeedInt dim, space_dim;
};

// References the rust file for the qfunction named build_mass_rs
CEED_QFUNCTION_RUST(build_mass)

// References the rust file for the qfunction named apply_mass_rs
CEED_QFUNCTION_RUST(apply_mass)
