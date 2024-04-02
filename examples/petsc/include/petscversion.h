// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Petsc version check
#pragma once

#if PETSC_VERSION_LT(3, 21, 0)
#error "PETSc v3.21 or later is required"
#endif
