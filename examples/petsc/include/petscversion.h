// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Petsc version check

#ifndef libceed_petsc_examples_version_h
#define libceed_petsc_examples_version_h

#if PETSC_VERSION_LT(3, 20, 0)
#error "PETSc v3.20 or later is required"
#endif

#endif
