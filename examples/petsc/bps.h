// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef libceed_petsc_examples_bps_h
#define libceed_petsc_examples_bps_h

// -----------------------------------------------------------------------------
// Command Line Options
// -----------------------------------------------------------------------------

// MemType Options
static const char *const mem_types[] = {"host", "device", "memType", "CEED_MEM_", 0};

// Coarsening options
typedef enum { COARSEN_UNIFORM = 0, COARSEN_LOGARITHMIC = 1 } CoarsenType;
static const char *const coarsen_types[] = {"uniform", "logarithmic", "CoarsenType", "COARSEN", 0};

static const char *const bp_types[] = {"bp1", "bp2", "bp3", "bp4", "bp5", "bp6", "BPType", "CEED_BP", 0};

#endif  // libceed_petsc_examples_bps_h
