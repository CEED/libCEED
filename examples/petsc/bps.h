// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
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

#ifndef bps_h
#define bps_h

// -----------------------------------------------------------------------------
// Command Line Options
// -----------------------------------------------------------------------------

// MemType Options
static const char *const mem_types[] = {"host","device", "memType",
                                        "CEED_MEM_", 0
                                       };

// Coarsening options
typedef enum {
  COARSEN_UNIFORM = 0, COARSEN_LOGARITHMIC = 1
} CoarsenType;
static const char *const coarsen_types [] = {"uniform", "logarithmic",
                                             "CoarsenType", "COARSEN", 0
                                            };

static const char *const bp_types[] = {"bp1", "bp2", "bp3", "bp4", "bp5", "bp6",
                                       "BPType", "CEED_BP", 0
                                      };

#endif // bps_h
