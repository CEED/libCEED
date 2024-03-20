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

#ifndef bddc_h
#define bddc_h

#include <ceed.h>
#include <petsc.h>
#include <petscdmplex.h>
#include <petscfe.h>
#include <petscsys.h>
#include <stdbool.h>
#include <string.h>

#include "bps.h"
#include "include/bpsproblemdata.h"
#include "include/libceedsetup.h"
#include "include/matops.h"
#include "include/petscutils.h"
#include "include/structs.h"

#if PETSC_VERSION_LT(3, 12, 0)
#ifdef PETSC_HAVE_CUDA
#include <petsccuda.h>
// Note: With PETSc prior to version 3.12.0, providing the source path to
//       include 'cublas_v2.h' will be needed to use 'petsccuda.h'.
#endif
#endif

// -----------------------------------------------------------------------------
// Command Line Options
// -----------------------------------------------------------------------------

// Coarsening options
typedef enum {
  INJECTION_SCALED   = 0,
  INJECTION_HARMONIC = 1,
} InjectionType;
static const char *const injection_types[] = {
    "scaled", "harmonic", "InjectionType", "INJECTION", 0,
};

#endif  // bddc_h
