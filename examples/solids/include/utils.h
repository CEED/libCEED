// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef libceed_solids_examples_utils_h
#define libceed_solids_examples_utils_h

#include <ceed.h>
#include <petsc.h>

// Translate PetscMemType to CeedMemType
static inline CeedMemType MemTypeP2C(PetscMemType mem_type) { return PetscMemTypeDevice(mem_type) ? CEED_MEM_DEVICE : CEED_MEM_HOST; }

#endif  // libceed_solids_examples_utils_h
