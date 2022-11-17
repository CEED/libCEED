// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef libceed_solids_examples_cl_options_h
#define libceed_solids_examples_cl_options_h

#include <petsc.h>

#include "../include/structs.h"

// Process general command line options
PetscErrorCode ProcessCommandLineOptions(MPI_Comm comm, AppCtx app_ctx);

#endif  // libceed_solids_examples_cl_options_h
