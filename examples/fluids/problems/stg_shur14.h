// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <petsc.h>
#include "../qfunctions/stg_shur14_type.h"

extern PetscErrorCode CreateSTGContext(const MPI_Comm comm, const DM dm,
                                       STGShur14Context *pstg_ctx,
                                       const NewtonianIdealGasContext newt_ctx,
                                       const bool is_implicit, const bool prescribe_T,
                                       const CeedScalar theta0, const CeedScalar P0);
