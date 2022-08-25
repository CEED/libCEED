// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <petsc.h>

#include "../navierstokes.h"
#include "../qfunctions/stg_shur14_type.h"

extern PetscErrorCode SetupSTG(const MPI_Comm comm, const DM dm, ProblemData *problem, User user, const bool prescribe_T, const CeedScalar theta0,
                               const CeedScalar P0, const CeedScalar ynodes[], const CeedInt nynodes);

extern PetscErrorCode SetupStrongSTG(DM dm, SimpleBC bc, ProblemData *problem, Physics phys);

extern PetscErrorCode SetupStrongSTG_QF(Ceed ceed, ProblemData *problem, CeedInt num_comp_x, CeedInt num_comp_q, CeedInt stg_data_size,
                                        CeedInt q_data_size_sur, CeedQFunction *qf_strongbc);

extern PetscErrorCode SetupStrongSTG_PreProcessing(Ceed ceed, ProblemData *problem, CeedInt num_comp_x, CeedInt stg_data_size,
                                                   CeedInt q_data_size_sur, CeedQFunction *pqf_strongbc);
