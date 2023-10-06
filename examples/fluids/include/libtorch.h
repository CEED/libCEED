// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <petsc.h>

#ifdef __cplusplus
extern "C" {
#endif
PetscErrorCode ModelInference_LibTorch(Vec DD_Inputs_loc, Vec DD_Outputs_loc);
PetscErrorCode LoadModel_LibTorch(const char *model_path);
PetscErrorCode CopyTest(Vec DD_Outputs_loc);
#ifdef __cplusplus
}
#endif

