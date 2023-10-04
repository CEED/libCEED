// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
//
// @file This creates weak functions for libtorch dependent functions.

#include <sgs_model_torch.h>

PetscErrorCode LoadModel_Torch(const char *model_path, TorchDeviceType device_enum) __attribute__((weak));
PetscErrorCode LoadModel_Torch(const char *model_path, TorchDeviceType device_enum) {
  PetscFunctionBeginUser;
  SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Must build with USE_TORCH set to run %s", __func__);
}

PetscErrorCode ModelInference_Torch(Vec DD_Inputs_loc, Vec DD_Outputs_loc) __attribute__((weak));
PetscErrorCode ModelInference_Torch(Vec DD_Inputs_loc, Vec DD_Outputs_loc) {
  PetscFunctionBeginUser;
  SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Must build with USE_TORCH set to run %s", __func__);
}
