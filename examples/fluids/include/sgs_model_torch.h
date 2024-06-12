// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <petsc.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  TORCH_DEVICE_CPU,
  TORCH_DEVICE_CUDA,
  TORCH_DEVICE_HIP,
  TORCH_DEVICE_XPU,
} TorchDeviceType;
static const char *const TorchDeviceTypes[] = {"cpu", "cuda", "hip", "xpu", "TorchDeviceType", "TORCH_DEVICE_", NULL};

PetscErrorCode LoadModel_Torch(const char *model_path, TorchDeviceType device_enum);
PetscErrorCode ModelInference_Torch(Vec DD_Inputs_loc, Vec DD_Outputs_loc);

#ifdef __cplusplus
}
#endif
