// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CUDA type definitions
#pragma once

#include <ceed/types.h>

#define CEED_CUDA_NUMBER_FIELDS 16

typedef struct {
  const CeedScalar *inputs[CEED_CUDA_NUMBER_FIELDS];
  CeedScalar       *outputs[CEED_CUDA_NUMBER_FIELDS];
} Fields_Cuda;

typedef struct {
  CeedInt *inputs[CEED_CUDA_NUMBER_FIELDS];
  CeedInt *outputs[CEED_CUDA_NUMBER_FIELDS];
} FieldsInt_Cuda;

typedef struct {
  CeedInt           num_elem;
  const CeedInt    *num_per_elem;
  const CeedInt    *indices;
  const CeedScalar *coords;
} Points_Cuda;

typedef struct {
  CeedInt     t_id_x;
  CeedInt     t_id_y;
  CeedInt     t_id_z;
  CeedInt     t_id;
  CeedScalar *slice;
} SharedData_Cuda;
