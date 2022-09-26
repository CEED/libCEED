// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for HIP type definitions
#ifndef _ceed_hip_types_h
#define _ceed_hip_types_h

#include <ceed/types.h>

#define CEED_HIP_NUMBER_FIELDS 16

typedef struct {
  const CeedScalar* inputs[CEED_HIP_NUMBER_FIELDS];
  CeedScalar*       outputs[CEED_HIP_NUMBER_FIELDS];
} Fields_Hip;

typedef struct {
  CeedInt* inputs[CEED_HIP_NUMBER_FIELDS];
  CeedInt* outputs[CEED_HIP_NUMBER_FIELDS];
} FieldsInt_Hip;

typedef struct {
  CeedInt     t_id_x;
  CeedInt     t_id_y;
  CeedInt     t_id_z;
  CeedInt     t_id;
  CeedScalar* slice;
} SharedData_Hip;

#endif
