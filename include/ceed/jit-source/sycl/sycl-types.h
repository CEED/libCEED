// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for SYCL type definitions
#ifndef CEED_SYCL_TYPES_H
#define CEED_SYCL_TYPES_H

#include <ceed/types.h>

#define CEED_SYCL_NUMBER_FIELDS 16

#ifdef __OPENCL_C_VERSION__
typedef struct {
  global const CeedScalar* inputs[CEED_SYCL_NUMBER_FIELDS];
  global CeedScalar*       outputs[CEED_SYCL_NUMBER_FIELDS];
} Fields_Sycl;

typedef struct {
  global const CeedInt* inputs[CEED_SYCL_NUMBER_FIELDS];
  global CeedInt*       outputs[CEED_SYCL_NUMBER_FIELDS];
} FieldsInt_Sycl;
#else
typedef struct {
  const CeedScalar* inputs[CEED_SYCL_NUMBER_FIELDS];
  CeedScalar*       outputs[CEED_SYCL_NUMBER_FIELDS];
} Fields_Sycl;

typedef struct {
  const CeedInt* inputs[CEED_SYCL_NUMBER_FIELDS];
  CeedInt*       outputs[CEED_SYCL_NUMBER_FIELDS];
} FieldsInt_Sycl;
#endif

#endif  // CEED_SYCL_TYPES_H
