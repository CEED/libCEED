// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CPU JiT type definitions
#pragma once

#include <ceed/types.h>

typedef struct {
  // CeedVector
  const CeedScalar *l_vec;
  // CeedElemRestriction
  const CeedInt  *offsets;
  const bool     *orients;
  const CeedInt8 *curl_orients;
  // CeedBasis
  const CeedScalar *weights;
  const CeedScalar *interp;
  const CeedScalar *grad;
  const CeedScalar *div;
  const CeedScalar *curl;
} InputFieldData_Cpu_Gen;

typedef struct {
  // CeedVector
  CeedScalar *l_vec;
  // CeedElemRestriction
  const CeedInt  *offsets;
  const bool     *orients;
  const CeedInt8 *curl_orients;
  // CeedBasis
  const CeedScalar *weights;
  const CeedScalar *interp;
  const CeedScalar *grad;
  const CeedScalar *div;
  const CeedScalar *curl;
} OutputFieldData_Cpu_Gen;

typedef int (*CeedOperatorFunction_Cpu_Gen)(void *, InputFieldData_Cpu_Gen *, OutputFieldData_Cpu_Gen *);
