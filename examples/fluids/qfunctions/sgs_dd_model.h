// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Structs and helper functions for data-driven subgrid-stress modeling

#ifndef sgs_dd_model_h
#define sgs_dd_model_h

#include <ceed.h>

typedef struct SGS_DD_ModelContext_ *SGS_DDModelContext;
struct SGS_DD_ModelContext_ {
  CeedInt    num_inputs, num_outputs;
  CeedInt    num_layers;
  CeedInt    num_neurons;
  CeedScalar alpha;

  struct {
    size_t bias1, bias2;
    size_t weight1, weight2;
    size_t out_scaling;
  } offsets;
  size_t     total_bytes;
  CeedScalar data[1];
};

#endif  // sgs_dd_model_h
