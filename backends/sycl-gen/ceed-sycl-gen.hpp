// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <sycl/sycl.hpp>

#include "../sycl/ceed-sycl-common.hpp"
#include "../sycl/ceed-sycl-compile.hpp"

typedef struct {
  CeedInt         dim;
  CeedInt         Q_1d;
  CeedInt         max_P_1d;
  SyclModule_t   *sycl_module;
  sycl::kernel   *op;
  FieldsInt_Sycl *indices;
  Fields_Sycl    *fields;
  Fields_Sycl    *B;
  Fields_Sycl    *G;
  CeedScalar     *W;
} CeedOperator_Sycl_gen;

typedef struct {
  const char *qfunction_name;
  const char *qfunction_source;
  void       *d_c;
} CeedQFunction_Sycl_gen;

CEED_INTERN int CeedQFunctionCreate_Sycl_gen(CeedQFunction qf);

CEED_INTERN int CeedOperatorCreate_Sycl_gen(CeedOperator op);
