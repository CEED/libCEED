// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef freestream_bc_type_h
#define freestream_bc_type_h

#include "newtonian_state.h"
#include "newtonian_types.h"

typedef struct FreestreamContext_ *FreestreamContext;
struct FreestreamContext_ {
  struct NewtonianIdealGasContext_ newtonian_ctx;
  State                            S_infty;
};

#endif
