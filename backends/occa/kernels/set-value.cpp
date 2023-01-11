// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "./kernel-defines.hpp"

// Expects the following types to be defined:
// - CeedInt
// - CeedScalar
//
// Expects the following constants to be defined:
// - BLOCK_SIZE : CeedInt

const char* occa_set_value_source = STRINGIFY_SOURCE(

    @kernel void setValue(CeedScalar* ptr, const CeedScalar value, const CeedInt count) {
      @tile(BLOCK_SIZE, @outer, @inner) for (CeedInt i = 0; i < count; ++i) {
        ptr[i] = value;
      }
    });
