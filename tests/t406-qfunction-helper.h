// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

// Note: testing 'pragma once'
// clang-format off
# pragma  once
// clang-format on

// Note - ceed/types.h should be used over ceed.h
#include <ceed.h>

// Test include path with "/./"
#include "./t406-qfunction-scales.h"

// Test include via -I....
#include <fake-sys-include.h>

CEED_QFUNCTION_HELPER CeedScalar times_two(CeedScalar x) { return FAKE_SYS_SCALE_ONE * SCALE_TWO * x; }

CEED_QFUNCTION_HELPER CeedScalar times_three(CeedScalar x) { return FAKE_SYS_SCALE_ONE * SCALE_THREE * x; }
