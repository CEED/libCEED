// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef _helper_h
#define _helper_h

#include <ceed.h>

#include "t406-qfunction-scales.h"

CEED_QFUNCTION_HELPER CeedScalar times_two(CeedScalar x) { return SCALE_TWO * x; }

CEED_QFUNCTION_HELPER CeedScalar times_three(CeedScalar x) { return SCALE_THREE * x; }

#endif
