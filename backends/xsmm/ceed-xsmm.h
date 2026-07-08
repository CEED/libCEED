// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed.h>
#include <ceed/backend.h>
#include <libxsmm.h>

#if !LIBXSMM_VERSION_GE(2, 0, 0, 0)
#error "LIBXSMM 2.0 or later is required"
#endif

CEED_INTERN int CeedTensorContractCreate_Xsmm(CeedTensorContract contract);
