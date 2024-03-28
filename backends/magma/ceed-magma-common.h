// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed.h>
#include <ceed/backend.h>
#include <magma_v2.h>

typedef struct {
  magma_device_t device_id;
  magma_queue_t  queue;
} Ceed_Magma;

CEED_INTERN int CeedInit_Magma_common(Ceed ceed, const char *resource);

CEED_INTERN int CeedDestroy_Magma(Ceed ceed);
