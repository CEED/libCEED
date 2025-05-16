// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed/backend.h>

#undef CEED_BACKEND
#define CEED_BACKEND(name, suffix, ...) CEED_INTERN int CeedRegister_##name##suffix(void);
#include "../backends/ceed-backend-list.h"
#undef CEED_BACKEND
