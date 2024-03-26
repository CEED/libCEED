// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed.h>
#include <petscdmplex.h>
#include <petscsnes.h>
#include <stdbool.h>
#include <string.h>

#include "include/cl-options.h"
#include "include/matops.h"
#include "include/misc.h"
#include "include/setup-dm.h"
#include "include/setup-libceed.h"
#include "include/structs.h"
#include "include/utils.h"
#include "problems/problems.h"

#if PETSC_VERSION_LT(3, 20, 0)
#error "PETSc v3.20 or later is required"
#endif

#if PETSC_VERSION_LT(3, 21, 0)
#define DMSetCoordinateDisc(a, b, c) DMProjectCoordinates(a, b)
#endif
