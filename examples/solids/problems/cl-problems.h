// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

// Problem options
typedef enum { ELAS_LINEAR = 0, ELAS_FS_NH = 2, ELAS_FS_MR = 2 } problemType;
static const char *const problemTypes[]        = {"Linear", "FS-NH", "FS-MR", "problemType", "ELAS_", 0};
static const char *const problemTypesForDisp[] = {"Linear elasticity", "Hyperelasticity finite strain Initial configuration Neo-Hookean",
                                                  "Hyperelasticity finite strain Initial configuration Moony-Rivlin"};
