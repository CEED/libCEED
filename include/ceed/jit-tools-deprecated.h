/// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
/// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
///
/// SPDX-License-Identifier: BSD-2-Clause
///
/// This file is part of CEED:  http://github.com/ceed

/// @file
/// Public header for JiT utility components of libCEED
#pragma once

#include <ceed.h>

CEED_EXTERN int CeedLoadSourceToBuffer(Ceed ceed, const char *source_file_path, char **buffer);
CEED_EXTERN int CeedQFunctionLoadSourceToBuffer(CeedQFunction qf, const char **source_buffer);
