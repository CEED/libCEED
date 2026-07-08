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

CEED_EXTERN int CeedCheckFilePath(Ceed ceed, const char *source_file_path, bool *is_valid);
CEED_EXTERN int CeedGetJitRelativePath(const char *absolute_file_path, const char **relative_file_path);
CEED_EXTERN int CeedGetJitAbsolutePath(Ceed ceed, const char *relative_file_path, const char **absolute_file_path);
