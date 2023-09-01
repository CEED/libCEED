/// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
/// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
///
/// SPDX-License-Identifier: BSD-2-Clause
///
/// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_JIT_H
#define CEED_JIT_H

#include <ceed.h>

CEED_EXTERN int CeedCheckFilePath(Ceed ceed, const char *source_file_path, bool *is_valid);
CEED_EXTERN int CeedLoadSourceToBuffer(Ceed ceed, const char *source_file_path, char **buffer);
CEED_EXTERN int CeedLoadSourceToInitializedBuffer(Ceed ceed, const char *source_file_path, char **buffer);
CEED_EXTERN int CeedPathConcatenate(Ceed ceed, const char *base_file_path, const char *relative_file_path, char **new_file_path);
CEED_EXTERN int CeedGetJitRelativePath(const char *absolute_file_path, const char **relative_file_path);
CEED_EXTERN int CeedGetJitAbsolutePath(Ceed ceed, const char *relative_file_path, char **absolute_file_path);

#endif  // CEED_JIT_H
