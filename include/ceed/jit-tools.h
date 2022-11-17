/// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
/// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
/// reserved. See files LICENSE and NOTICE for details.
///
/// This file is part of CEED, a collection of benchmarks, miniapps, software
/// libraries and APIs for efficient high-order finite element and spectral
/// element discretizations for exascale applications. For more information and
/// source code availability see http://github.com/ceed
///
/// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
/// a collaborative effort of two U.S. Department of Energy organizations (Office
/// of Science and the National Nuclear Security Administration) responsible for
/// the planning and preparation of a capable exascale ecosystem, including
/// software, applications, hardware, advanced system engineering and early
/// testbed platforms, in support of the nation's exascale computing imperative.

#ifndef _ceed_jit_h
#define _ceed_jit_h

#include <ceed/ceed.h>

CEED_EXTERN int CeedCheckFilePath(Ceed ceed, const char *source_file_path, bool *is_valid);
CEED_EXTERN int CeedLoadSourceToBuffer(Ceed ceed, const char *source_file_path, char **buffer);
CEED_EXTERN int CeedLoadSourceToInitializedBuffer(Ceed ceed, const char *source_file_path, char **buffer);
CEED_EXTERN int CeedPathConcatenate(Ceed ceed, const char *base_file_path, const char *relative_file_path, char **new_file_path);
CEED_EXTERN int CeedGetJitRelativePath(const char *absolute_file_path, const char **relative_file_path);
CEED_EXTERN int CeedGetJitAbsolutePath(Ceed ceed, const char *relative_file_path, char **absolute_file_path);

#endif
