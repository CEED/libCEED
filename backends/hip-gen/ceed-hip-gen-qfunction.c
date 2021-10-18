// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <string.h>
#include "ceed-hip-gen.h"
#include "../hip/ceed-hip.h"

//------------------------------------------------------------------------------
// Apply QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionApply_Hip_gen(CeedQFunction qf, CeedInt Q,
                                      CeedVector *U, CeedVector *V) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChkBackend(ierr);
  return CeedError(ceed, CEED_ERROR_BACKEND,
                   "Backend does not implement QFunctionApply");
}

//------------------------------------------------------------------------------
// Destroy QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionDestroy_Hip_gen(CeedQFunction qf) {
  int ierr;
  CeedQFunction_Hip_gen *data;
  ierr = CeedQFunctionGetData(qf, &data); CeedChkBackend(ierr);
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChkBackend(ierr);
  ierr = hipFree(data->d_c); CeedChk_Hip(ceed, ierr);
  ierr = CeedFree(&data->qFunctionSource); CeedChkBackend(ierr);
  ierr = CeedFree(&data); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Load QFunction
//------------------------------------------------------------------------------
static int loadHipFunction(CeedQFunction qf, CeedInt num_files,
                           const char **c_src_files) {
  int ierr;
  Ceed ceed;
  CeedQFunctionGetCeed(qf, &ceed);
  CeedQFunction_Hip_gen *data;
  ierr = CeedQFunctionGetData(qf, &data); CeedChkBackend(ierr);

  // Loop over all source file(s)
  char *buffer;
  CeedInt buffer_offset = 1;
  ierr = CeedCalloc(buffer_offset + 1, &buffer); CeedChkBackend(ierr);
  strncpy(buffer, "\n", 2);
  for (CeedInt i = 0; i < num_files; i++) {    // Open source file
    FILE *fp;
    long lSize;
    fp = fopen (c_src_files[i], "rb");
    if (!fp)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "Couldn't open the Cuda file for the QFunction.");
    // LCOV_EXCL_STOP

    // Compute size of source
    fseek(fp, 0L, SEEK_END);
    lSize = ftell(fp);
    rewind(fp);

    // Allocate memory for entire content
    ierr = CeedRealloc(buffer_offset+lSize+2, &buffer); CeedChkBackend(ierr);

    // Copy the file into the buffer
    if (1 != fread(&buffer[buffer_offset], lSize, 1, fp)) {
      // LCOV_EXCL_START
      fclose(fp);
      ierr = CeedFree(&buffer); CeedChkBackend(ierr);
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "Couldn't read the Cuda file for the QFunction.");
      // LCOV_EXCL_STOP
    }
    strncpy(&buffer[buffer_offset + lSize], "\n", 2);

    // Cleanup
    fclose(fp);

    // Update offsets
    buffer_offset += lSize + 1;
  }

  // Append typedef and save source string
  // FIXME: the magic number 16 should be defined somewhere...
  char *fields_string =
    "typedef struct { const CeedScalar* inputs[16]; CeedScalar* outputs[16]; } Fields_Hip_gen;";
  ierr = CeedMalloc(1 + strlen(fields_string) + strlen(buffer),
                    &data->qFunctionSource); CeedChkBackend(ierr);
  memcpy(data->qFunctionSource, fields_string, strlen(fields_string));
  memcpy(data->qFunctionSource + strlen(fields_string), buffer,
         strlen(buffer) + 1);

  // Cleanup
  ierr = CeedFree(&buffer); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create QFunction
//------------------------------------------------------------------------------
int CeedQFunctionCreate_Hip_gen(CeedQFunction qf) {
  int ierr;
  Ceed ceed;
  CeedQFunctionGetCeed(qf, &ceed);
  CeedQFunction_Hip_gen *data;
  ierr = CeedCalloc(1, &data); CeedChkBackend(ierr);
  ierr = CeedQFunctionSetData(qf, data); CeedChkBackend(ierr);

  // Read source
  const char **sources;
  CeedInt num_sources;
  ierr = CeedQFunctionGetSourcePaths(qf, &num_sources, &sources);
  CeedChkBackend(ierr);
  const char *funname;
  ierr = CeedQFunctionGetName(qf, &funname); CeedChkBackend(ierr);
  data->qFunctionName = (char *)funname;
  ierr = loadHipFunction(qf, num_sources, sources); CeedChkBackend(ierr);

  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Apply",
                                CeedQFunctionApply_Hip_gen); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Destroy",
                                CeedQFunctionDestroy_Hip_gen); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
