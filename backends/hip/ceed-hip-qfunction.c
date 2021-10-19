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
#include "ceed-hip.h"
#include "ceed-hip-compile.h"
#include "ceed-hip-qfunction-load.h"

//------------------------------------------------------------------------------
// Apply QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionApply_Hip(CeedQFunction qf, CeedInt Q,
                                  CeedVector *U, CeedVector *V) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChkBackend(ierr);

  // Build and compile kernel, if not done
  ierr = CeedHipBuildQFunction(qf); CeedChkBackend(ierr);

  CeedQFunction_Hip *data;
  ierr = CeedQFunctionGetData(qf, &data); CeedChkBackend(ierr);
  Ceed_Hip *ceed_Hip;
  ierr = CeedGetData(ceed, &ceed_Hip); CeedChkBackend(ierr);
  CeedInt numinputfields, numoutputfields;
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChkBackend(ierr);
  const int blocksize = ceed_Hip->optblocksize;

  // Read vectors
  for (CeedInt i = 0; i < numinputfields; i++) {
    ierr = CeedVectorGetArrayRead(U[i], CEED_MEM_DEVICE, &data->fields.inputs[i]);
    CeedChkBackend(ierr);
  }
  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedVectorGetArray(V[i], CEED_MEM_DEVICE, &data->fields.outputs[i]);
    CeedChkBackend(ierr);
  }

  // Get context data
  CeedQFunctionContext ctx;
  ierr = CeedQFunctionGetInnerContext(qf, &ctx); CeedChkBackend(ierr);
  if (ctx) {
    ierr = CeedQFunctionContextGetData(ctx, CEED_MEM_DEVICE, &data->d_c);
    CeedChkBackend(ierr);
  }

  // Run kernel
  void *args[] = {&data->d_c, (void *) &Q, &data->fields};
  ierr = CeedRunKernelHip(ceed, data->qFunction, CeedDivUpInt(Q, blocksize),
                          blocksize, args); CeedChkBackend(ierr);

  // Restore vectors
  for (CeedInt i = 0; i < numinputfields; i++) {
    ierr = CeedVectorRestoreArrayRead(U[i], &data->fields.inputs[i]);
    CeedChkBackend(ierr);
  }
  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedVectorRestoreArray(V[i], &data->fields.outputs[i]);
    CeedChkBackend(ierr);
  }

  // Restore context
  if (ctx) {
    ierr = CeedQFunctionContextRestoreData(ctx, &data->d_c);
    CeedChkBackend(ierr);
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy QFunction
//------------------------------------------------------------------------------
static int CeedQFunctionDestroy_Hip(CeedQFunction qf) {
  int ierr;
  CeedQFunction_Hip *data;
  ierr = CeedQFunctionGetData(qf, &data); CeedChkBackend(ierr);
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChkBackend(ierr);
  if  (data->module)
    CeedChk_Hip(ceed, hipModuleUnload(data->module));
  ierr = CeedFree(&data); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Load QFunction source file
//------------------------------------------------------------------------------
static int CeedHipLoadQFunction(CeedQFunction qf, CeedInt num_files,
                                const char **c_src_files) {
  int ierr;
  Ceed ceed;
  CeedQFunctionGetCeed(qf, &ceed);

  // Loop over all source file(s)
  char *buffer;
  CeedInt buffer_offset = 1;
  ierr = CeedCalloc(buffer_offset + 1, &buffer); CeedChkBackend(ierr);
  strncpy(buffer, "\n", 2);
  for (CeedInt i = 0; i < num_files; i++) {
    // Open source file
    FILE *fp;
    long lSize;
    fp = fopen (c_src_files[i], "rb");
    if (!fp)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "Error opening QFunction source file");
    // LCOV_EXCL_STOP

    // Compute size of source
    fseek(fp, 0L, SEEK_END);
    lSize = ftell(fp);
    rewind(fp);

    // Allocate memory for entire content
    ierr = CeedRealloc(buffer_offset + lSize + 2, &buffer); CeedChkBackend(ierr);

    // Copy the file into the buffer
    if (1 != fread(&buffer[buffer_offset], lSize, 1, fp)) {
      // LCOV_EXCL_START
      fclose(fp);
      ierr = CeedFree(&buffer); CeedChkBackend(ierr);
      return CeedError(ceed, CEED_ERROR_BACKEND,
                       "Error reading QFunction source file");
      // LCOV_EXCL_STOP
    }
    strncpy(&buffer[buffer_offset + lSize], "\n", 2);

    // Cleanup
    fclose(fp);

    // Update offsets
    buffer_offset += lSize + 1;
  }

  // Save QFunction source
  CeedQFunction_Hip *data;
  ierr = CeedQFunctionGetData(qf, &data); CeedChkBackend(ierr);
  data->qFunctionSource = buffer;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create QFunction
//------------------------------------------------------------------------------
int CeedQFunctionCreate_Hip(CeedQFunction qf) {
  int ierr;
  Ceed ceed;
  CeedQFunctionGetCeed(qf, &ceed);
  CeedQFunction_Hip *data;
  ierr = CeedCalloc(1,&data); CeedChkBackend(ierr);
  ierr = CeedQFunctionSetData(qf, data); CeedChkBackend(ierr);
  CeedInt numinputfields, numoutputfields;
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChkBackend(ierr);

  // Read source
  const char **sources;
  CeedInt num_sources;
  ierr = CeedQFunctionGetSourcePaths(qf, &num_sources, &sources);
  CeedChkBackend(ierr);
  const char *funname;
  ierr = CeedQFunctionGetName(qf, &funname); CeedChkBackend(ierr);
  data->qFunctionName = (char *)funname;
  ierr = CeedHipLoadQFunction(qf, num_sources, sources); CeedChkBackend(ierr);

  // Register backend functions
  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Apply",
                                CeedQFunctionApply_Hip); CeedChkBackend(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Destroy",
                                CeedQFunctionDestroy_Hip); CeedChkBackend(ierr);
  return CEED_ERROR_SUCCESS;
}
//------------------------------------------------------------------------------
