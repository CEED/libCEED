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

#include "ceed-magma.h"

static int CeedDestroy_Magma(Ceed ceed) {
  int ierr;
  Ceed_Magma *data;
  ierr = CeedGetData(ceed, &data); CeedChk(ierr);
  magma_queue_destroy( data->queue );
  ierr = CeedFree(&data); CeedChk(ierr);
  return 0;
}

static int CeedInit_Magma(const char *resource, Ceed ceed) {
  int ierr;
  if (strcmp(resource, "/gpu/magma"))
    // LCOV_EXCL_START
    return CeedError(ceed, 1, "Magma backend cannot use resource: %s", resource);
  // LCOV_EXCL_STOP

  // Create reference CEED that implementation will be dispatched
  //   through unless overridden
  Ceed ceedref;
  CeedInit("/gpu/cuda/ref", &ceedref);
  ierr = CeedSetDelegate(ceed, ceedref); CeedChk(ierr);

  ierr = magma_init();
  if (ierr)
    // LCOV_EXCL_START
    return CeedError(ceed, 1, "error in magma_init(): %d\n", ierr);
  // LCOV_EXCL_STOP

  Ceed_Magma *data;
  ierr = CeedCalloc(sizeof(Ceed_Magma), &data); CeedChk(ierr);
  ierr = CeedSetData(ceed, data); CeedChk(ierr);

  // kernel selection
  data->basis_kernel_mode = MAGMA_KERNEL_DIM_SPECIFIC;

  // kernel max threads per thread-block
  data->maxthreads[0] = 128;  // for 1D kernels
  data->maxthreads[1] = 128;  // for 2D kernels
  data->maxthreads[2] =  64;  // for 3D kernels

  // create a queue that uses the null stream
  magma_getdevice( &(data->device) );
  magma_queue_create_from_cuda(data->device, NULL, NULL, NULL, &(data->queue));

  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "ElemRestrictionCreate",
                                CeedElemRestrictionCreate_Magma); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed,
                                "ElemRestrictionCreateBlocked",
                                CeedElemRestrictionCreateBlocked_Magma); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateTensorH1",
                                CeedBasisCreateTensorH1_Magma); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateH1",
                                CeedBasisCreateH1_Magma); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy",
                                CeedDestroy_Magma); CeedChk(ierr);
  return 0;
}

__attribute__((constructor))
static void Register(void) {
  CeedRegister("/gpu/magma", CeedInit_Magma, 20);
}
