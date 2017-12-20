// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
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
#include "ceed-occa.h"

// *****************************************************************************
// * RESTRICTIONS: Create, Apply, Destroy
// *****************************************************************************
typedef struct {
  occaKernel ceed_occa_restrict;
  const CeedInt* indices;
  occaMemory* indices_allocated;
} CeedElemRestrictionOcca;


// *****************************************************************************
static int CeedElemRestrictionApplyOcca(CeedElemRestriction r,
                                           CeedTransposeMode tmode, CeedVector u,
                                           CeedVector v, CeedRequest* request) {
  CeedElemRestrictionOcca* impl = r->data;
  //int ierr;
  //const occaMemory idx;
  //const occaMemory uu;
  //occaMemory vv;

  dbg("[CeedElemRestriction][Apply][Occa]");
  //ierr = CeedVectorGetArrayRead(u, CEED_MEM_HOST, &uu); CeedChk(ierr);
  //ierr = CeedVectorGetArray(v, CEED_MEM_HOST, &vv); CeedChk(ierr);
  dbg("[CeedElemRestriction][Apply][Occa] got arrays");
  //assert(uu);
  //assert(vv);
  occaKernelRun(impl->ceed_occa_restrict,occaInt(12));//,idx,uu,vv);
  dbg("[CeedElemRestriction][Apply][Occa] restricted");
  //ierr = CeedVectorRestoreArrayRead(u, &uu); CeedChk(ierr);
  //ierr = CeedVectorRestoreArray(v, &vv); CeedChk(ierr);
  assert(false);
  if (request != CEED_REQUEST_IMMEDIATE) *request = NULL;
  return 0;
}

// *****************************************************************************
static int CeedElemRestrictionDestroyOcca(CeedElemRestriction r) {
  CeedElemRestrictionOcca* impl = r->data;
  int ierr;

  dbg("[CeedElemRestriction][Destroy][Occa]");
  ierr = CeedFree(&impl->indices_allocated); CeedChk(ierr);
  ierr = CeedFree(&r->data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
int CeedElemRestrictionCreateOcca(CeedElemRestriction r,
                                  CeedMemType mtype,
                                  CeedCopyMode cmode,
                                  const CeedInt* indices) {
  int ierr;
  CeedElemRestrictionOcca* impl;

  dbg("[CeedElemRestriction][Create][Occa]");
  if (mtype != CEED_MEM_HOST)
    return CeedError(r->ceed, 1, "Only MemType = HOST supported");
  ierr = CeedCalloc(1,&impl); CeedChk(ierr);
  ierr = CeedCalloc(1,&impl->indices_allocated); CeedChk(ierr);
  const size_t bytes = r->nelem * sizeof(CeedInt);
//*impl->array_allocated = occaDeviceMalloc(device, impl->size*sizeof(CeedScalar), NULL, occaDefault);
  switch (cmode) {
    case CEED_COPY_VALUES:
      assert(false);
      //ierr = CeedMalloc(r->nelem*r->elemsize, &impl->indices_allocated);CeedChk(ierr);
      //memcpy(impl->indices_allocated, indices,r->nelem * r->elemsize * sizeof(indices[0]));
      //impl->indices = impl->indices_allocated;
      break;
    case CEED_OWN_POINTER:
      *impl->indices_allocated = occaDeviceMalloc(device, bytes, NULL, occaDefault);
      //impl->indices_allocated = (CeedInt*)indices;
      //impl->indices = impl->indices_allocated;
      break;
    case CEED_USE_POINTER:
      impl->indices = indices;
  }
  
  occaProperties props = occaCreateProperties();
  occaPropertiesSet(props, "defines/nelemsize", occaInt(r->nelem*r->elemsize));
  occaPropertiesSet(props, "defines/TRANSPOSE", occaInt(false));
  occaPropertiesSet(props, "defines/TILE_SIZE", occaInt(occaTileSize));
  
  impl->ceed_occa_restrict = occaDeviceBuildKernel(device,"occa/ceed-occa-restrict.okl", "ceed_occa_restrict",props);
  
  r->data = impl;
  r->Apply = CeedElemRestrictionApplyOcca;
  r->Destroy = CeedElemRestrictionDestroyOcca;
  return 0;
}


// *****************************************************************************
// * TENSORS: Contracts on the middle index
// *          NOTRANSPOSE: V_ajc = T_jb U_abc
// *          TRANSPOSE:   V_ajc = T_bj U_abc
// *****************************************************************************
int CeedTensorContractOcca(Ceed ceed,
                           CeedInt A, CeedInt B, CeedInt C, CeedInt J,
                           const CeedScalar* t, CeedTransposeMode tmode,
                           const CeedScalar* u, CeedScalar* v) {
  CeedInt tstride0 = B, tstride1 = 1;
  dbg("[CeedTensorContract][Occa]");
  if (tmode == CEED_TRANSPOSE) {
    tstride0 = 1; tstride1 = B;
  }

  for (CeedInt a=0; a<A; a++) {
    for (CeedInt j=0; j<J; j++) {
      for (CeedInt c=0; c<C; c++)
        v[(a*J+j)*C+c] = 0;
      for (CeedInt b=0; b<B; b++) {
        for (CeedInt c=0; c<C; c++) {
          v[(a*J+j)*C+c] += t[j*tstride0 + b*tstride1] * u[(a*B+b)*C+c];
        }
      }
    }
  }
  return 0;
}
