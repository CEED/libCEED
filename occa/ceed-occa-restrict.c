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
  const CeedInt* host;
  occaMemory* device;
  occaKernel kRestrict;
} CeedElemRestrictionOcca;

// *****************************************************************************
// * Bytes used
// *****************************************************************************
static inline size_t bytes(const CeedElemRestriction res){
  return res->nelem * res->elemsize * sizeof(CeedInt);
}

// *****************************************************************************
// * OCCA SYNC functions
// * Ptr == void*, Mem == device
// * occaCopyPtrToMem(occaMemory dest, const void *src,
// * occaCopyMemToPtr(void *dest, occaMemory src,
// *****************************************************************************
static inline void occaSyncH2D(const CeedElemRestriction res){
  const CeedElemRestrictionOcca* impl = res->data;
  occaCopyPtrToMem(*impl->device, impl->host, bytes(res), NO_OFFSET, NO_PROPS);
}
static inline void occaSyncD2H(const CeedElemRestriction res){
  const CeedElemRestrictionOcca* impl = res->data;
  occaCopyMemToPtr((void*)impl->host, *impl->device, bytes(res), NO_OFFSET, NO_PROPS);
}

// *****************************************************************************
// * OCCA COPY functions
// *****************************************************************************
static inline void occaCopyH2D(const CeedElemRestriction res, const void *from){
  const CeedElemRestrictionOcca* impl = res->data;
  assert(from);
  assert(impl);
  assert(impl->device);
  occaCopyPtrToMem(*impl->device, from, bytes(res), NO_OFFSET, NO_PROPS);
}
static inline void occaCopyD2H(const CeedElemRestriction res, void *to){
  const CeedElemRestrictionOcca* impl = res->data;
  assert(to);
  assert(impl);
  assert(impl->device);
  occaCopyMemToPtr(to, *impl->device, bytes(res), NO_OFFSET, NO_PROPS);
}


// *****************************************************************************
// * CeedElemRestrictionApplyOcca
// *****************************************************************************
static int CeedElemRestrictionApplyOcca(CeedElemRestriction res,
                                        CeedTransposeMode tmode, CeedVector u,
                                        CeedVector v, CeedRequest* request) {
  const bool TRANSPOSE = tmode == CEED_NOTRANSPOSE;
  const CeedElemRestrictionOcca* impl = res->data;
  const occaMemory indices = *impl->device;
  const CeedVectorOcca* u_impl = u->data;
  const occaMemory uu = *u_impl->device;
  const CeedVectorOcca* v_impl = v->data;
  occaMemory vv = *v_impl->device;
  
  dbg("\033[35m[CeedElemRestriction][Apply]");
  occaKernelRun(impl->kRestrict,occaInt(TRANSPOSE),indices,uu,vv);  
  if (request != CEED_REQUEST_IMMEDIATE) *request = NULL;
  return 0;
}

// *****************************************************************************
// * CeedElemRestrictionDestroyOcca
// *****************************************************************************
static int CeedElemRestrictionDestroyOcca(CeedElemRestriction res) {
  CeedElemRestrictionOcca* impl = res->data;

  dbg("\033[35m[CeedElemRestriction][Destroy]");
  // free device memory
  occaMemoryFree(*impl->device);
  // free device object
  CeedChk(CeedFree(&impl->device));
  // free our CeedElemRestrictionOcca struct
  CeedChk(CeedFree(&res->data));
  return 0;
}

// *****************************************************************************
int CeedElemRestrictionCreateOcca(const CeedElemRestriction res,
                                  const CeedMemType mtype,
                                  const CeedCopyMode cmode,
                                  const CeedInt* indices) {
  if (mtype != CEED_MEM_HOST)
    return CeedError(res->ceed, 1, "Only MemType = HOST supported");
  // Allocating impl & device **************************************************
  dbg("\033[35m[CeedElemRestriction][Create] Allocating");
  CeedChk(CeedCalloc(1,&res->data));
  CeedElemRestrictionOcca* impl=res->data;
  // for now, target the device, whatever =cmode= is
  CeedChk(CeedCalloc(1,&impl->device));
  *impl->device = occaDeviceMalloc(device, bytes(res), NULL, NO_PROPS);
  // ***************************************************************************
  switch (cmode) {
  case CEED_COPY_VALUES:
    dbg("\t\033[35m[CeedElemRestriction][Create] CEED_COPY_VALUES");
    break;
  case CEED_OWN_POINTER:
    dbg("\t\033[35m[CeedElemRestriction][Create] CEED_OWN_POINTER");
    impl->host = indices;
    break;
  case CEED_USE_POINTER:
    dbg("\t\033[35m[CeedElemRestriction][Create] CEED_USE_POINTER");
    impl->host = indices;
    break;
  default: CeedError(res->ceed,1," OCCA backend no default error");
  }
  dbg("\033[35m[CeedElemRestriction][Create] occaCopyH2D");
  occaCopyH2D(res,indices);
  // ***************************************************************************
  dbg("\033[35m[CeedElemRestriction][Create] Building kRestrict");
  occaProperties pKR = occaCreateProperties();
  occaPropertiesSet(pKR, "defines/nelemsize", occaInt(res->nelem*res->elemsize));
  occaPropertiesSet(pKR, "defines/TILE_SIZE", occaInt(TILE_SIZE));
  impl->kRestrict = occaDeviceBuildKernel(device,
                                          "occa/ceed-occa-restrict.okl",
                                          "kRestrict",pKR);
  // ***************************************************************************
  res->Apply = CeedElemRestrictionApplyOcca;
  res->Destroy = CeedElemRestrictionDestroyOcca;
  res->data = impl;
  dbg("\033[35m[CeedElemRestriction][Create] done");
  return 0;
}


// *****************************************************************************
// * TENSORS: Contracts on the middle index
// *          NOTRANSPOSE: V_ajc = T_jb U_abc
// *          TRANSPOSE:   V_ajc = T_bj U_abc
// * CeedScalars are used here, not CeedVectors: we don't touch it yet
// *****************************************************************************
int CeedTensorContractOcca(Ceed ceed,
                           CeedInt A, CeedInt B, CeedInt C, CeedInt J,
                           const CeedScalar* t, CeedTransposeMode tmode,
                           const CeedScalar* u, CeedScalar* v) {
  CeedInt tstride0 = B, tstride1 = 1;
  
  dbg("\033[35m[CeedTensorContract]");
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
