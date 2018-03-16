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
// * Bytes used
// *****************************************************************************
static inline size_t bytes(const CeedElemRestriction res) {
  return res->nelem * res->elemsize * sizeof(CeedInt);
}

// *****************************************************************************
// * OCCA SYNC functions
// *****************************************************************************
static inline void occaSyncH2D(const CeedElemRestriction res) {
  const CeedElemRestriction_Occa *impl = res->data;
  assert(impl);
  assert(impl->d_indices);
  occaCopyPtrToMem(*impl->d_indices, impl->h_indices, bytes(res),
                   NO_OFFSET, NO_PROPS);
}

// *****************************************************************************
// * Restrict an L-vector to an E-vector or apply transpose
// *****************************************************************************
static int CeedElemRestrictionApply_Occa(CeedElemRestriction r,
    CeedTransposeMode tmode,
    CeedInt ncomp,
    CeedTransposeMode lmode,
    CeedVector u, CeedVector v,
    CeedRequest *request) {
  CeedDebug("\033[35m[CeedElemRestriction][Apply]");
  const CeedElemRestriction_Occa *occa = r->data;
  const occaMemory id = *occa->d_indices;
  const CeedVector_Occa *u_data = u->data;
  const CeedVector_Occa *v_data = v->data;
  const occaMemory ud = *u_data->d_array;
  const occaMemory vd = *v_data->d_array;
  if (tmode == CEED_NOTRANSPOSE) {
    // Perform: v = r * u
    if (ncomp == 1) {
      occaKernelRun(occa->kRestrict[0], id, ud, vd);
    } else {
      // v is (elemsize x ncomp x nelem), column-major
      if (lmode == CEED_NOTRANSPOSE) {
        // u is (ndof x ncomp), column-major
        occaKernelRun(occa->kRestrict[1], occaInt(ncomp), id, ud, vd);
      } else {
        // u is (ncomp x ndof), column-major
        occaKernelRun(occa->kRestrict[2], occaInt(ncomp), id, ud, vd);
      }
    }
  } else {
    // Note: in transpose mode, we perform: v += r^t * u
    if (ncomp == 1) {
      occaKernelRun(occa->kRestrict[3], id, ud, vd);
    } else {
      // u is (elemsize x ncomp x nelem)
      if (lmode == CEED_NOTRANSPOSE) {
        // v is (ndof x ncomp), column-major
        occaKernelRun(occa->kRestrict[4], occaInt(ncomp), id, ud, vd);
      } else {
        // v is (ncomp x ndof), column-major
        occaKernelRun(occa->kRestrict[5], occaInt(ncomp), id, ud, vd);
      }
    }
  }
  if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_ORDERED)
    *request = NULL;
  return 0;
}


// *****************************************************************************
// * CeedElemRestrictionDestroy_Occa
// *****************************************************************************
static int CeedElemRestrictionDestroy_Occa(CeedElemRestriction r) {
  int ierr;
  CeedElemRestriction_Occa *data = r->data;
  CeedDebug("\033[35m[CeedElemRestriction][Destroy]");
  occaMemoryFree(*data->d_indices);
  occaKernelFree(data->kRestrict[0]);
  occaKernelFree(data->kRestrict[1]);
  occaKernelFree(data->kRestrict[2]);
  occaKernelFree(data->kRestrict[3]);
  occaKernelFree(data->kRestrict[4]);
  occaKernelFree(data->kRestrict[5]);
  ierr = CeedFree(&data->h_indices); CeedChk(ierr);
  ierr = CeedFree(&data->d_indices); CeedChk(ierr);
  ierr = CeedFree(&data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
// * CeedElemRestrictionCreate_Occa
// *****************************************************************************
int CeedElemRestrictionCreate_Occa(const CeedElemRestriction r,
                                   const CeedMemType mtype,
                                   const CeedCopyMode cmode,
                                   const CeedInt *indices) {
  CeedDebug("\033[35m[CeedElemRestriction][Create]");
  int ierr;
  CeedElemRestriction_Occa *data;
  const Ceed_Occa *occa=r->ceed->data;
  // ***************************************************************************
  if (mtype != CEED_MEM_HOST)
    return CeedError(r->ceed, 1, "Only MemType = HOST supported");
  // Set the functions *********************************************************
  r->Apply = CeedElemRestrictionApply_Occa;
  r->Destroy = CeedElemRestrictionDestroy_Occa;
  // Allocating occa & device **************************************************
  CeedDebug("\033[35m[CeedElemRestriction][Create] Allocating");
  ierr = CeedCalloc(1,&data); CeedChk(ierr);
  ierr = CeedCalloc(1,&data->d_indices); CeedChk(ierr);
  *data->d_indices = occaDeviceMalloc(*occa->device, bytes(r), NULL, NO_PROPS);
  r->data = data;
  assert(indices);
  // ***************************************************************************
  switch (cmode) {
  // Will copy the values and not store the passed pointer
  case CEED_COPY_VALUES:
    CeedDebug("\t\033[35m[CeedElemRestriction][Create] CEED_COPY_VALUES");
    ierr = CeedMalloc(r->nelem*r->elemsize, &data->h_indices); CeedChk(ierr);
    assert(indices);
    assert(data->h_indices);
    memcpy((CeedInt*)data->h_indices, indices, bytes(r));
    occaSyncH2D(r);
    break;
  // Takes ownership of the pointer and will free using CeedFree()
  case CEED_OWN_POINTER:
    CeedDebug("\t\033[35m[CeedElemRestriction][Create] CEED_OWN_POINTER");
    data->h_indices = indices;
    occaSyncH2D(r);
    break;
  /// Can use and modify the data provided by the user
  case CEED_USE_POINTER:
    CeedDebug("\t\033[35m[CeedElemRestriction][Create] CEED_USE_POINTER");
    data->h_indices = indices;
    occaSyncH2D(r);
    data->h_indices = NULL; /// but does not take ownership
    break;
  default: CeedError(r->ceed,1," OCCA backend no default error");
  }
  // ***************************************************************************
  CeedDebug("\033[35m[CeedElemRestriction][Create] Building kRestrict");
  occaProperties pKR = occaCreateProperties();
  occaPropertiesSet(pKR, "defines/esize", occaInt(r->nelem*r->elemsize));
  occaPropertiesSet(pKR, "defines/ndof", occaInt(r->ndof));
  occaPropertiesSet(pKR, "defines/nelem", occaInt(r->nelem));
  occaPropertiesSet(pKR, "defines/elemsize", occaInt(r->elemsize));
  occaPropertiesSet(pKR, "defines/TILE_SIZE", occaInt(TILE_SIZE));
  const occaDevice dev = *occa->device;
  char oklPath[4096] = __FILE__;
  // path to ceed-occa-restrict.okl
  const size_t oklPathLen = strlen(oklPath);
  // consider using realpath(3) or something dynamic
  strcpy(&oklPath[oklPathLen-2],".okl");
  data->kRestrict[0] = occaDeviceBuildKernel(dev, oklPath, "kRestrict0", pKR);
  data->kRestrict[1] = occaDeviceBuildKernel(dev, oklPath, "kRestrict1", pKR);
  data->kRestrict[2] = occaDeviceBuildKernel(dev, oklPath, "kRestrict2", pKR);
  data->kRestrict[3] = occaDeviceBuildKernel(dev, oklPath, "kRestrict3", pKR);
  data->kRestrict[4] = occaDeviceBuildKernel(dev, oklPath, "kRestrict4", pKR);
  data->kRestrict[5] = occaDeviceBuildKernel(dev, oklPath, "kRestrict5", pKR);
  occaPropertiesFree(pKR);
  CeedDebug("\033[35m[CeedElemRestriction][Create] done");
  return 0;
}

// *****************************************************************************
// * TENSORS: Contracts on the middle index
// *          NOTRANSPOSE: V_ajc = T_jb U_abc
// *          TRANSPOSE:   V_ajc = T_bj U_abc
// * CeedScalars are used here, not CeedVectors: we don't touch it yet
// *****************************************************************************
int CeedTensorContract_Occa(Ceed ceed,
                            CeedInt A, CeedInt B, CeedInt C, CeedInt J,
                            const CeedScalar *t, CeedTransposeMode tmode,
                            const CeedInt Add,
                            const CeedScalar *u, CeedScalar *v) {
  CeedInt tstride0 = B, tstride1 = 1;
  //CeedDebug("\033[35m[CeedTensorContract] A=%d, J=%d, C=%d, B=%d: %d",A,J,C,B,A*J*B*C);
  if (tmode == CEED_TRANSPOSE) {
    tstride0 = 1; tstride1 = J;
  }
  for (CeedInt a=0; a<A; a++) {
    for (CeedInt j=0; j<J; j++) {
      if (!Add) {
        for (CeedInt c=0; c<C; c++)
          v[(a*J+j)*C+c] = 0;
      }
      for (CeedInt b=0; b<B; b++) {
        for (CeedInt c=0; c<C; c++) {
          v[(a*J+j)*C+c] += t[j*tstride0 + b*tstride1] * u[(a*B+b)*C+c];
        }
      }
    }
  }
  return 0;
}
