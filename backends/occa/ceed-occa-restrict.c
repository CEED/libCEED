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
#define CEED_DEBUG_COLOR 13
#include "ceed-occa.h"
#include <sys/stat.h>

// *****************************************************************************
// * Bytes used
// *****************************************************************************
static inline size_t bytes(const CeedElemRestriction res) {
  return res->nelem * res->elemsize * sizeof(CeedInt);
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
  dbg("[CeedElemRestriction][Apply]");
  const CeedElemRestriction_Occa *data = r->data;
  const occaMemory id = data->d_indices;
  const occaMemory tid = data->d_tindices;
  const occaMemory od = data->d_toffsets;
  const CeedVector_Occa *u_data = u->data;
  const CeedVector_Occa *v_data = v->data;
  const occaMemory ud = u_data->d_array;
  const occaMemory vd = v_data->d_array;
  const CeedTransposeMode restriction = (tmode == CEED_NOTRANSPOSE);
  const CeedTransposeMode ordering = (lmode == CEED_NOTRANSPOSE);
  // ***************************************************************************
  if (restriction) {
    // Perform: v = r * u
    if (ncomp == 1) {
      dbg("[CeedElemRestriction][Apply] kRestrict[0]");
      occaKernelRun(data->kRestrict[0], id, ud, vd);
    } else {
      // v is (elemsize x ncomp x nelem), column-major
      if (ordering) {
        // u is (ndof x ncomp), column-major
        dbg("[CeedElemRestriction][Apply] kRestrict[1]");
        occaKernelRun(data->kRestrict[1], occaInt(ncomp), id, ud, vd);
      } else {
        // u is (ncomp x ndof), column-major
        dbg("[CeedElemRestriction][Apply] kRestrict[2]");
        occaKernelRun(data->kRestrict[2], occaInt(ncomp), id, ud, vd);
      }
    }
  } else { // ******************************************************************
    // Note: in transpose mode, we perform: v += r^t * u
    if (ncomp == 1) {
      dbg("[CeedElemRestriction][Apply] kRestrict[3]");
      //occaKernelRun(occa->kRestrict[3], id, ud, vd);
      occaKernelRun(data->kRestrict[6], tid, od, ud, vd);
    } else {
      // u is (elemsize x ncomp x nelem)
      if (ordering) {
        // v is (ndof x ncomp), column-major
        dbg("[CeedElemRestriction][Apply] kRestrict[4]");
        //occaKernelRun(data->kRestrict[4], occaInt(ncomp), id, ud, vd);
        occaKernelRun(data->kRestrict[7], occaInt(ncomp), id, od,ud, vd);
      } else {
        // v is (ncomp x ndof), column-major
        dbg("[CeedElemRestriction][Apply] kRestrict[5]");
        //occaKernelRun(data->kRestrict[5], occaInt(ncomp), id, ud, vd);
        occaKernelRun(data->kRestrict[8], occaInt(ncomp), id, od,ud, vd);
      }
    }
  }
  if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_ORDERED)
    *request = NULL;
  return 0;
}

// *****************************************************************************
static int CeedElemRestrictionDestroy_Occa(CeedElemRestriction r) {
  int ierr;
  CeedElemRestriction_Occa *data = r->data;
  dbg("[CeedElemRestriction][Destroy]");
  ierr = CeedFree(&data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
// * Compute the transposed Tindices and Toffsets from indices
// *****************************************************************************
static int CeedElemRestrictionOffset_Occa(const CeedElemRestriction r,
    const CeedInt *indices,
    CeedInt *toffsets,
    CeedInt *tindices) {
  const CeedInt nelem = r->nelem;
  const CeedInt elemsize = r->elemsize;
  const CeedInt ndof = r->ndof;
  for (int i=0; i<=ndof; ++i) toffsets[i]=0;
  for (int e=0; e < nelem; ++e)
    for (int i=0; i < elemsize; ++i)
      ++toffsets[indices[elemsize*e+i]+1];
  for (int i = 1; i <= ndof; ++i)
    toffsets[i] += toffsets[i-1];
  for (int e = 0; e < nelem; ++e) {
    for (int i = 0; i < elemsize; ++i) {
      const int lid = elemsize*e+i;
      const int gid = indices[lid];
      tindices[toffsets[gid]++] = lid;
    }
  }
  for (int i = ndof; i > 0; --i)
    toffsets[i] = toffsets[i - 1];
  toffsets[0] = 0;
  return 0;
}

// *****************************************************************************
int CeedElemRestrictionCreate_Occa(const CeedElemRestriction r,
                                   const CeedMemType mtype,
                                   const CeedCopyMode cmode,
                                   const CeedInt *indices) {
  dbg("[CeedElemRestriction][Create]");
  int ierr;
  CeedElemRestriction_Occa *data;
  Ceed_Occa *ceed_data = r->ceed->data;
  const occaDevice dev = ceed_data->device;
  // ***************************************************************************
  if (mtype != CEED_MEM_HOST)
    return CeedError(r->ceed, 1, "Only MemType = HOST supported");
  r->Apply = CeedElemRestrictionApply_Occa;
  r->Destroy = CeedElemRestrictionDestroy_Occa;
  // Allocating occa & device **************************************************
  dbg("[CeedElemRestriction][Create] Allocating");
  ierr = CeedCalloc(1,&data); CeedChk(ierr);
  r->data = data;
  // ***************************************************************************
  data->d_indices = occaDeviceMalloc(dev, bytes(r), NULL, NO_PROPS);
  data->d_toffsets = occaDeviceMalloc(dev,(1+r->ndof)*sizeof(CeedInt),
                                      NULL, NO_PROPS);
  data->d_tindices = occaDeviceMalloc(dev, bytes(r), NULL, NO_PROPS);
  // ***************************************************************************
  CeedInt toffsets[r->ndof+1];
  CeedInt tindices[r->elemsize*r->nelem];
  CeedElemRestrictionOffset_Occa(r,indices,toffsets,tindices);
  occaCopyPtrToMem(data->d_toffsets,toffsets,
                   (1+r->ndof)*sizeof(CeedInt),NO_OFFSET,NO_PROPS);
  occaCopyPtrToMem(data->d_tindices,tindices,bytes(r),NO_OFFSET,NO_PROPS);
  // ***************************************************************************
  occaCopyPtrToMem(data->d_indices,indices,bytes(r),NO_OFFSET,NO_PROPS);
  // ***************************************************************************
  dbg("[CeedElemRestriction][Create] Building kRestrict");
  occaProperties pKR = occaCreateProperties();
  occaPropertiesSet(pKR, "defines/ndof", occaInt(r->ndof));
  occaPropertiesSet(pKR, "defines/nelem", occaInt(r->nelem));
  occaPropertiesSet(pKR, "defines/elemsize", occaInt(r->elemsize));
  occaPropertiesSet(pKR, "defines/nelem_x_elemsize",
                    occaInt(r->nelem*r->elemsize));
  // OpenCL check for this requirement
  const CeedInt tile_size = (r->nelem>TILE_SIZE)?TILE_SIZE:r->nelem;
  occaPropertiesSet(pKR, "defines/TILE_SIZE", occaInt(tile_size));
  char oklPath[4096] = __FILE__;
  // path to ceed-occa-restrict.okl
  const size_t oklPathLen = strlen(oklPath);
  // consider using realpath(3) or something dynamic
  // should look for last '.' chr, instead of '-1'
  strcpy(&oklPath[oklPathLen-2],".okl");
  dbg("[CeedElemRestriction][Create] filename=%s",oklPath);
  // Test if we can get file's status, if not revert to occa://ceed/*.okl ******
  struct stat buf;
  if (stat(oklPath, &buf)!=0){
    dbg("[CeedElemRestriction][Create] Could NOT stat this OKL file: %s",oklPath);
    dbg("[CeedElemRestriction][Create] Reverting to occa://ceed/*.okl");
    strcpy(oklPath,"occa://ceed/ceed-occa-restrict.okl");
  }
  // ***************************************************************************
  data->kRestrict[0] = occaDeviceBuildKernel(dev, oklPath, "kRestrict0", pKR);
  data->kRestrict[1] = occaDeviceBuildKernel(dev, oklPath, "kRestrict1", pKR);
  data->kRestrict[2] = occaDeviceBuildKernel(dev, oklPath, "kRestrict2", pKR);
  data->kRestrict[3] = occaDeviceBuildKernel(dev, oklPath, "kRestrict3", pKR);
  data->kRestrict[4] = occaDeviceBuildKernel(dev, oklPath, "kRestrict4", pKR);
  data->kRestrict[5] = occaDeviceBuildKernel(dev, oklPath, "kRestrict5", pKR);
  data->kRestrict[6] = occaDeviceBuildKernel(dev, oklPath, "kRestrict3b", pKR);
  data->kRestrict[7] = occaDeviceBuildKernel(dev, oklPath, "kRestrict4b", pKR);
  data->kRestrict[8] = occaDeviceBuildKernel(dev, oklPath, "kRestrict5b", pKR);
  occaFree(pKR);
  dbg("[CeedElemRestriction][Create] done");
  return 0;
}
