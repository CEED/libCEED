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

// *****************************************************************************
// * Bytes used
// *****************************************************************************
static inline size_t bytes(const CeedElemRestriction res) {
  int ierr;
  CeedInt nelem, elemsize;
  ierr = CeedElemRestrictionGetNumElements(res, &nelem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(res, &elemsize); CeedChk(ierr);
  return nelem * elemsize * sizeof(CeedInt);
}

// *****************************************************************************
// * Restrict an L-vector to an E-vector or apply transpose
// *****************************************************************************
static
int CeedElemRestrictionApply_Occa(CeedElemRestriction r,
                                  CeedTransposeMode tmode,
                                  CeedTransposeMode lmode,
                                  CeedVector u, CeedVector v,
                                  CeedRequest *request) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  CeedInt ncomp;
  ierr = CeedElemRestrictionGetNumComponents(r, &ncomp); CeedChk(ierr);
  dbg("[CeedElemRestriction][Apply]");
  CeedElemRestriction_Occa *data;
  ierr = CeedElemRestrictionGetData(r, (void *)&data); CeedChk(ierr);
  const occaMemory id = data->d_indices;
  const occaMemory tid = data->d_tindices;
  const occaMemory od = data->d_toffsets;
  CeedVector_Occa *u_data;
  ierr = CeedVectorGetData(u, (void *)&u_data); CeedChk(ierr);
  CeedVector_Occa *v_data;
  ierr = CeedVectorGetData(v, (void *)&v_data); CeedChk(ierr);
  const occaMemory ud = u_data->d_array;
  const occaMemory vd = v_data->d_array;
  const CeedTransposeMode restriction = (tmode == CEED_NOTRANSPOSE);
  const CeedTransposeMode ordering = (lmode == CEED_NOTRANSPOSE);
  const bool identity = data->identity;
  // ***************************************************************************
  if (identity) {
    dbg("[CeedElemRestriction][Apply] kRestrict[6]");
    occaKernelRun(data->kRestrict[6], ud, vd);
  } else if (restriction) {
    // Perform: v = r * u
    if (ncomp == 1) {
      dbg("[CeedElemRestriction][Apply] kRestrict[0]");
      occaKernelRun(data->kRestrict[0], id, ud, vd);
    } else {
      // v is (elemsize x ncomp x nelem), column-major
      if (ordering) {
        // u is (nnodes x ncomp), column-major
        dbg("[CeedElemRestriction][Apply] kRestrict[1]");
        occaKernelRun(data->kRestrict[1], occaInt(ncomp), id, ud, vd);
      } else {
        // u is (ncomp x nnodes), column-major
        dbg("[CeedElemRestriction][Apply] kRestrict[2]");
        occaKernelRun(data->kRestrict[2], occaInt(ncomp), id, ud, vd);
      }
    }
  } else { // ******************************************************************
    // Note: in transpose mode, we perform: v += r^t * u
    if (ncomp == 1) {
      dbg("[CeedElemRestriction][Apply] kRestrict[3]");
      occaKernelRun(data->kRestrict[3], tid, od, ud, vd);
    } else {
      // u is (elemsize x ncomp x nelem)
      if (ordering) {
        // v is (nnodes x ncomp), column-major
        dbg("[CeedElemRestriction][Apply] kRestrict[4]");
        occaKernelRun(data->kRestrict[4], occaInt(ncomp), tid, od,ud, vd);
      } else {
        // v is (ncomp x nnodes), column-major
        dbg("[CeedElemRestriction][Apply] kRestrict[5]");
        occaKernelRun(data->kRestrict[5], occaInt(ncomp), tid, od,ud, vd);
      }
    }
  }
  if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_ORDERED)
    *request = NULL;
  return 0;
}

// *****************************************************************************
int CeedElemRestrictionApplyBlock_Occa(CeedElemRestriction r,
                                       CeedInt block, CeedTransposeMode tmode, CeedTransposeMode lmode,
                                       CeedVector u, CeedVector v, CeedRequest *request) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  return CeedError(ceed, 1, "Backend does not implement blocked restrictions");
}

// *****************************************************************************
static int CeedElemRestrictionDestroy_Occa(CeedElemRestriction r) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  CeedElemRestriction_Occa *data;
  ierr = CeedElemRestrictionGetData(r, (void *)&data); CeedChk(ierr);
  dbg("[CeedElemRestriction][Destroy]");
  for (int i=0; i<7; i++) {
    occaFree(data->kRestrict[i]);
    data->kRestrict[i] = occaUndefined;
  }
  ierr = CeedFree(&data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
// * Compute the transposed Tindices and Toffsets from indices
// *****************************************************************************
static
int CeedElemRestrictionOffset_Occa(const CeedElemRestriction r,
                                   const CeedInt *indices,
                                   CeedInt *toffsets,
                                   CeedInt *tindices) {
  int ierr;
  CeedInt nelem, elemsize, nnodes;
  ierr = CeedElemRestrictionGetNumElements(r, &nelem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(r, &elemsize); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumNodes(r, &nnodes); CeedChk(ierr);
  for (int i=0; i<=nnodes; ++i) toffsets[i]=0;
  for (int e=0; e < nelem; ++e)
    for (int i=0; i < elemsize; ++i)
      ++toffsets[indices[elemsize*e+i]+1];
  for (int i = 1; i <= nnodes; ++i)
    toffsets[i] += toffsets[i-1];
  for (int e = 0; e < nelem; ++e) {
    for (int i = 0; i < elemsize; ++i) {
      const int lid = elemsize*e+i;
      const int gid = indices[lid];
      tindices[toffsets[gid]++] = lid;
    }
  }
  for (int i = nnodes; i > 0; --i)
    toffsets[i] = toffsets[i - 1];
  toffsets[0] = 0;
  return 0;
}

// *****************************************************************************
int CeedElemRestrictionCreate_Occa(const CeedMemType mtype,
                                   const CeedCopyMode cmode,
                                   const CeedInt *indices,
                                   const CeedElemRestriction r) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  CeedInt nnodes, nelem, ncomp, elemsize;
  ierr = CeedElemRestrictionGetNumNodes(r, &nnodes); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumElements(r, &nelem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(r, &elemsize); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumComponents(r, &ncomp); CeedChk(ierr);
  dbg("[CeedElemRestriction][Create]");
  CeedElemRestriction_Occa *data;
  Ceed_Occa *ceed_data;
  ierr = CeedGetData(ceed, (void *)&ceed_data); CeedChk(ierr);
  const bool ocl = ceed_data->ocl;
  const occaDevice dev = ceed_data->device;
  // ***************************************************************************
  if (mtype != CEED_MEM_HOST)
    return CeedError(ceed, 1, "Only MemType = HOST supported");
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Apply",
                                CeedElemRestrictionApply_Occa); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "ApplyBlock",
                                CeedElemRestrictionApplyBlock_Occa);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Destroy",
                                CeedElemRestrictionDestroy_Occa); CeedChk(ierr);
  // Allocating occa & device **************************************************
  dbg("[CeedElemRestriction][Create] Allocating");
  ierr = CeedCalloc(1,&data); CeedChk(ierr);
  // ***************************************************************************
  CeedInt *toffsets;
  CeedInt *tindices;
  data->d_indices = occaDeviceMalloc(dev, bytes(r), NULL, NO_PROPS);
  data->d_toffsets = occaDeviceMalloc(dev,(1+nnodes)*sizeof(CeedInt),
                                      NULL, NO_PROPS);
  data->d_tindices = occaDeviceMalloc(dev, bytes(r), NULL, NO_PROPS);
  // ***************************************************************************
  ierr = CeedMalloc(nnodes+1, &toffsets); CeedChk(ierr);
  ierr = CeedMalloc(elemsize*nelem, &tindices); CeedChk(ierr);
  if (indices) {
    CeedElemRestrictionOffset_Occa(r,indices,toffsets,tindices);
    occaCopyPtrToMem(data->d_toffsets,toffsets,
                     (1+nnodes)*sizeof(CeedInt),NO_OFFSET,NO_PROPS);
    occaCopyPtrToMem(data->d_tindices,tindices,bytes(r),NO_OFFSET,NO_PROPS);
    // ***************************************************************************
    occaCopyPtrToMem(data->d_indices,indices,bytes(r),NO_OFFSET,NO_PROPS);
  } else {
    data->identity = true;
  }
  // ***************************************************************************
  dbg("[CeedElemRestriction][Create] Building kRestrict");

  dbg("[CeedElemRestriction][Create] Initialize kRestrict");
  for (int i = 0; i < CEED_OCCA_NUM_RESTRICTION_KERNELS; ++i) {
    data->kRestrict[i] = occaUndefined;
  }
  ierr = CeedElemRestrictionSetData(r, (void *)&data); CeedChk(ierr);
  dbg("[CeedElemRestriction][Create] nelem=%d",nelem);
  occaProperties pKR = occaCreateProperties();
  occaPropertiesSet(pKR, "defines/nnodes", occaInt(nnodes));
  occaPropertiesSet(pKR, "defines/nelem", occaInt(nelem));
  occaPropertiesSet(pKR, "defines/elemsize", occaInt(elemsize));
  occaPropertiesSet(pKR, "defines/nelem_x_elemsize",
                    occaInt(nelem*elemsize));
  occaPropertiesSet(pKR, "defines/nelem_x_elemsize_x_ncomp",
                    occaInt(nelem*elemsize*ncomp));
  // OpenCL check for this requirement
  const CeedInt nelem_tile_size = (nelem>TILE_SIZE)?TILE_SIZE:nelem;
  // OCCA+MacOS implementation need that for now (if DeviceID targets a CPU)
  const CeedInt tile_size = ocl?1:nelem_tile_size;
  occaPropertiesSet(pKR, "defines/TILE_SIZE", occaInt(tile_size));
  // ***************************************************************************
  char *oklPath;
  ierr = CeedOklPath_Occa(ceed,__FILE__, "ceed-occa-restrict",&oklPath);
  CeedChk(ierr);
  // ***************************************************************************
  data->kRestrict[0] = occaDeviceBuildKernel(dev, oklPath, "kRestrict0", pKR);
  data->kRestrict[1] = occaDeviceBuildKernel(dev, oklPath, "kRestrict1", pKR);
  data->kRestrict[2] = occaDeviceBuildKernel(dev, oklPath, "kRestrict2", pKR);
  data->kRestrict[3] = occaDeviceBuildKernel(dev, oklPath, "kRestrict3", pKR);
  data->kRestrict[4] = occaDeviceBuildKernel(dev, oklPath, "kRestrict4", pKR);
  data->kRestrict[5] = occaDeviceBuildKernel(dev, oklPath, "kRestrict5", pKR);
  data->kRestrict[6] = occaDeviceBuildKernel(dev, oklPath, "kRestrict6", pKR);
  // free local usage **********************************************************
  occaFree(pKR);
  ierr = CeedFree(&oklPath); CeedChk(ierr);
  dbg("[CeedElemRestriction][Create] done");
  // free indices as needed ****************************************************
  if (cmode == CEED_OWN_POINTER) {
    ierr = CeedFree(&indices); CeedChk(ierr);
  }
  ierr = CeedFree(&toffsets); CeedChk(ierr);
  ierr = CeedFree(&tindices); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
int CeedElemRestrictionCreateBlocked_Occa(const CeedMemType mtype,
    const CeedCopyMode cmode,
    const CeedInt *indices,
    const CeedElemRestriction r) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  return CeedError(ceed, 1, "Backend does not implement blocked restrictions");
}
