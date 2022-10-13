// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "./ceed-occa-elem-restriction.hpp"

#include <cstring>
#include <map>

#include "./ceed-occa-kernels.hpp"
#include "./ceed-occa-vector.hpp"

namespace ceed {
namespace occa {
ElemRestriction::ElemRestriction()
    : ceedElementCount(0),
      ceedElementSize(0),
      ceedComponentCount(0),
      ceedLVectorSize(0),
      ceedNodeStride(0),
      ceedComponentStride(0),
      ceedElementStride(0),
      ceedUnstridedComponentStride(0),
      freeHostIndices(true),
      hostIndices(NULL),
      freeIndices(true) {}

ElemRestriction::~ElemRestriction() {
  if (freeHostIndices) {
    CeedFree(&hostIndices);
  }
  if (freeIndices) {
    indices.free();
  }
}

void ElemRestriction::setup(CeedMemType memType, CeedCopyMode copyMode, const CeedInt *indicesInput) {
  if (memType == CEED_MEM_HOST) {
    setupFromHostMemory(copyMode, indicesInput);
  } else {
    setupFromDeviceMemory(copyMode, indicesInput);
  }

  setupTransposeIndices();
}

void ElemRestriction::setupFromHostMemory(CeedCopyMode copyMode, const CeedInt *indices_h) {
  const CeedInt entries = ceedElementCount * ceedElementSize;

  freeHostIndices = (copyMode == CEED_OWN_POINTER || copyMode == CEED_COPY_VALUES);

  if (copyMode != CEED_COPY_VALUES) {
    hostIndices = const_cast<CeedInt *>(indices_h);
  } else {
    const size_t bytes = entries * sizeof(CeedInt);
    hostIndices        = (CeedInt *)::malloc(bytes);
    std::memcpy(hostIndices, indices_h, bytes);
  }

  if (hostIndices) {
    indices = getDevice().malloc<CeedInt>(entries, hostIndices);
  }
}

void ElemRestriction::setupFromDeviceMemory(CeedCopyMode copyMode, const CeedInt *indices_d) {
  ::occa::memory deviceIndices = arrayToMemory(indices_d);

  freeIndices = (copyMode == CEED_OWN_POINTER);

  if (copyMode == CEED_COPY_VALUES) {
    indices = deviceIndices.clone();
  } else {
    indices = deviceIndices;
  }
}

bool ElemRestriction::usesIndices() { return indices.isInitialized(); }

void ElemRestriction::setupTransposeIndices() {
  if (!usesIndices() || transposeQuadIndices.isInitialized()) {
    return;
  }

  const CeedInt elementEntryCount = ceedElementCount * ceedElementSize;

  bool *indexIsUsed = new bool[ceedLVectorSize];
  std::memset(indexIsUsed, 0, ceedLVectorSize * sizeof(bool));

  for (CeedInt i = 0; i < elementEntryCount; ++i) {
    indexIsUsed[hostIndices[i]] = true;
  }

  CeedInt nodeCount = 0;
  for (CeedInt i = 0; i < ceedLVectorSize; ++i) {
    nodeCount += indexIsUsed[i];
  }

  const CeedInt dofOffsetCount         = nodeCount + 1;
  CeedInt      *quadIndexToDofOffset   = new CeedInt[ceedLVectorSize];
  CeedInt      *transposeQuadIndices_h = new CeedInt[nodeCount];
  CeedInt      *transposeDofOffsets_h  = new CeedInt[dofOffsetCount];
  CeedInt      *transposeDofIndices_h  = new CeedInt[elementEntryCount];

  std::memset(transposeDofOffsets_h, 0, dofOffsetCount * sizeof(CeedInt));

  // Compute ids
  CeedInt offsetId = 0;
  for (CeedInt i = 0; i < ceedLVectorSize; ++i) {
    if (indexIsUsed[i]) {
      transposeQuadIndices_h[offsetId] = i;
      quadIndexToDofOffset[i]          = offsetId++;
    }
  }

  // Count how many times a specific quad node is used
  for (CeedInt i = 0; i < elementEntryCount; ++i) {
    ++transposeDofOffsets_h[quadIndexToDofOffset[hostIndices[i]] + 1];
  }

  // Aggregate to find true offsets
  for (CeedInt i = 1; i < dofOffsetCount; ++i) {
    transposeDofOffsets_h[i] += transposeDofOffsets_h[i - 1];
  }

  // Compute dof indices
  for (CeedInt i = 0; i < elementEntryCount; ++i) {
    const CeedInt quadIndex         = hostIndices[i];
    const CeedInt dofIndex          = transposeDofOffsets_h[quadIndexToDofOffset[quadIndex]]++;
    transposeDofIndices_h[dofIndex] = i;
  }

  // Reset offsets
  for (int i = dofOffsetCount - 1; i > 0; --i) {
    transposeDofOffsets_h[i] = transposeDofOffsets_h[i - 1];
  }
  transposeDofOffsets_h[0] = 0;

  // Copy to device
  ::occa::device device = getDevice();

  transposeQuadIndices = device.malloc<CeedInt>(nodeCount, transposeQuadIndices_h);
  transposeDofOffsets  = device.malloc<CeedInt>(dofOffsetCount, transposeDofOffsets_h);
  transposeDofIndices  = device.malloc<CeedInt>(elementEntryCount, transposeDofIndices_h);

  // Clean up temporary arrays
  delete[] indexIsUsed;
  delete[] quadIndexToDofOffset;
  delete[] transposeQuadIndices_h;
  delete[] transposeDofOffsets_h;
  delete[] transposeDofIndices_h;
}

void ElemRestriction::setKernelProperties() {
  kernelProperties["defines/CeedInt"]                    = ::occa::dtype::get<CeedInt>().name();
  kernelProperties["defines/CeedScalar"]                 = ::occa::dtype::get<CeedScalar>().name();
  kernelProperties["defines/COMPONENT_COUNT"]            = ceedComponentCount;
  kernelProperties["defines/ELEMENT_SIZE"]               = ceedElementSize;
  kernelProperties["defines/TILE_SIZE"]                  = 64;
  kernelProperties["defines/USES_INDICES"]               = usesIndices();
  kernelProperties["defines/USER_STRIDES"]               = StrideType::USER_STRIDES;
  kernelProperties["defines/NOT_STRIDED"]                = StrideType::NOT_STRIDED;
  kernelProperties["defines/BACKEND_STRIDES"]            = StrideType::BACKEND_STRIDES;
  kernelProperties["defines/STRIDE_TYPE"]                = ceedStrideType;
  kernelProperties["defines/NODE_COUNT"]                 = transposeQuadIndices.length();
  kernelProperties["defines/NODE_STRIDE"]                = ceedNodeStride;
  kernelProperties["defines/COMPONENT_STRIDE"]           = ceedComponentStride;
  kernelProperties["defines/ELEMENT_STRIDE"]             = ceedElementStride;
  kernelProperties["defines/UNSTRIDED_COMPONENT_STRIDE"] = ceedUnstridedComponentStride;
}

ElemRestriction *ElemRestriction::getElemRestriction(CeedElemRestriction r, const bool assertValid) {
  if (!r || r == CEED_ELEMRESTRICTION_NONE) {
    return NULL;
  }

  int              ierr;
  ElemRestriction *elemRestriction = NULL;

  ierr = CeedElemRestrictionGetData(r, (void **)&elemRestriction);
  if (assertValid) {
    CeedOccaFromChk(ierr);
  }

  return elemRestriction;
}

ElemRestriction *ElemRestriction::from(CeedElemRestriction r) {
  ElemRestriction *elemRestriction = getElemRestriction(r);
  if (!elemRestriction) {
    return NULL;
  }

  CeedCallOcca(CeedElemRestrictionGetCeed(r, &elemRestriction->ceed));

  return elemRestriction->setupFrom(r);
}

ElemRestriction *ElemRestriction::from(CeedOperatorField operatorField) {
  CeedElemRestriction ceedElemRestriction;

  CeedCallOcca(CeedOperatorFieldGetElemRestriction(operatorField, &ceedElemRestriction));

  return from(ceedElemRestriction);
}

ElemRestriction *ElemRestriction::setupFrom(CeedElemRestriction r) {
  CeedCallOcca(CeedElemRestrictionGetNumElements(r, &ceedElementCount));

  CeedCallOcca(CeedElemRestrictionGetElementSize(r, &ceedElementSize));

  CeedCallOcca(CeedElemRestrictionGetNumComponents(r, &ceedComponentCount));

  CeedCallOcca(CeedElemRestrictionGetLVectorSize(r, &ceedLVectorSize));

  // Find what type of striding the restriction uses
  bool isStrided         = false;
  bool hasBackendStrides = false;

  CeedCallOcca(CeedElemRestrictionIsStrided(r, &isStrided));

  if (isStrided) {
    CeedCallOcca(CeedElemRestrictionHasBackendStrides(r, &hasBackendStrides));
  }

  if (isStrided) {
    if (hasBackendStrides) {
      ceedStrideType = BACKEND_STRIDES;
    } else {
      ceedStrideType = USER_STRIDES;
    }
  } else {
    ceedStrideType = NOT_STRIDED;
  }

  // Default strides
  ceedNodeStride               = 1;
  ceedComponentStride          = ceedElementSize;
  ceedElementStride            = ceedElementSize * ceedComponentCount;
  ceedUnstridedComponentStride = 1;

  if (ceedStrideType == USER_STRIDES) {
    CeedInt strides[3];

    CeedCallOcca(CeedElemRestrictionGetStrides(r, &strides));

    ceedNodeStride      = strides[0];
    ceedComponentStride = strides[1];
    ceedElementStride   = strides[2];

  } else if (ceedStrideType == NOT_STRIDED) {
    CeedCallOcca(CeedElemRestrictionGetCompStride(r, &ceedUnstridedComponentStride));
  }

  return this;
}

int ElemRestriction::apply(CeedTransposeMode rTransposeMode, Vector &u, Vector &v) {
  const bool rIsTransposed = (rTransposeMode != CEED_NOTRANSPOSE);

  // Todo: refactor
  if (rIsTransposed) {
    if (!restrictionTransposeKernel.isInitialized()) {
      setKernelProperties();
      restrictionTransposeKernel = getDevice().buildKernelFromString(occa_elem_restriction_source, "applyRestrictionTranspose", kernelProperties);
    }
    restrictionTransposeKernel(ceedElementCount, transposeQuadIndices, transposeDofOffsets, transposeDofIndices, u.getConstKernelArg(),
                               v.getKernelArg());
  } else {
    if (!restrictionKernel.isInitialized()) {
      setKernelProperties();
      restrictionKernel = getDevice().buildKernelFromString(occa_elem_restriction_source, "applyRestriction", kernelProperties);
    }
    restrictionKernel(ceedElementCount, indices, u.getConstKernelArg(), v.getKernelArg());
  }
  return CEED_ERROR_SUCCESS;
}

int ElemRestriction::getOffsets(CeedMemType memType, const CeedInt **offsets) {
  switch (memType) {
    case CEED_MEM_HOST: {
      *offsets = hostIndices;
      return CEED_ERROR_SUCCESS;
    }
    case CEED_MEM_DEVICE: {
      *offsets = memoryToArray<CeedInt>(indices);
      return CEED_ERROR_SUCCESS;
    }
  }
  return ceedError("Unsupported CeedMemType passed to ElemRestriction::getOffsets");
}

//---[ Ceed Callbacks ]-----------
int ElemRestriction::registerCeedFunction(Ceed ceed, CeedElemRestriction r, const char *fname, ceed::occa::ceedFunction f) {
  return CeedSetBackendFunction(ceed, "ElemRestriction", r, fname, f);
}

int ElemRestriction::ceedCreate(CeedMemType memType, CeedCopyMode copyMode, const CeedInt *indicesInput, CeedElemRestriction r) {
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));

  if ((memType != CEED_MEM_DEVICE) && (memType != CEED_MEM_HOST)) {
    return staticCeedError("Only HOST and DEVICE CeedMemType supported");
  }

  ElemRestriction *elemRestriction = new ElemRestriction();
  CeedCallBackend(CeedElemRestrictionSetData(r, elemRestriction));

  // Setup Ceed objects before setting up memory
  elemRestriction = ElemRestriction::from(r);
  elemRestriction->setup(memType, copyMode, indicesInput);

  CeedInt defaultLayout[3] = {1, elemRestriction->ceedElementSize, elemRestriction->ceedElementSize * elemRestriction->ceedComponentCount};
  CeedChkBackend(CeedElemRestrictionSetELayout(r, defaultLayout));

  CeedOccaRegisterFunction(r, "Apply", ElemRestriction::ceedApply);
  CeedOccaRegisterFunction(r, "ApplyBlock", ElemRestriction::ceedApplyBlock);
  CeedOccaRegisterFunction(r, "GetOffsets", ElemRestriction::ceedGetOffsets);
  CeedOccaRegisterFunction(r, "Destroy", ElemRestriction::ceedDestroy);

  return CEED_ERROR_SUCCESS;
}

int ElemRestriction::ceedCreateBlocked(CeedMemType memType, CeedCopyMode copyMode, const CeedInt *indicesInput, CeedElemRestriction r) {
  return staticCeedError("(OCCA) Backend does not implement CeedElemRestrictionCreateBlocked");
}

int ElemRestriction::ceedApply(CeedElemRestriction r, CeedTransposeMode tmode, CeedVector u, CeedVector v, CeedRequest *request) {
  ElemRestriction *elemRestriction = ElemRestriction::from(r);
  Vector          *uVector         = Vector::from(u);
  Vector          *vVector         = Vector::from(v);

  if (!elemRestriction) {
    return staticCeedError("Incorrect CeedElemRestriction argument: r");
  }
  if (!uVector) {
    return elemRestriction->ceedError("Incorrect CeedVector argument: u");
  }
  if (!vVector) {
    return elemRestriction->ceedError("Incorrect CeedVector argument: v");
  }

  return elemRestriction->apply(tmode, *uVector, *vVector);
}

int ElemRestriction::ceedApplyBlock(CeedElemRestriction r, CeedInt block, CeedTransposeMode tmode, CeedVector u, CeedVector v, CeedRequest *request) {
  return staticCeedError("(OCCA) Backend does not implement CeedElemRestrictionApplyBlock");
}

int ElemRestriction::ceedGetOffsets(CeedElemRestriction r, CeedMemType memType, const CeedInt **offsets) {
  ElemRestriction *elemRestriction = ElemRestriction::from(r);

  if (!elemRestriction) {
    return staticCeedError("Incorrect CeedElemRestriction argument: r");
  }

  return elemRestriction->getOffsets(memType, offsets);
}

int ElemRestriction::ceedDestroy(CeedElemRestriction r) {
  delete getElemRestriction(r, false);
  return CEED_ERROR_SUCCESS;
}
}  // namespace occa
}  // namespace ceed
