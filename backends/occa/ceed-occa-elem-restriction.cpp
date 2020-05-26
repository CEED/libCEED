// Copyright (c) 2019, Lawrence Livermore National Security, LLC.
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

#include <map>

#include "./ceed-occa-elem-restriction.hpp"
#include "./ceed-occa-kernels.hpp"
#include "./ceed-occa-vector.hpp"

namespace ceed {
  namespace occa {
    ElemRestriction::ElemRestriction() :
        ceedElementCount(0),
        ceedElementSize(0),
        ceedComponentCount(0),
        ceedLVectorSize(0),
        ceedNodeStride(0),
        ceedComponentStride(0),
        ceedElementStride(0),
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

    void ElemRestriction::setup(CeedMemType memType,
                                CeedCopyMode copyMode,
                                const CeedInt *indicesInput) {
      if (memType == CEED_MEM_HOST) {
        setupFromHostMemory(copyMode, indicesInput);
      } else {
        setupFromDeviceMemory(copyMode, indicesInput);
      }

      setupTransposeIndices();

      setupKernelBuilders();
    }

    void ElemRestriction::setupFromHostMemory(CeedCopyMode copyMode,
                                              const CeedInt *indices_h) {
      const CeedInt entries = ceedElementCount * ceedElementSize;

      freeHostIndices = (copyMode == CEED_OWN_POINTER || copyMode == CEED_COPY_VALUES);

      if (copyMode != CEED_COPY_VALUES) {
        hostIndices = const_cast<CeedInt*>(indices_h);
      } else {
        const size_t bytes = entries * sizeof(CeedInt);
        hostIndices = (CeedInt*) ::malloc(bytes);
        ::memcpy(hostIndices, indices_h, bytes);
      }

      if (hostIndices) {
        indices = getDevice().malloc<CeedInt>(entries, hostIndices);
      }
    }

    void ElemRestriction::setupFromDeviceMemory(CeedCopyMode copyMode,
                                                const CeedInt *indices_d) {
      ::occa::memory deviceIndices = arrayToMemory(indices_d);

      freeIndices = (copyMode == CEED_OWN_POINTER);

      if (copyMode == CEED_COPY_VALUES) {
        indices = deviceIndices.clone();
      } else {
        indices = deviceIndices;
      }
    }

    bool ElemRestriction::usesIndices() {
      return indices.isInitialized();
    }

    void ElemRestriction::setupTransposeIndices() {
      if (!usesIndices() || transposeQuadIndices.isInitialized()) {
        return;
      }

      const CeedInt elementEntryCount = ceedElementCount * ceedElementSize;

      bool *indexIsUsed = new bool[ceedLVectorSize];
      ::memset(indexIsUsed, 0, ceedLVectorSize * sizeof(bool));

      for (CeedInt i = 0; i < elementEntryCount; ++i) {
        indexIsUsed[hostIndices[i]] = true;
      }

      CeedInt nodeCount = 0;
      for (CeedInt i = 0; i < ceedLVectorSize; ++i) {
        nodeCount += indexIsUsed[i];
      }

      const CeedInt dofOffsetCount = nodeCount + 1;
      CeedInt *quadIndexToDofOffset   = new CeedInt[ceedLVectorSize];
      CeedInt *transposeQuadIndices_h = new CeedInt[nodeCount];
      CeedInt *transposeDofOffsets_h  = new CeedInt[dofOffsetCount];
      CeedInt *transposeDofIndices_h  = new CeedInt[elementEntryCount];

      ::memset(transposeDofOffsets_h, 0, dofOffsetCount * sizeof(CeedInt));

      // Compute ids
      CeedInt offsetId = 0;
      for (CeedInt i = 0; i < ceedLVectorSize; ++i) {
        if (indexIsUsed[i]) {
          transposeQuadIndices_h[offsetId] = i;
          quadIndexToDofOffset[i] = offsetId++;
        }
      }

      // Count how many times a specific quad node is used
      for (CeedInt i = 0; i < elementEntryCount; ++i) {
        ++transposeDofOffsets_h[
          quadIndexToDofOffset[hostIndices[i]] + 1
        ];
      }

      // Aggregate to find true offsets
      for (CeedInt i = 1; i < dofOffsetCount; ++i) {
        transposeDofOffsets_h[i] += transposeDofOffsets_h[i - 1];
      }

      // Compute dof indices
      for (CeedInt i = 0; i < elementEntryCount; ++i) {
        const CeedInt quadIndex = hostIndices[i];
        const CeedInt dofIndex = transposeDofOffsets_h[
          quadIndexToDofOffset[quadIndex]
        ]++;
        transposeDofIndices_h[dofIndex] = i;
      }

      // Reset offsets
      for (int i = dofOffsetCount - 1; i > 0; --i) {
        transposeDofOffsets_h[i] = transposeDofOffsets_h[i - 1];
      }
      transposeDofOffsets_h[0] = 0;

      // Copy to device
      ::occa::device device = getDevice();

      transposeQuadIndices = device.malloc<CeedInt>(nodeCount,
                                                    transposeQuadIndices_h);
      transposeDofOffsets = device.malloc<CeedInt>(dofOffsetCount,
                                                   transposeDofOffsets_h);
      transposeDofIndices = device.malloc<CeedInt>(elementEntryCount,
                                                   transposeDofIndices_h);

      // Clean up temporary arrays
      delete [] indexIsUsed;
      delete [] quadIndexToDofOffset;
      delete [] transposeQuadIndices_h;
      delete [] transposeDofOffsets_h;
      delete [] transposeDofIndices_h;
    }

    void ElemRestriction::setupKernelBuilders() {
      ::occa::properties kernelProps;
      kernelProps["defines/CeedInt"]    = ::occa::dtype::get<CeedInt>().name();
      kernelProps["defines/CeedScalar"] = ::occa::dtype::get<CeedScalar>().name();

      kernelProps["defines/COMPONENT_COUNT"] = ceedComponentCount;
      kernelProps["defines/ELEMENT_SIZE"]    = ceedElementSize;
      kernelProps["defines/TILE_SIZE"]       = 64;
      kernelProps["defines/USES_INDICES"]    = usesIndices();

      applyKernelBuilder = ::occa::kernelBuilder::fromString(
        occa_elem_restriction_source, "applyRestriction", kernelProps
      );

      applyTransposeKernelBuilder = ::occa::kernelBuilder::fromString(
        occa_elem_restriction_source, "applyRestrictionTranspose", kernelProps
      );
    }

    ElemRestriction* ElemRestriction::from(CeedElemRestriction r) {
      if (!r || r == CEED_ELEMRESTRICTION_NONE) {
        return NULL;
      }

      int ierr;
      ElemRestriction *elemRestriction;

      ierr = CeedElemRestrictionGetData(r, (void**) &elemRestriction);
      CeedOccaFromChk(ierr);

      ierr = CeedElemRestrictionGetCeed(r, &elemRestriction->ceed);
      CeedOccaFromChk(ierr);

      return elemRestriction->setupFrom(r);
    }

    ElemRestriction* ElemRestriction::from(CeedOperatorField operatorField) {
      int ierr;
      CeedElemRestriction ceedElemRestriction;

      ierr = CeedOperatorFieldGetElemRestriction(operatorField, &ceedElemRestriction);
      CeedOccaFromChk(ierr);

      return from(ceedElemRestriction);
    }

    ElemRestriction* ElemRestriction::setupFrom(CeedElemRestriction r) {
      int ierr;

      ierr = CeedElemRestrictionGetNumElements(r, &ceedElementCount);
      CeedOccaFromChk(ierr);

      ierr = CeedElemRestrictionGetElementSize(r, &ceedElementSize);
      CeedOccaFromChk(ierr);

      ierr = CeedElemRestrictionGetNumComponents(r, &ceedComponentCount);
      CeedOccaFromChk(ierr);

      ierr = CeedElemRestrictionGetLVectorSize(r, &ceedLVectorSize);
      CeedOccaFromChk(ierr);

      // Find what type of striding the restriction uses
      bool isStrided = false;
      bool hasBackendStrides = false;

      ierr = CeedElemRestrictionGetStridedStatus(r, &isStrided);
      CeedOccaFromChk(ierr);

      if (isStrided) {
        ierr = CeedElemRestrictionGetBackendStridesStatus(r, &hasBackendStrides);
        CeedOccaFromChk(ierr);
      }

      if (isStrided) {
        if (hasBackendStrides) {
          ceedStrideType = FULLY_STRIDED;
        } else {
          ceedStrideType = NOT_STRIDED;
        }
      } else {
        ceedStrideType = COMPONENT_STRIDED;
      }

      if (ceedStrideType == FULLY_STRIDED) {
        CeedInt strides[3];

        ierr = CeedElemRestrictionGetStrides(r, &strides);
        CeedOccaFromChk(ierr);

        ceedNodeStride      = strides[0];
        ceedComponentStride = strides[1];
        ceedElementStride   = strides[2];
      } else if (ceedStrideType == COMPONENT_STRIDED) {
        ierr = CeedElemRestrictionGetCompStride(r, &ceedComponentStride);
        CeedOccaFromChk(ierr);
      } else {
        ceedNodeStride      = 1;
        ceedComponentStride = ceedElementSize * ceedElementCount;
        ceedElementStride   = ceedElementSize;
      }

      return this;
    }

    int ElemRestriction::apply(CeedTransposeMode rTransposeMode,
                               Vector &u,
                               Vector &v) {
      const bool rIsTransposed = (rTransposeMode != CEED_NOTRANSPOSE);

      ::occa::properties kernelProps;
      kernelProps["defines/FULLY_STRIDED"]     = StrideType::FULLY_STRIDED;
      kernelProps["defines/COMPONENT_STRIDED"] = StrideType::COMPONENT_STRIDED;
      kernelProps["defines/NOT_STRIDED"]       = StrideType::NOT_STRIDED;
      kernelProps["defines/STRIDE_TYPE"]       = ceedStrideType;

      kernelProps["defines/NODE_COUNT"]       = transposeQuadIndices.length();
      kernelProps["defines/NODE_STRIDE"]      = ceedNodeStride;
      kernelProps["defines/COMPONENT_STRIDE"] = ceedComponentStride;
      kernelProps["defines/ELEMENT_STRIDE"]   = ceedElementStride;

      if (rIsTransposed) {
        ::occa::kernel applyTranspose = applyTransposeKernelBuilder.build(
          getDevice(),
          kernelProps
        );

        applyTranspose(ceedElementCount,
                       transposeQuadIndices,
                       transposeDofOffsets,
                       transposeDofIndices,
                       u.getConstKernelArg(),
                       v.getKernelArg());
      } else {
        ::occa::kernel apply = applyKernelBuilder.build(
          getDevice(),
          kernelProps
        );

        apply(ceedElementCount,
              indices,
              u.getConstKernelArg(),
              v.getKernelArg());
      }

      return 0;
    }

    int ElemRestriction::getOffsets(CeedMemType memType,
                                    const CeedInt **offsets) {
      switch (memType) {
        case CEED_MEM_HOST: {
          *offsets = hostIndices;
          return 0;
        }
        case CEED_MEM_DEVICE: {
          *offsets = memoryToArray<CeedInt>(indices);
          return 0;
        }
      }
      return ceedError("Unsupported CeedMemType passed to ElemRestriction::getOffsets");
    }

    //---[ Ceed Callbacks ]-----------
    int ElemRestriction::registerRestrictionFunction(Ceed ceed, CeedElemRestriction r,
                                                     const char *fname, ceed::occa::ceedFunction f) {
      return CeedSetBackendFunction(ceed, "ElemRestriction", r, fname, f);
    }

    int ElemRestriction::ceedCreate(CeedMemType memType,
                                    CeedCopyMode copyMode,
                                    const CeedInt *indicesInput,
                                    CeedElemRestriction r) {
      int ierr;
      Ceed ceed;
      ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);

      if ((memType != CEED_MEM_DEVICE) && (memType != CEED_MEM_HOST)) {
        return staticCeedError("Only HOST and DEVICE CeedMemType supported");
      }

      ElemRestriction *elemRestriction = new ElemRestriction();
      ierr = CeedElemRestrictionSetData(r, (void**) &elemRestriction); CeedChk(ierr);

      // Setup Ceed objects before setting up memory
      elemRestriction = ElemRestriction::from(r);
      elemRestriction->setup(memType, copyMode, indicesInput);

      ierr = registerRestrictionFunction(ceed, r, "Apply",
                                         (ceed::occa::ceedFunction) ElemRestriction::ceedApply);
      CeedChk(ierr);

      ierr = registerRestrictionFunction(ceed, r, "ApplyBlock",
                                         (ceed::occa::ceedFunction) ElemRestriction::ceedApplyBlock);
      CeedChk(ierr);

      ierr = registerRestrictionFunction(ceed, r, "GetOffsets",
                                         (ceed::occa::ceedFunction) ElemRestriction::ceedGetOffsets);
      CeedChk(ierr);

      ierr = registerRestrictionFunction(ceed, r, "Destroy",
                                         (ceed::occa::ceedFunction) ElemRestriction::ceedDestroy);
      CeedChk(ierr);

      return 0;
    }

    int ElemRestriction::ceedCreateBlocked(CeedMemType memType,
                                           CeedCopyMode copyMode,
                                           const CeedInt *indicesInput,
                                           CeedElemRestriction r) {
      return staticCeedError("(OCCA) Backend does not implement CeedElemRestrictionCreateBlocked");
    }

    int ElemRestriction::ceedApply(CeedElemRestriction r,
                                   CeedTransposeMode tmode,
                                   CeedVector u, CeedVector v,
                                   CeedRequest *request) {
      ElemRestriction *elemRestriction = ElemRestriction::from(r);
      Vector *uVector = Vector::from(u);
      Vector *vVector = Vector::from(v);

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

    int ElemRestriction::ceedApplyBlock(CeedElemRestriction r,
                                        CeedInt block, CeedTransposeMode tmode,
                                        CeedVector u, CeedVector v,
                                        CeedRequest *request) {
      return staticCeedError("(OCCA) Backend does not implement CeedElemRestrictionApplyBlock");
    }

    int ElemRestriction::ceedGetOffsets(CeedElemRestriction r,
                                        CeedMemType memType,
                                        const CeedInt **offsets) {
      ElemRestriction *elemRestriction = ElemRestriction::from(r);

      if (!elemRestriction) {
        return staticCeedError("Incorrect CeedElemRestriction argument: r");
      }

      return elemRestriction->getOffsets(memType, offsets);
    }

    int ElemRestriction::ceedDestroy(CeedElemRestriction r) {
      delete ElemRestriction::from(r);
      return 0;
    }
  }
}
