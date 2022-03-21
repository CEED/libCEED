// Copyright (c) 2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#ifndef CEED_OCCA_ELEMRESTRICTION_HEADER
#define CEED_OCCA_ELEMRESTRICTION_HEADER

#include "ceed-occa-ceed-object.hpp"
#include "ceed-occa-vector.hpp"

namespace ceed {
  namespace occa {
    enum StrideType {
      BACKEND_STRIDES = 0,
      USER_STRIDES = 1,
      NOT_STRIDED = 2,
    };

    class ElemRestriction : public CeedObject {
     public:
      // Ceed object information
      CeedInt ceedElementCount;
      CeedInt ceedElementSize;
      CeedInt ceedComponentCount;
      CeedSize ceedLVectorSize;
      StrideType ceedStrideType;
      CeedInt ceedNodeStride;
      CeedInt ceedComponentStride;
      CeedInt ceedElementStride;
      CeedInt ceedUnstridedComponentStride;

      // Passed resources
      bool freeHostIndices;
      CeedInt *hostIndices;

      // Owned resources
      bool freeIndices;
      ::occa::memory indices;

      ::occa::memory transposeQuadIndices;
      ::occa::memory transposeDofOffsets;
      ::occa::memory transposeDofIndices;

      ::occa::kernelBuilder applyKernelBuilder;
      ::occa::kernelBuilder applyTransposeKernelBuilder;

      ElemRestriction();

      ~ElemRestriction();

      void setup(CeedMemType memType,
                 CeedCopyMode copyMode,
                 const CeedInt *indicesInput);

      void setupFromHostMemory(CeedCopyMode copyMode,
                               const CeedInt *indices_h);

      void setupFromDeviceMemory(CeedCopyMode copyMode,
                                 const CeedInt *indices_d);

      bool usesIndices();

      void setupTransposeIndices();

      void setupKernelBuilders();

      static ElemRestriction* getElemRestriction(CeedElemRestriction r,
                                                 const bool assertValid = true);

      static ElemRestriction* from(CeedElemRestriction r);
      static ElemRestriction* from(CeedOperatorField operatorField);
      ElemRestriction* setupFrom(CeedElemRestriction r);

      int apply(CeedTransposeMode rTransposeMode,
                Vector &u,
                Vector &v);

      int getOffsets(CeedMemType memType,
                     const CeedInt **offsets);

      //---[ Ceed Callbacks ]-----------
      static int registerCeedFunction(Ceed ceed, CeedElemRestriction r,
                                      const char *fname, ceed::occa::ceedFunction f);

      static int ceedCreate(CeedMemType memType,
                            CeedCopyMode copyMode,
                            const CeedInt *indicesInput,
                            CeedElemRestriction r);

      static int ceedCreateBlocked(CeedMemType memType,
                                   CeedCopyMode copyMode,
                                   const CeedInt *indicesInput,
                                   CeedElemRestriction r);

      static int ceedApply(CeedElemRestriction r,
                           CeedTransposeMode tmode,
                           CeedVector u, CeedVector v, CeedRequest *request);

      static int ceedGetOffsets(CeedElemRestriction r,
                                CeedMemType memType,
                                const CeedInt **offsets);

      static int ceedApplyBlock(CeedElemRestriction r,
                                CeedInt block, CeedTransposeMode tmode,
                                CeedVector u, CeedVector v,
                                CeedRequest *request);

      static int ceedDestroy(CeedElemRestriction r);
    };
  }
}

#endif
