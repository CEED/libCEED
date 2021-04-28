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

#ifndef CEED_OCCA_QFUNCTIONCONTEXT_HEADER
#define CEED_OCCA_QFUNCTIONCONTEXT_HEADER

#include "ceed-occa-ceed-object.hpp"

namespace ceed {
  namespace occa {
    class QFunctionContext : public CeedObject {
     public:
      // Owned resources
      size_t ctxSize;
      ::occa::memory memory;
      void *hostBuffer;

      // Current resources
      ::occa::memory currentMemory;
      void *currentHostBuffer;

      // State information
      int syncState;

      QFunctionContext();

      ~QFunctionContext();

      static QFunctionContext* from(CeedQFunctionContext ctx);

      ::occa::memory dataToMemory(const void *data) {
        ::occa::memory mem((::occa::modeMemory_t*) data);
        return mem;
      }

      void* memoryToData(::occa::memory &memory) {
        return memory.getModeMemory();
      }

      void resizeCtx(const size_t ctxSize_);

      void resizeCtxMemory(const size_t ctxSize_);

      void resizeCtxMemory(::occa::device device, const size_t ctxSize_);

      void resizeHostCtxBuffer(const size_t ctxSize_);

      void setCurrentCtxMemoryIfNeeded();

      void setCurrentHostCtxBufferIfNeeded();

      void freeHostCtxBuffer();

      int setData(CeedMemType mtype, CeedCopyMode cmode, void *data);

      int copyDataValues(CeedMemType mtype, void *data);

      int ownDataPointer(CeedMemType mtype, void *data);

      int useDataPointer(CeedMemType mtype, void *data);

      int takeData(CeedMemType mtype, void *data);

      int getData(CeedMemType mtype, void *data);

      int restoreData();

      ::occa::memory getKernelArg();

      //---[ Ceed Callbacks ]-----------
      static int registerCeedFunction(Ceed ceed, CeedQFunctionContext ctx,
                                      const char *fname, ceed::occa::ceedFunction f);

      static int ceedCreate(CeedQFunctionContext ctx);

      static int ceedSetData(CeedQFunctionContext ctx, CeedMemType mtype,
                             CeedCopyMode cmode, void *data);

      static int ceedTakeData(CeedQFunctionContext ctx, CeedMemType mtype,
                              void *data);

      static int ceedGetData(CeedQFunctionContext ctx, CeedMemType mtype,
                             void *data);

      static int ceedRestoreData(CeedQFunctionContext ctx);

      static int ceedDestroy(CeedQFunctionContext ctx);
    };
  }
}

#endif
