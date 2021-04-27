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

#include "ceed-occa-qfunctioncontext.hpp"

namespace ceed {
  namespace occa {
    QFunctionContext::QFunctionContext() :
        ctxSize(0),
        hostBuffer(NULL),
        currentHostBuffer(NULL),
        syncState(SyncState::none) {}

    QFunctionContext::~QFunctionContext() {
      memory.free();
      freeHostCtxBuffer();
    }

    QFunctionContext* QFunctionContext::from(CeedQFunctionContext ctx) {
      if (!ctx) {
        return NULL;
      }

      int ierr;

      QFunctionContext *ctx_ = NULL;
      ierr = CeedQFunctionContextGetBackendData(ctx, &ctx_); CeedOccaFromChk(ierr);

      ierr = CeedQFunctionContextGetContextSize(ctx, &ctx_->ctxSize);
      CeedOccaFromChk(ierr);

      if (ctx_ != NULL) {
        ierr = CeedQFunctionContextGetCeed(ctx, &ctx_->ceed); CeedOccaFromChk(ierr);
      }

      return ctx_;
    }

    void QFunctionContext::resizeCtx(const size_t ctxSize_) {
      ctxSize = ctxSize_;
    }

    void QFunctionContext::resizeCtxMemory(const size_t ctxSize_) {
      resizeCtxMemory(getDevice(), ctxSize_);
    }

    void QFunctionContext::resizeCtxMemory(::occa::device device, const size_t ctxSize_) {
      if (ctxSize_ != memory.size()) {
        memory.free();
        memory = device.malloc(ctxSize_);
      }
    }

    void QFunctionContext::resizeHostCtxBuffer(const size_t ctxSize_) {
      CeedFree(&hostBuffer);
      CeedMallocArray(1, ctxSize, &hostBuffer);
    }

    void QFunctionContext::setCurrentCtxMemoryIfNeeded() {
      if (!currentMemory.isInitialized()) {
        resizeCtxMemory(ctxSize);
        currentMemory = memory;
      }
    }

    void QFunctionContext::setCurrentHostCtxBufferIfNeeded() {
      if (!currentHostBuffer) {
        resizeHostCtxBuffer(ctxSize);
        currentHostBuffer = hostBuffer;
      }
    }

    void QFunctionContext::freeHostCtxBuffer() {
      if (hostBuffer) {
        CeedFree(&hostBuffer);
      }
    }

    int QFunctionContext::setData(CeedMemType mtype,
                                  CeedCopyMode cmode, void *data) {
      switch (cmode) {
        case CEED_COPY_VALUES:
          return copyDataValues(mtype, data);
        case CEED_OWN_POINTER:
          return ownDataPointer(mtype, data);
        case CEED_USE_POINTER:
          return useDataPointer(mtype, data);
      }
      return ceedError("Invalid CeedCopyMode passed");
    }

    int QFunctionContext::copyDataValues(CeedMemType mtype, void *data) {
      switch (mtype) {
        case CEED_MEM_HOST:
          setCurrentHostCtxBufferIfNeeded();
          ::memcpy(currentHostBuffer, data, ctxSize);
          syncState = SyncState::host;
          return CEED_ERROR_SUCCESS;
        case CEED_MEM_DEVICE:
          setCurrentCtxMemoryIfNeeded();
          currentMemory.copyFrom(dataToMemory(data));
          syncState = SyncState::device;
          return CEED_ERROR_SUCCESS;
      }
      return ceedError("Invalid CeedMemType passed");
    }

    int QFunctionContext::ownDataPointer(CeedMemType mtype, void *data) {
      switch (mtype) {
        case CEED_MEM_HOST:
          freeHostCtxBuffer();
          hostBuffer = currentHostBuffer = data;
          syncState = SyncState::host;
          return CEED_ERROR_SUCCESS;
        case CEED_MEM_DEVICE:
          memory.free();
          memory = currentMemory = dataToMemory(data);
          syncState = SyncState::device;
          return CEED_ERROR_SUCCESS;
      }
      return ceedError("Invalid CeedMemType passed");
    }

    int QFunctionContext::useDataPointer(CeedMemType mtype, void *data) {
      switch (mtype) {
        case CEED_MEM_HOST:
          freeHostCtxBuffer();
          currentHostBuffer = data;
          syncState = SyncState::host;
          return CEED_ERROR_SUCCESS;
        case CEED_MEM_DEVICE:
          memory.free();
          currentMemory = dataToMemory(data);
          syncState = SyncState::device;
          return CEED_ERROR_SUCCESS;
      }
      return ceedError("Invalid CeedMemType passed");
    }

    int QFunctionContext::getData(CeedMemType mtype,
                                  void *data) {
      // The passed `data` might be modified before restoring
      switch (mtype) {
        case CEED_MEM_HOST:
          setCurrentHostCtxBufferIfNeeded();
          if (syncState == SyncState::device) {
            setCurrentCtxMemoryIfNeeded();
            currentMemory.copyTo(currentHostBuffer);
          }
          syncState = SyncState::host;
          *(void **)data = currentHostBuffer;
          return CEED_ERROR_SUCCESS;
        case CEED_MEM_DEVICE:
          setCurrentCtxMemoryIfNeeded();
          if (syncState == SyncState::host) {
            setCurrentHostCtxBufferIfNeeded();
            currentMemory.copyFrom(currentHostBuffer);
          }
          syncState = SyncState::device;
          *(void **)data = memoryToData(currentMemory);
          return CEED_ERROR_SUCCESS;
      }
      return ceedError("Invalid CeedMemType passed");
    }

    int QFunctionContext::restoreData() {
      return CEED_ERROR_SUCCESS;
    }

    ::occa::memory QFunctionContext::getKernelArg() {
      setCurrentCtxMemoryIfNeeded();
      if (syncState == SyncState::host) {
        setCurrentHostCtxBufferIfNeeded();
        currentMemory.copyFrom(currentHostBuffer);
      }
      syncState = SyncState::device;
      return currentMemory;
    }

    //---[ Ceed Callbacks ]-----------
    int QFunctionContext::registerCeedFunction(Ceed ceed, CeedQFunctionContext ctx,
                                     const char *fname, ceed::occa::ceedFunction f) {
      return CeedSetBackendFunction(ceed, "QFunctionContext", ctx, fname, f);
    }

    int QFunctionContext::ceedCreate(CeedQFunctionContext ctx) {
      int ierr;

      Ceed ceed;
      ierr = CeedQFunctionContextGetCeed(ctx, &ceed); CeedChk(ierr);

      CeedOccaRegisterFunction(ctx, "SetData", QFunctionContext::ceedSetData);
      CeedOccaRegisterFunction(ctx, "GetData", QFunctionContext::ceedGetData);
      CeedOccaRegisterFunction(ctx, "RestoreData", QFunctionContext::ceedRestoreData);
      CeedOccaRegisterFunction(ctx, "Destroy", QFunctionContext::ceedDestroy);

      QFunctionContext *ctx_ = new QFunctionContext();
      ierr = CeedQFunctionContextSetBackendData(ctx, ctx_); CeedChk(ierr);

      return CEED_ERROR_SUCCESS;
    }

    int QFunctionContext::ceedSetData(CeedQFunctionContext ctx, CeedMemType mtype,
                                      CeedCopyMode cmode, void *data) {
      QFunctionContext *ctx_ = QFunctionContext::from(ctx);
      if (!ctx_) {
        return staticCeedError("Invalid CeedQFunctionContext passed");
      }
      return ctx_->setData(mtype, cmode, data);
    }

    int QFunctionContext::ceedGetData(CeedQFunctionContext ctx, CeedMemType mtype,
                                      void *data) {
      QFunctionContext *ctx_ = QFunctionContext::from(ctx);
      if (!ctx_) {
        return staticCeedError("Invalid CeedQFunctionContext passed");
      }
      return ctx_->getData(mtype, data);
    }

    int QFunctionContext::ceedRestoreData(CeedQFunctionContext ctx) {
      QFunctionContext *ctx_ = QFunctionContext::from(ctx);
      if (!ctx_) {
        return staticCeedError("Invalid CeedQFunctionContext passed");
      }
      return ctx_->restoreData();
    }

    int QFunctionContext::ceedDestroy(CeedQFunctionContext ctx) {
      delete QFunctionContext::from(ctx);
      return CEED_ERROR_SUCCESS;
    }
  }
}
