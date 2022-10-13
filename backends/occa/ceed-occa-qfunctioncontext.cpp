// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-occa-qfunctioncontext.hpp"

#include <cstring>

namespace ceed {
namespace occa {
QFunctionContext::QFunctionContext() : ctxSize(0), hostBuffer(NULL), currentHostBuffer(NULL), syncState(SyncState::none) {}

QFunctionContext::~QFunctionContext() {
  memory.free();
  freeHostCtxBuffer();
}

QFunctionContext *QFunctionContext::getQFunctionContext(CeedQFunctionContext ctx, const bool assertValid) {
  if (!ctx) {
    return NULL;
  }

  int               ierr;
  QFunctionContext *ctx_ = NULL;

  ierr = CeedQFunctionContextGetBackendData(ctx, &ctx_);
  if (assertValid) {
    CeedOccaFromChk(ierr);
  }

  return ctx_;
}

QFunctionContext *QFunctionContext::from(CeedQFunctionContext ctx) {
  QFunctionContext *ctx_ = getQFunctionContext(ctx);
  if (!ctx_) {
    return NULL;
  }

  CeedCallOcca(CeedQFunctionContextGetContextSize(ctx, &ctx_->ctxSize));

  if (ctx_ != NULL) {
    CeedCallOcca(CeedQFunctionContextGetCeed(ctx, &ctx_->ceed));
  }

  return ctx_;
}

void QFunctionContext::resizeCtx(const size_t ctxSize_) { ctxSize = ctxSize_; }

void QFunctionContext::resizeCtxMemory(const size_t ctxSize_) { resizeCtxMemory(getDevice(), ctxSize_); }

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

int QFunctionContext::hasValidData(bool *has_valid_data) const {
  (*has_valid_data) = (!!hostBuffer) || (!!currentHostBuffer) || (memory.isInitialized()) || (currentMemory.isInitialized());
  return CEED_ERROR_SUCCESS;
}

int QFunctionContext::hasBorrowedDataOfType(CeedMemType mem_type, bool *has_borrowed_data_of_type) const {
  switch (mem_type) {
    case CEED_MEM_HOST:
      (*has_borrowed_data_of_type) = !!currentHostBuffer;
      break;
    case CEED_MEM_DEVICE:
      (*has_borrowed_data_of_type) = currentMemory.isInitialized();
      break;
  }
  return CEED_ERROR_SUCCESS;
}

int QFunctionContext::setData(CeedMemType mtype, CeedCopyMode cmode, void *data) {
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
      std::memcpy(currentHostBuffer, data, ctxSize);
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
      syncState                      = SyncState::host;
      return CEED_ERROR_SUCCESS;
    case CEED_MEM_DEVICE:
      memory.free();
      memory = currentMemory = dataToMemory(data);
      syncState              = SyncState::device;
      return CEED_ERROR_SUCCESS;
  }
  return ceedError("Invalid CeedMemType passed");
}

int QFunctionContext::useDataPointer(CeedMemType mtype, void *data) {
  switch (mtype) {
    case CEED_MEM_HOST:
      freeHostCtxBuffer();
      currentHostBuffer = data;
      syncState         = SyncState::host;
      return CEED_ERROR_SUCCESS;
    case CEED_MEM_DEVICE:
      memory.free();
      currentMemory = dataToMemory(data);
      syncState     = SyncState::device;
      return CEED_ERROR_SUCCESS;
  }
  return ceedError("Invalid CeedMemType passed");
}

int QFunctionContext::takeData(CeedMemType mtype, void *data) {
  if (currentHostBuffer == NULL && currentMemory == ::occa::null) return ceedError("No context data set");
  switch (mtype) {
    case CEED_MEM_HOST:
      setCurrentHostCtxBufferIfNeeded();
      if (syncState == SyncState::device) {
        setCurrentCtxMemoryIfNeeded();
        currentMemory.copyTo(currentHostBuffer);
      }
      syncState         = SyncState::host;
      *(void **)data    = currentHostBuffer;
      hostBuffer        = NULL;
      currentHostBuffer = NULL;
      return CEED_ERROR_SUCCESS;
    case CEED_MEM_DEVICE:
      setCurrentCtxMemoryIfNeeded();
      if (syncState == SyncState::host) {
        setCurrentHostCtxBufferIfNeeded();
        currentMemory.copyFrom(currentHostBuffer);
      }
      syncState      = SyncState::device;
      *(void **)data = memoryToData(currentMemory);
      memory         = ::occa::null;
      currentMemory  = ::occa::null;
      return CEED_ERROR_SUCCESS;
  }
  return ceedError("Invalid CeedMemType passed");
}

int QFunctionContext::getData(CeedMemType mtype, void *data) {
  // The passed `data` might be modified before restoring
  if (currentHostBuffer == NULL && currentMemory == ::occa::null) return ceedError("No context data set");
  switch (mtype) {
    case CEED_MEM_HOST:
      setCurrentHostCtxBufferIfNeeded();
      if (syncState == SyncState::device) {
        setCurrentCtxMemoryIfNeeded();
        currentMemory.copyTo(currentHostBuffer);
      }
      syncState      = SyncState::host;
      *(void **)data = currentHostBuffer;
      return CEED_ERROR_SUCCESS;
    case CEED_MEM_DEVICE:
      setCurrentCtxMemoryIfNeeded();
      if (syncState == SyncState::host) {
        setCurrentHostCtxBufferIfNeeded();
        currentMemory.copyFrom(currentHostBuffer);
      }
      syncState      = SyncState::device;
      *(void **)data = memoryToData(currentMemory);
      return CEED_ERROR_SUCCESS;
  }
  return ceedError("Invalid CeedMemType passed");
}

int QFunctionContext::restoreData() { return CEED_ERROR_SUCCESS; }

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
int QFunctionContext::registerCeedFunction(Ceed ceed, CeedQFunctionContext ctx, const char *fname, ceed::occa::ceedFunction f) {
  return CeedSetBackendFunction(ceed, "QFunctionContext", ctx, fname, f);
}

int QFunctionContext::ceedCreate(CeedQFunctionContext ctx) {
  Ceed ceed;
  CeedCallBackend(CeedQFunctionContextGetCeed(ctx, &ceed));

  CeedOccaRegisterFunction(ctx, "HasValidData", QFunctionContext::ceedHasValidData);
  CeedOccaRegisterFunction(ctx, "HasBorrowedDataOfType", QFunctionContext::ceedHasBorrowedDataOfType);
  CeedOccaRegisterFunction(ctx, "SetData", QFunctionContext::ceedSetData);
  CeedOccaRegisterFunction(ctx, "TakeData", QFunctionContext::ceedTakeData);
  CeedOccaRegisterFunction(ctx, "GetData", QFunctionContext::ceedGetData);
  CeedOccaRegisterFunction(ctx, "GetDataRead", QFunctionContext::ceedGetDataRead);
  CeedOccaRegisterFunction(ctx, "RestoreData", QFunctionContext::ceedRestoreData);
  CeedOccaRegisterFunction(ctx, "Destroy", QFunctionContext::ceedDestroy);

  QFunctionContext *ctx_ = new QFunctionContext();
  CeedCallBackend(CeedQFunctionContextSetBackendData(ctx, ctx_));

  return CEED_ERROR_SUCCESS;
}

int QFunctionContext::ceedHasValidData(const CeedQFunctionContext ctx, bool *has_valid_data) {
  QFunctionContext *ctx_ = QFunctionContext::from(ctx);
  if (!ctx_) {
    return staticCeedError("Invalid CeedQFunctionContext passed");
  }
  return ctx_->hasValidData(has_valid_data);
}

int QFunctionContext::ceedHasBorrowedDataOfType(const CeedQFunctionContext ctx, CeedMemType mem_type, bool *has_borrowed_data_of_type) {
  QFunctionContext *ctx_ = QFunctionContext::from(ctx);
  if (!ctx_) {
    return staticCeedError("Invalid CeedQFunctionContext passed");
  }
  return ctx_->hasBorrowedDataOfType(mem_type, has_borrowed_data_of_type);
}

int QFunctionContext::ceedSetData(CeedQFunctionContext ctx, CeedMemType mtype, CeedCopyMode cmode, void *data) {
  QFunctionContext *ctx_ = QFunctionContext::from(ctx);
  if (!ctx_) {
    return staticCeedError("Invalid CeedQFunctionContext passed");
  }
  return ctx_->setData(mtype, cmode, data);
}

int QFunctionContext::ceedTakeData(CeedQFunctionContext ctx, CeedMemType mtype, void *data) {
  QFunctionContext *ctx_ = QFunctionContext::from(ctx);
  if (!ctx_) {
    return staticCeedError("Invalid CeedQFunctionContext passed");
  }
  return ctx_->takeData(mtype, data);
}

int QFunctionContext::ceedGetData(CeedQFunctionContext ctx, CeedMemType mtype, void *data) {
  QFunctionContext *ctx_ = QFunctionContext::from(ctx);
  if (!ctx_) {
    return staticCeedError("Invalid CeedQFunctionContext passed");
  }
  return ctx_->getData(mtype, data);
}

int QFunctionContext::ceedGetDataRead(CeedQFunctionContext ctx, CeedMemType mtype, void *data) {
  QFunctionContext *ctx_ = QFunctionContext::from(ctx);
  if (!ctx_) {
    return staticCeedError("Invalid CeedQFunctionContext passed");
  }
  // Todo: Determine if calling getData is sufficient
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
  delete getQFunctionContext(ctx, false);
  return CEED_ERROR_SUCCESS;
}
}  // namespace occa
}  // namespace ceed
