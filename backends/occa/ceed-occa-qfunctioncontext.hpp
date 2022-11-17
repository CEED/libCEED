// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_OCCA_QFUNCTIONCONTEXT_HEADER
#define CEED_OCCA_QFUNCTIONCONTEXT_HEADER

#include "ceed-occa-ceed-object.hpp"

namespace ceed {
namespace occa {
class QFunctionContext : public CeedObject {
 public:
  // Owned resources
  size_t         ctxSize;
  ::occa::memory memory;
  void          *hostBuffer;

  // Current resources
  ::occa::memory currentMemory;
  void          *currentHostBuffer;

  // State information
  int syncState;

  QFunctionContext();

  ~QFunctionContext();

  static QFunctionContext *getQFunctionContext(CeedQFunctionContext ctx, const bool assertValid = true);

  static QFunctionContext *from(CeedQFunctionContext ctx);

  ::occa::memory dataToMemory(const void *data) {
    ::occa::memory mem((::occa::modeMemory_t *)data);
    return mem;
  }

  void *memoryToData(::occa::memory &memory) { return memory.getModeMemory(); }

  void resizeCtx(const size_t ctxSize_);

  void resizeCtxMemory(const size_t ctxSize_);

  void resizeCtxMemory(::occa::device device, const size_t ctxSize_);

  void resizeHostCtxBuffer(const size_t ctxSize_);

  void setCurrentCtxMemoryIfNeeded();

  void setCurrentHostCtxBufferIfNeeded();

  void freeHostCtxBuffer();

  int hasValidData(bool *has_valid_data) const;

  int hasBorrowedDataOfType(CeedMemType mem_type, bool *has_borrowed_data_of_type) const;

  int setData(CeedMemType mtype, CeedCopyMode cmode, void *data);

  int copyDataValues(CeedMemType mtype, void *data);

  int ownDataPointer(CeedMemType mtype, void *data);

  int useDataPointer(CeedMemType mtype, void *data);

  int takeData(CeedMemType mtype, void *data);

  int getData(CeedMemType mtype, void *data);

  int restoreData();

  ::occa::memory getKernelArg();

  //---[ Ceed Callbacks ]-----------
  static int registerCeedFunction(Ceed ceed, CeedQFunctionContext ctx, const char *fname, ceed::occa::ceedFunction f);

  static int ceedCreate(CeedQFunctionContext ctx);

  static int ceedHasValidData(const CeedQFunctionContext ctx, bool *has_valid_data);

  static int ceedHasBorrowedDataOfType(const CeedQFunctionContext ctx, CeedMemType mem_type, bool *has_borrowed_data_of_type);

  static int ceedSetData(CeedQFunctionContext ctx, CeedMemType mtype, CeedCopyMode cmode, void *data);

  static int ceedTakeData(CeedQFunctionContext ctx, CeedMemType mtype, void *data);

  static int ceedGetData(CeedQFunctionContext ctx, CeedMemType mtype, void *data);

  static int ceedGetDataRead(CeedQFunctionContext ctx, CeedMemType mtype, void *data);

  static int ceedRestoreData(CeedQFunctionContext ctx);

  static int ceedDestroy(CeedQFunctionContext ctx);
};
}  // namespace occa
}  // namespace ceed

#endif
