// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_OCCA_VECTOR_HEADER
#define CEED_OCCA_VECTOR_HEADER

#include "ceed-occa-ceed-object.hpp"

namespace ceed {
namespace occa {
template <class TM>
::occa::memory arrayToMemory(const TM *array) {
  if (array) {
    ::occa::memory mem((::occa::modeMemory_t *)array);
    mem.setDtype(::occa::dtype::get<TM>());
    return mem;
  }
  return ::occa::null;
}

template <class TM>
TM *memoryToArray(::occa::memory &memory) {
  return (TM *)memory.getModeMemory();
}

class Vector : public CeedObject {
 public:
  // Owned resources
  CeedSize       length;
  ::occa::memory memory;
  CeedSize       hostBufferLength;
  CeedScalar    *hostBuffer;

  ::occa::kernel setValueKernel;

  // Current resources
  ::occa::memory currentMemory;
  CeedScalar    *currentHostBuffer;

  // State information
  int syncState;

  Vector();

  ~Vector();

  int hasValidArray(bool *has_valid_array);

  int hasBorrowedArrayOfType(CeedMemType mem_type, bool *has_borrowed_array_of_type);

  static Vector *getVector(CeedVector vec, const bool assertValid = true);

  static Vector *from(CeedVector vec);

  void resize(const CeedSize length_);

  void resizeMemory(const CeedSize length_);

  void resizeMemory(::occa::device device, const CeedSize length_);

  void resizeHostBuffer(const CeedSize length_);

  void setCurrentMemoryIfNeeded();

  void setCurrentHostBufferIfNeeded();

  void freeHostBuffer();

  int setValue(CeedScalar value);

  int setArray(CeedMemType mtype, CeedCopyMode cmode, CeedScalar *array);

  int takeArray(CeedMemType mtype, CeedScalar **array);

  int copyArrayValues(CeedMemType mtype, CeedScalar *array);

  int ownArrayPointer(CeedMemType mtype, CeedScalar *array);

  int useArrayPointer(CeedMemType mtype, CeedScalar *array);

  int getArray(CeedMemType mtype, CeedScalar **array);

  int getReadOnlyArray(CeedMemType mtype, CeedScalar **array);

  int getWriteOnlyArray(CeedMemType mtype, CeedScalar **array);

  int restoreArray(CeedScalar **array);

  int restoreReadOnlyArray(CeedScalar **array);

  ::occa::memory getKernelArg();

  ::occa::memory getConstKernelArg();

  void printValues(const std::string &name);
  void printNonZeroValues(const std::string &name);
  void printSummary(const std::string &name);

  //---[ Ceed Callbacks ]-----------
  static int registerCeedFunction(Ceed ceed, CeedVector vec, const char *fname, ceed::occa::ceedFunction f);

  static int ceedHasValidArray(CeedVector vec, bool *has_valid_array);

  static int ceedHasBorrowedArrayOfType(CeedVector vec, CeedMemType mem_type, bool *has_borrowed_array_of_type);

  static int ceedCreate(CeedSize length, CeedVector vec);

  static int ceedSetValue(CeedVector vec, CeedScalar value);

  static int ceedSetArray(CeedVector vec, CeedMemType mtype, CeedCopyMode cmode, CeedScalar *array);

  static int ceedTakeArray(CeedVector vec, CeedMemType mtype, CeedScalar **array);

  static int ceedGetArray(CeedVector vec, CeedMemType mtype, CeedScalar **array);

  static int ceedGetArrayRead(CeedVector vec, CeedMemType mtype, CeedScalar **array);

  static int ceedGetArrayWrite(CeedVector vec, CeedMemType mtype, CeedScalar **array);

  static int ceedRestoreArray(CeedVector vec, CeedScalar **array);

  static int ceedRestoreArrayRead(CeedVector vec, CeedScalar **array);

  static int ceedDestroy(CeedVector vec);
};
}  // namespace occa
}  // namespace ceed

#endif
