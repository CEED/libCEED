// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-occa-vector.hpp"

#include <cstring>

#include "ceed-occa-kernels.hpp"

namespace ceed {
namespace occa {
Vector::Vector() : length(0), hostBufferLength(0), hostBuffer(NULL), currentHostBuffer(NULL), syncState(SyncState::none) {}

Vector::~Vector() {
  memory.free();
  freeHostBuffer();
}

int Vector::hasValidArray(bool *has_valid_array) {
  (*has_valid_array) = (!!hostBuffer) || (!!currentHostBuffer) || (memory.isInitialized()) || (currentMemory.isInitialized());
  return CEED_ERROR_SUCCESS;
}

int Vector::hasBorrowedArrayOfType(CeedMemType mem_type, bool *has_borrowed_array_of_type) {
  switch (mem_type) {
    case CEED_MEM_HOST:
      (*has_borrowed_array_of_type) = !!currentHostBuffer;
      break;
    case CEED_MEM_DEVICE:
      (*has_borrowed_array_of_type) = currentMemory.isInitialized();
      break;
  }
  return CEED_ERROR_SUCCESS;
}

Vector *Vector::getVector(CeedVector vec, const bool assertValid) {
  if (!vec || vec == CEED_VECTOR_NONE) {
    return NULL;
  }

  int     ierr;
  Vector *vector = NULL;

  ierr = CeedVectorGetData(vec, &vector);
  if (assertValid) {
    CeedOccaFromChk(ierr);
  }

  return vector;
}

Vector *Vector::from(CeedVector vec) {
  Vector *vector = getVector(vec);
  if (!vector) {
    return NULL;
  }

  CeedCallOcca(CeedVectorGetCeed(vec, &vector->ceed));
  CeedCallOcca(CeedVectorGetLength(vec, &vector->length));

  return vector;
}

void Vector::resize(const CeedSize length_) { length = length_; }

void Vector::resizeMemory(const CeedSize length_) { resizeMemory(getDevice(), length_); }

void Vector::resizeMemory(::occa::device device, const CeedSize length_) {
  if (length_ != (CeedSize)memory.length()) {
    memory.free();
    memory = device.malloc<CeedScalar>(length_);
  }
}

void Vector::resizeHostBuffer(const CeedSize length_) {
  if (length_ != hostBufferLength) {
    delete hostBuffer;
    hostBuffer = new CeedScalar[length_];
  }
}

void Vector::setCurrentMemoryIfNeeded() {
  if (!currentMemory.isInitialized()) {
    resizeMemory(length);
    currentMemory = memory;
  }
}

void Vector::setCurrentHostBufferIfNeeded() {
  if (!currentHostBuffer) {
    resizeHostBuffer(length);
    currentHostBuffer = hostBuffer;
  }
}

void Vector::freeHostBuffer() {
  if (hostBuffer) {
    delete[] hostBuffer;
    hostBuffer = NULL;
  }
}

int Vector::setValue(CeedScalar value) {
  // Prioritize keeping data in the device
  if (syncState & SyncState::device) {
    setCurrentMemoryIfNeeded();
    if (!setValueKernel.isInitialized()) {
      ::occa::json kernelProperties;
      CeedInt constexpr block_size{256};
      kernelProperties["defines/CeedInt"]    = ::occa::dtype::get<CeedInt>().name();
      kernelProperties["defines/CeedScalar"] = ::occa::dtype::get<CeedScalar>().name();
      kernelProperties["defines/BLOCK_SIZE"] = block_size;

      std::string kernelSource = occa_set_value_source;
      setValueKernel           = getDevice().buildKernelFromString(kernelSource, "setValue", kernelProperties);
      setValueKernel(currentMemory, value, length);
    }
    syncState = SyncState::device;
  } else {
    setCurrentHostBufferIfNeeded();
    for (CeedInt i = 0; i < length; ++i) {
      currentHostBuffer[i] = value;
    }
    syncState = SyncState::host;
  }
  return CEED_ERROR_SUCCESS;
}

int Vector::setArray(CeedMemType mtype, CeedCopyMode cmode, CeedScalar *array) {
  switch (cmode) {
    case CEED_COPY_VALUES:
      return copyArrayValues(mtype, array);
    case CEED_OWN_POINTER:
      return ownArrayPointer(mtype, array);
    case CEED_USE_POINTER:
      return useArrayPointer(mtype, array);
  }
  return ceedError("Invalid CeedCopyMode passed");
}

int Vector::takeArray(CeedMemType mtype, CeedScalar **array) {
  switch (mtype) {
    case CEED_MEM_HOST:
      setCurrentHostBufferIfNeeded();
      if (syncState == SyncState::device) {
        setCurrentMemoryIfNeeded();
        currentMemory.copyTo(currentHostBuffer);
      }
      *array            = currentHostBuffer;
      hostBuffer        = NULL;
      currentHostBuffer = NULL;

      syncState = SyncState::host;
      return CEED_ERROR_SUCCESS;
    case CEED_MEM_DEVICE:
      setCurrentMemoryIfNeeded();
      if (syncState == SyncState::host) {
        setCurrentHostBufferIfNeeded();
        currentMemory.copyFrom(currentHostBuffer);
      }
      *array        = memoryToArray<CeedScalar>(currentMemory);
      memory        = ::occa::null;
      currentMemory = ::occa::null;

      syncState = SyncState::device;
      return CEED_ERROR_SUCCESS;
  }
  return ceedError("Invalid CeedMemType passed");
}

int Vector::copyArrayValues(CeedMemType mtype, CeedScalar *array) {
  switch (mtype) {
    case CEED_MEM_HOST:
      setCurrentHostBufferIfNeeded();
      if (array) {
        std::memcpy(currentHostBuffer, array, length * sizeof(CeedScalar));
      }
      syncState = SyncState::host;
      return CEED_ERROR_SUCCESS;
    case CEED_MEM_DEVICE:
      setCurrentMemoryIfNeeded();
      if (array) {
        currentMemory.copyFrom(arrayToMemory(array));
      }
      syncState = SyncState::device;
      return CEED_ERROR_SUCCESS;
  }
  return ceedError("Invalid CeedMemType passed");
}

int Vector::ownArrayPointer(CeedMemType mtype, CeedScalar *array) {
  switch (mtype) {
    case CEED_MEM_HOST:
      freeHostBuffer();
      hostBuffer = currentHostBuffer = array;
      syncState                      = SyncState::host;
      return CEED_ERROR_SUCCESS;
    case CEED_MEM_DEVICE:
      memory.free();
      memory = currentMemory = arrayToMemory(array);
      syncState              = SyncState::device;
      return CEED_ERROR_SUCCESS;
  }
  return ceedError("Invalid CeedMemType passed");
}

int Vector::useArrayPointer(CeedMemType mtype, CeedScalar *array) {
  switch (mtype) {
    case CEED_MEM_HOST:
      freeHostBuffer();
      currentHostBuffer = array;
      syncState         = SyncState::host;
      return CEED_ERROR_SUCCESS;
    case CEED_MEM_DEVICE:
      memory.free();
      currentMemory = arrayToMemory(array);
      syncState     = SyncState::device;
      return CEED_ERROR_SUCCESS;
  }
  return ceedError("Invalid CeedMemType passed");
}

int Vector::getArray(CeedMemType mtype, CeedScalar **array) {
  // The passed `array` might be modified before restoring
  // so we can't set sync state to SyncState::all
  switch (mtype) {
    case CEED_MEM_HOST:
      setCurrentHostBufferIfNeeded();
      if (syncState == SyncState::device) {
        setCurrentMemoryIfNeeded();
        currentMemory.copyTo(currentHostBuffer);
      }
      syncState = SyncState::host;
      *array    = currentHostBuffer;
      return CEED_ERROR_SUCCESS;
    case CEED_MEM_DEVICE:
      setCurrentMemoryIfNeeded();
      if (syncState == SyncState::host) {
        setCurrentHostBufferIfNeeded();
        currentMemory.copyFrom(currentHostBuffer);
      }
      syncState = SyncState::device;
      *array    = memoryToArray<CeedScalar>(currentMemory);
      return CEED_ERROR_SUCCESS;
  }
  return ceedError("Invalid CeedMemType passed");
}

int Vector::getReadOnlyArray(CeedMemType mtype, CeedScalar **array) {
  const bool willBeFullySynced =
      ((syncState == SyncState::device && mtype == CEED_MEM_HOST) || (syncState == SyncState::host && mtype == CEED_MEM_DEVICE));

  const int error = getArray(mtype, const_cast<CeedScalar **>(array));
  // Take advantage the vector will be fully synced
  if (!error && willBeFullySynced) {
    syncState = SyncState::all;
  }

  return error;
}

int Vector::getWriteOnlyArray(CeedMemType mtype, CeedScalar **array) {
  // const bool willBeFullySynced = (
  //   (syncState == SyncState::device && mtype == CEED_MEM_HOST) ||
  //   (syncState == SyncState::host && mtype == CEED_MEM_DEVICE)
  // );

  const int error = getArray(mtype, const_cast<CeedScalar **>(array));
  // // Take advantage the vector will be fully synced
  // if (!error && willBeFullySynced) {
  //   syncState = SyncState::all;
  // }

  return error;
}

int Vector::restoreArray(CeedScalar **array) { return CEED_ERROR_SUCCESS; }

int Vector::restoreReadOnlyArray(CeedScalar **array) { return CEED_ERROR_SUCCESS; }

::occa::memory Vector::getKernelArg() {
  setCurrentMemoryIfNeeded();
  if (syncState == SyncState::host) {
    setCurrentHostBufferIfNeeded();
    currentMemory.copyFrom(currentHostBuffer);
  }
  syncState = SyncState::device;
  return currentMemory;
}

::occa::memory Vector::getConstKernelArg() {
  setCurrentMemoryIfNeeded();
  if (syncState == SyncState::host) {
    setCurrentHostBufferIfNeeded();
    currentMemory.copyFrom(currentHostBuffer);
    syncState = SyncState::all;
  }
  return currentMemory;
}

void Vector::printValues(const std::string &name) {
  CeedScalar *values;
  getReadOnlyArray(CEED_MEM_HOST, &values);

  std::cout << std::setprecision(8) << "Vector: " << name << std::endl << "  - Values: " << std::endl;

  for (int i = 0; i < length; ++i) {
    printf("    %12.8f\n", values[i]);
  }
}

void Vector::printNonZeroValues(const std::string &name) {
  CeedScalar *values;
  getReadOnlyArray(CEED_MEM_HOST, &values);

  std::cout << std::setprecision(8) << "Vector: " << name << std::endl << "  - Non-zero values: " << std::endl;

  for (int i = 0; i < length; ++i) {
    if (fabs(values[i]) > 1e-8) {
      printf("    %d: %12.8f\n", i, values[i]);
    }
  }
}

void Vector::printSummary(const std::string &name) {
  CeedScalar *values;
  getReadOnlyArray(CEED_MEM_HOST, &values);

  CeedScalar minValue = values[0];
  CeedScalar maxValue = values[0];

  for (int i = 0; i < length; ++i) {
    const CeedScalar value = values[i];
    minValue               = minValue < value ? minValue : value;
    maxValue               = maxValue > value ? maxValue : value;
  }

  std::cout << std::setprecision(8) << "Vector: " << name << std::endl
            << "  - Length: " << length << std::endl
            << "  - Min   : " << minValue << std::endl
            << "  - Max   : " << maxValue << std::endl;
}

//---[ Ceed Callbacks ]-----------
int Vector::registerCeedFunction(Ceed ceed, CeedVector vec, const char *fname, ceed::occa::ceedFunction f) {
  return CeedSetBackendFunction(ceed, "Vector", vec, fname, f);
}

int Vector::ceedCreate(CeedSize length, CeedVector vec) {
  Ceed ceed;
  CeedCallBackend(CeedVectorGetCeed(vec, &ceed));

  CeedOccaRegisterFunction(vec, "HasValidArray", Vector::ceedHasValidArray);
  CeedOccaRegisterFunction(vec, "HasBorrowedArrayOfType", Vector::ceedHasBorrowedArrayOfType);
  CeedOccaRegisterFunction(vec, "SetValue", Vector::ceedSetValue);
  CeedOccaRegisterFunction(vec, "SetArray", Vector::ceedSetArray);
  CeedOccaRegisterFunction(vec, "TakeArray", Vector::ceedTakeArray);
  CeedOccaRegisterFunction(vec, "GetArray", Vector::ceedGetArray);
  CeedOccaRegisterFunction(vec, "GetArrayRead", Vector::ceedGetArrayRead);
  CeedOccaRegisterFunction(vec, "GetArrayWrite", Vector::ceedGetArrayWrite);
  CeedOccaRegisterFunction(vec, "RestoreArray", Vector::ceedRestoreArray);
  CeedOccaRegisterFunction(vec, "RestoreArrayRead", Vector::ceedRestoreArrayRead);
  CeedOccaRegisterFunction(vec, "Destroy", Vector::ceedDestroy);

  Vector *vector = new Vector();
  CeedCallBackend(CeedVectorSetData(vec, vector));

  return CEED_ERROR_SUCCESS;
}

int Vector::ceedHasValidArray(CeedVector vec, bool *has_valid_array) {
  Vector *vector = Vector::from(vec);
  if (!vector) {
    return staticCeedError("Invalid CeedVector passed");
  }
  return vector->hasValidArray(has_valid_array);
}

int Vector::ceedHasBorrowedArrayOfType(CeedVector vec, CeedMemType mem_type, bool *has_borrowed_array_of_type) {
  Vector *vector = Vector::from(vec);
  if (!vector) {
    return staticCeedError("Invalid CeedVector passed");
  }
  return vector->hasBorrowedArrayOfType(mem_type, has_borrowed_array_of_type);
}

int Vector::ceedSetValue(CeedVector vec, CeedScalar value) {
  Vector *vector = Vector::from(vec);
  if (!vector) {
    return staticCeedError("Invalid CeedVector passed");
  }
  return vector->setValue(value);
}

int Vector::ceedSetArray(CeedVector vec, CeedMemType mtype, CeedCopyMode cmode, CeedScalar *array) {
  Vector *vector = Vector::from(vec);
  if (!vector) {
    return staticCeedError("Invalid CeedVector passed");
  }
  return vector->setArray(mtype, cmode, array);
}

int Vector::ceedTakeArray(CeedVector vec, CeedMemType mtype, CeedScalar **array) {
  Vector *vector = Vector::from(vec);
  if (!vector) {
    return staticCeedError("Invalid CeedVector passed");
  }
  return vector->takeArray(mtype, array);
}

int Vector::ceedGetArray(CeedVector vec, CeedMemType mtype, CeedScalar **array) {
  Vector *vector = Vector::from(vec);
  if (!vector) {
    return staticCeedError("Invalid CeedVector passed");
  }
  return vector->getArray(mtype, array);
}

int Vector::ceedGetArrayRead(CeedVector vec, CeedMemType mtype, CeedScalar **array) {
  Vector *vector = Vector::from(vec);
  if (!vector) {
    return staticCeedError("Invalid CeedVector passed");
  }
  return vector->getReadOnlyArray(mtype, array);
}

int Vector::ceedGetArrayWrite(CeedVector vec, CeedMemType mtype, CeedScalar **array) {
  Vector *vector = Vector::from(vec);
  if (!vector) {
    return staticCeedError("Invalid CeedVector passed");
  }
  return vector->getWriteOnlyArray(mtype, array);
}

int Vector::ceedRestoreArray(CeedVector vec, CeedScalar **array) {
  Vector *vector = Vector::from(vec);
  if (!vector) {
    return staticCeedError("Invalid CeedVector passed");
  }
  return vector->restoreArray(array);
}

int Vector::ceedRestoreArrayRead(CeedVector vec, CeedScalar **array) {
  Vector *vector = Vector::from(vec);
  if (!vector) {
    return staticCeedError("Invalid CeedVector passed");
  }
  return vector->restoreReadOnlyArray(array);
}

int Vector::ceedDestroy(CeedVector vec) {
  delete getVector(vec, false);
  return CEED_ERROR_SUCCESS;
}
}  // namespace occa
}  // namespace ceed
