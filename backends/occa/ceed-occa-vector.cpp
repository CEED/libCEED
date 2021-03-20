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

#include "ceed-occa-vector.hpp"


namespace ceed {
  namespace occa {
    Vector::Vector() :
        length(0),
        hostBufferLength(0),
        hostBuffer(NULL),
        currentHostBuffer(NULL),
        syncState(SyncState::none) {}

    Vector::~Vector() {
      memory.free();
      freeHostBuffer();
    }

    Vector* Vector::from(CeedVector vec) {
      if (!vec || vec == CEED_VECTOR_NONE) {
        return NULL;
      }

      int ierr;

      Vector *vector = NULL;
      ierr = CeedVectorGetData(vec, &vector); CeedOccaFromChk(ierr);

      if (vector != NULL) {
        ierr = CeedVectorGetCeed(vec, &vector->ceed); CeedOccaFromChk(ierr);
        ierr = CeedVectorGetLength(vec, &vector->length); CeedOccaFromChk(ierr);
      }

      return vector;
    }

    void Vector::resize(const CeedInt length_) {
      length = length_;
    }

    void Vector::resizeMemory(const CeedInt length_) {
      resizeMemory(getDevice(), length_);
    }

    void Vector::resizeMemory(::occa::device device, const CeedInt length_) {
      if (length_ != (CeedInt) memory.length()) {
        memory.free();
        memory = device.malloc<CeedScalar>(length_);
      }
    }

    void Vector::resizeHostBuffer(const CeedInt length_) {
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
        delete [] hostBuffer;
        hostBuffer = NULL;
      }
    }

    int Vector::setValue(CeedScalar value) {
      // Prioritize keeping data in the device
      if (syncState & SyncState::device) {
        setCurrentMemoryIfNeeded();
        ::occa::linalg::operator_eq(currentMemory, value);
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

    int Vector::setArray(CeedMemType mtype,
                         CeedCopyMode cmode, CeedScalar *array) {
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
          *array = currentHostBuffer;
          hostBuffer = NULL;
          currentHostBuffer = NULL;

          syncState = SyncState::host;
          return CEED_ERROR_SUCCESS;
        case CEED_MEM_DEVICE:
          setCurrentMemoryIfNeeded();
          if (syncState == SyncState::host) {
            setCurrentHostBufferIfNeeded();
            currentMemory.copyFrom(currentHostBuffer);
          }
          *array = memoryToArray<CeedScalar>(currentMemory);
          memory = ::occa::null;
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
            ::memcpy(currentHostBuffer, array, length * sizeof(CeedScalar));
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
          syncState = SyncState::host;
          return CEED_ERROR_SUCCESS;
        case CEED_MEM_DEVICE:
          memory.free();
          memory = currentMemory = arrayToMemory(array);
          syncState = SyncState::device;
          return CEED_ERROR_SUCCESS;
      }
      return ceedError("Invalid CeedMemType passed");
    }

    int Vector::useArrayPointer(CeedMemType mtype, CeedScalar *array) {
      switch (mtype) {
        case CEED_MEM_HOST:
          freeHostBuffer();
          currentHostBuffer = array;
          syncState = SyncState::host;
          return CEED_ERROR_SUCCESS;
        case CEED_MEM_DEVICE:
          memory.free();
          currentMemory = arrayToMemory(array);
          syncState = SyncState::device;
          return CEED_ERROR_SUCCESS;
      }
      return ceedError("Invalid CeedMemType passed");
    }

    int Vector::getArray(CeedMemType mtype,
                         CeedScalar **array) {
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
          *array = currentHostBuffer;
          return CEED_ERROR_SUCCESS;
        case CEED_MEM_DEVICE:
          setCurrentMemoryIfNeeded();
          if (syncState == SyncState::host) {
            setCurrentHostBufferIfNeeded();
            currentMemory.copyFrom(currentHostBuffer);
          }
          syncState = SyncState::device;
          *array = memoryToArray<CeedScalar>(currentMemory);
          return CEED_ERROR_SUCCESS;
      }
      return ceedError("Invalid CeedMemType passed");
    }

    int Vector::getReadOnlyArray(CeedMemType mtype,
                                 CeedScalar **array) {
      const bool willBeFullySynced = (
        (syncState == SyncState::device && mtype == CEED_MEM_HOST) ||
        (syncState == SyncState::host && mtype == CEED_MEM_DEVICE)
      );

      const int error = getArray(mtype, const_cast<CeedScalar**>(array));
      // Take advantage the vector will be fully synced
      if (!error && willBeFullySynced) {
        syncState = SyncState::all;
      }

      return error;
    }

    int Vector::restoreArray(CeedScalar **array) {
      return CEED_ERROR_SUCCESS;
    }

    int Vector::restoreReadOnlyArray(CeedScalar **array) {
      return CEED_ERROR_SUCCESS;
    }

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

      std::cout << std::setprecision(8)
                << "Vector: " << name << std::endl
                << "  - Values: " << std::endl;

      for (int i = 0; i < length; ++i) {
        printf("    %12.8f\n", values[i]);
      }
    }

    void Vector::printNonZeroValues(const std::string &name) {
      CeedScalar *values;
      getReadOnlyArray(CEED_MEM_HOST, &values);

      std::cout << std::setprecision(8)
                << "Vector: " << name << std::endl
                << "  - Non-zero values: " << std::endl;

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
        minValue = minValue < value ? minValue : value;
        maxValue = maxValue > value ? maxValue : value;
      }

      std::cout << std::setprecision(8)
                << "Vector: " << name << std::endl
                << "  - Length: " << length << std::endl
                << "  - Min   : " << minValue << std::endl
                << "  - Max   : " << maxValue << std::endl;
    }

    //---[ Ceed Callbacks ]-----------
    int Vector::registerCeedFunction(Ceed ceed, CeedVector vec,
                                     const char *fname, ceed::occa::ceedFunction f) {
      return CeedSetBackendFunction(ceed, "Vector", vec, fname, f);
    }

    int Vector::ceedCreate(CeedInt length, CeedVector vec) {
      int ierr;

      Ceed ceed;
      ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);

      CeedOccaRegisterFunction(vec, "SetValue", Vector::ceedSetValue);
      CeedOccaRegisterFunction(vec, "SetArray", Vector::ceedSetArray);
      CeedOccaRegisterFunction(vec, "TakeArray", Vector::ceedTakeArray);
      CeedOccaRegisterFunction(vec, "GetArray", Vector::ceedGetArray);
      CeedOccaRegisterFunction(vec, "GetArrayRead", Vector::ceedGetArrayRead);
      CeedOccaRegisterFunction(vec, "RestoreArray", Vector::ceedRestoreArray);
      CeedOccaRegisterFunction(vec, "RestoreArrayRead", Vector::ceedRestoreArrayRead);
      CeedOccaRegisterFunction(vec, "Destroy", Vector::ceedDestroy);

      Vector *vector = new Vector();
      ierr = CeedVectorSetData(vec, vector); CeedChk(ierr);

      return CEED_ERROR_SUCCESS;
    }

    int Vector::ceedSetValue(CeedVector vec, CeedScalar value) {
      Vector *vector = Vector::from(vec);
      if (!vector) {
        return staticCeedError("Invalid CeedVector passed");
      }
      return vector->setValue(value);
    }

    int Vector::ceedSetArray(CeedVector vec, CeedMemType mtype,
                             CeedCopyMode cmode, CeedScalar *array) {
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

    int Vector::ceedGetArray(CeedVector vec, CeedMemType mtype,
                             CeedScalar **array) {
      Vector *vector = Vector::from(vec);
      if (!vector) {
        return staticCeedError("Invalid CeedVector passed");
      }
      return vector->getArray(mtype, array);
    }

    int Vector::ceedGetArrayRead(CeedVector vec, CeedMemType mtype,
                                 CeedScalar **array) {
      Vector *vector = Vector::from(vec);
      if (!vector) {
        return staticCeedError("Invalid CeedVector passed");
      }
      return vector->getReadOnlyArray(mtype, array);
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
      delete Vector::from(vec);
      return CEED_ERROR_SUCCESS;
    }
  }
}
