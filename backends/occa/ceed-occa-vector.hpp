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

#ifndef CEED_OCCA_VECTOR_HEADER
#define CEED_OCCA_VECTOR_HEADER

#include "ceed-occa-ceed-object.hpp"

namespace ceed {
  namespace occa {
    template <class TM>
    ::occa::memory arrayToMemory(const TM *array) {
      if (array) {
        ::occa::memory mem((::occa::modeMemory_t*) array);
        mem.setDtype(::occa::dtype::get<TM>());
        return mem;
      }
      return ::occa::null;
    }

    template <class TM>
    TM* memoryToArray(::occa::memory &memory) {
      return (TM*) memory.getModeMemory();
    }

    class Vector : public CeedObject {
     public:
      // Owned resources
      CeedSize length;
      ::occa::memory memory;
      CeedSize hostBufferLength;
      CeedScalar *hostBuffer;

      ::occa::kernel setValueKernel;

      // Current resources
      ::occa::memory currentMemory;
      CeedScalar *currentHostBuffer;

      // State information
      int syncState;

      Vector();

      ~Vector();

      static Vector* getVector(CeedVector vec,
                               const bool assertValid = true);

      static Vector* from(CeedVector vec);

      void resize(const CeedSize length_);

      void resizeMemory(const CeedSize length_);

      void resizeMemory(::occa::device device, const CeedSize length_);

      void resizeHostBuffer(const CeedSize length_);

      void setCurrentMemoryIfNeeded();

      void setCurrentHostBufferIfNeeded();

      void freeHostBuffer();

      int setValue(CeedScalar value);

      int setArray(CeedMemType mtype,
                   CeedCopyMode cmode, CeedScalar *array);

      int takeArray(CeedMemType mtype, CeedScalar **array);

      int copyArrayValues(CeedMemType mtype, CeedScalar *array);

      int ownArrayPointer(CeedMemType mtype, CeedScalar *array);

      int useArrayPointer(CeedMemType mtype, CeedScalar *array);

      int getArray(CeedMemType mtype,
                   CeedScalar **array);

      int getReadOnlyArray(CeedMemType mtype,
                           CeedScalar **array);

      int restoreArray(CeedScalar **array);

      int restoreReadOnlyArray(CeedScalar **array);

      ::occa::memory getKernelArg();

      ::occa::memory getConstKernelArg();

      void printValues(const std::string &name);
      void printNonZeroValues(const std::string &name);
      void printSummary(const std::string &name);

      //---[ Ceed Callbacks ]-----------
      static int registerCeedFunction(Ceed ceed, CeedVector vec,
                                      const char *fname, ceed::occa::ceedFunction f);

      static int ceedCreate(CeedSize length, CeedVector vec);

      static int ceedSetValue(CeedVector vec, CeedScalar value);

      static int ceedSetArray(CeedVector vec, CeedMemType mtype,
                              CeedCopyMode cmode, CeedScalar *array);

      static int ceedTakeArray(CeedVector vec, CeedMemType mtype, CeedScalar **array);

      static int ceedGetArray(CeedVector vec, CeedMemType mtype,
                              CeedScalar **array);

      static int ceedGetArrayRead(CeedVector vec, CeedMemType mtype,
                                  CeedScalar **array);

      static int ceedRestoreArray(CeedVector vec, CeedScalar **array);

      static int ceedRestoreArrayRead(CeedVector vec, CeedScalar **array);

      static int ceedDestroy(CeedVector vec);
    };
  }
}

#endif
