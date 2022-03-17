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

#ifndef CEED_OCCA_OPERATOR_HEADER
#define CEED_OCCA_OPERATOR_HEADER

#include <vector>

#include "ceed-occa-ceed-object.hpp"
#include "ceed-occa-operator-args.hpp"

namespace ceed {
  namespace occa {
    typedef std::vector<ceed::occa::Vector*> VectorVector_t;

    class QFunction;

    class Operator : public CeedObject {
     public:
      // Ceed object information
      CeedInt ceedQ;
      CeedInt ceedElementCount;

      // Owned resources
      QFunction *qfunction;
      OperatorArgs args;
      ::occa::kernel applyAddKernel;
      bool needsInitialSetup;

      // Reference to other memory
      ::occa::memory qFunctionContextData;

      Operator();
      virtual ~Operator();

      static Operator* getOperator(CeedOperator op,
                                   const bool assertValid = true);

      static Operator* from(CeedOperator op);

      bool isApplyingIdentityFunction();

      int applyAdd(Vector *in, Vector *out, CeedRequest *request);

      //---[ Virtual Methods ]----------
      virtual ::occa::kernel buildApplyAddKernel() = 0;

      virtual void initialSetup();

      virtual void applyAdd(Vector *in, Vector *out) = 0;

      //---[ Ceed Callbacks ]-----------
      static int registerCeedFunction(Ceed ceed, CeedOperator op,
                                      const char *fname, ceed::occa::ceedFunction f);

      static int ceedCreate(CeedOperator op);
      static int ceedCreateComposite(CeedOperator op);

      static int ceedLinearAssembleQFunction(CeedOperator op);
      static int ceedLinearAssembleAddDiagonal(CeedOperator op);
      static int ceedLinearAssembleAddPointBlockDiagonal(CeedOperator op);
      static int ceedCreateFDMElementInverse(CeedOperator op);

      static int ceedApplyAdd(CeedOperator op,
                              CeedVector invec, CeedVector outvec, CeedRequest *request);

      static int ceedDestroy(CeedOperator op);
    };
  }
}

#endif
