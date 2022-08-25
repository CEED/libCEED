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

#include "ceed-occa-basis.hpp"
#include "ceed-occa-elem-restriction.hpp"
#include "ceed-occa-operator.hpp"
#include "ceed-occa-cpu-operator.hpp"
#include "ceed-occa-gpu-operator.hpp"
#include "ceed-occa-qfunction.hpp"

namespace ceed {
  namespace occa {
    Operator::Operator() :
        ceedQ(0),
        ceedElementCount(0),
        qfunction(NULL),
        needsInitialSetup(true) {}

    Operator::~Operator() {}

    Operator* Operator::getOperator(CeedOperator op,
                                    const bool assertValid) {
      if (!op) {
        return NULL;
      }

      int ierr;
      Operator *operator_ = NULL;

      ierr = CeedOperatorGetData(op, (void**) &operator_);
      if (assertValid) {
        CeedOccaFromChk(ierr);
      }

      return operator_;
    }

    Operator* Operator::from(CeedOperator op) {
      Operator *operator_ = getOperator(op);
      if (!operator_) {
        return NULL;
      }

      int ierr;
      ierr = CeedOperatorGetCeed(op, &operator_->ceed); CeedOccaFromChk(ierr);

      operator_->qfunction = QFunction::from(op);
      if (!operator_->qfunction) {
        return NULL;
      }

      ierr = CeedOperatorGetNumQuadraturePoints(op, &operator_->ceedQ); CeedOccaFromChk(ierr);
      ierr = CeedOperatorGetNumElements(op, &operator_->ceedElementCount); CeedOccaFromChk(ierr);

      operator_->args.setupArgs(op);
      if (!operator_->args.isValid()) {
        return NULL;
      }

      return operator_;
    }

    bool Operator::isApplyingIdentityFunction() {
      return qfunction->ceedIsIdentity;
    }

    int Operator::applyAdd(Vector *in, Vector *out, CeedRequest *request) {
      // TODO: Cache kernel objects rather than relying on OCCA kernel caching
      applyAddKernel = buildApplyAddKernel();

      if (needsInitialSetup) {
        initialSetup();
        needsInitialSetup = false;
      }

      applyAdd(in, out);

      return CEED_ERROR_SUCCESS;
    }

    //---[ Virtual Methods ]------------
    void Operator::initialSetup() {}

    //---[ Ceed Callbacks ]-------------
    int Operator::registerCeedFunction(Ceed ceed, CeedOperator op,
                                       const char *fname, ceed::occa::ceedFunction f) {
      return CeedSetBackendFunction(ceed, "Operator", op, fname, f);
    }

    int Operator::ceedCreate(CeedOperator op) {
      int ierr;
      Ceed ceed;
      ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);

#if 1
      Operator *operator_ = new CpuOperator();
#else
      // TODO: Add GPU specific operator
      Operator *operator_ = (
        Context::from(ceed)->usingCpuDevice()
        ? ((Operator*) new CpuOperator())
        : ((Operator*) new GpuOperator())
      );
#endif

      ierr = CeedOperatorSetData(op, operator_); CeedChk(ierr);

      CeedOccaRegisterFunction(op, "LinearAssembleQFunction", Operator::ceedLinearAssembleQFunction);
      CeedOccaRegisterFunction(op, "LinearAssembleQFunctionUpdate", Operator::ceedLinearAssembleQFunction);
      CeedOccaRegisterFunction(op, "LinearAssembleAddDiagonal", Operator::ceedLinearAssembleAddDiagonal);
      CeedOccaRegisterFunction(op, "LinearAssembleAddPointBlockDiagonal", Operator::ceedLinearAssembleAddPointBlockDiagonal);
      CeedOccaRegisterFunction(op, "CreateFDMElementInverse", Operator::ceedCreateFDMElementInverse);
      CeedOccaRegisterFunction(op, "ApplyAdd", Operator::ceedApplyAdd);
      CeedOccaRegisterFunction(op, "Destroy", Operator::ceedDestroy);

      return CEED_ERROR_SUCCESS;
    }

    int Operator::ceedCreateComposite(CeedOperator op) {
      int ierr;
      Ceed ceed;
      ierr = CeedOperatorGetCeed(op, &ceed); CeedChk(ierr);

      CeedOccaRegisterFunction(op, "LinearAssembleAddDiagonal", Operator::ceedLinearAssembleAddDiagonal);
      CeedOccaRegisterFunction(op, "LinearAssembleAddPointBlockDiagonal", Operator::ceedLinearAssembleAddPointBlockDiagonal);

      return CEED_ERROR_SUCCESS;
    }

    int Operator::ceedLinearAssembleQFunction(CeedOperator op) {
      return staticCeedError("(OCCA) Backend does not implement LinearAssembleQFunction");
    }

    int Operator::ceedLinearAssembleQFunctionUpdate(CeedOperator op) {
      return staticCeedError("(OCCA) Backend does not implement LinearAssembleQFunctionUpdate");
    }

    int Operator::ceedLinearAssembleAddDiagonal(CeedOperator op) {
      return staticCeedError("(OCCA) Backend does not implement LinearAssembleDiagonal");
    }


    int Operator::ceedLinearAssembleAddPointBlockDiagonal(CeedOperator op) {
      return staticCeedError("(OCCA) Backend does not implement LinearAssemblePointBlockDiagonal");
    }

    int Operator::ceedCreateFDMElementInverse(CeedOperator op) {
      return staticCeedError("(OCCA) Backend does not implement CreateFDMElementInverse");
    }

    int Operator::ceedApplyAdd(CeedOperator op,
                               CeedVector invec, CeedVector outvec, CeedRequest *request) {
      Operator *operator_ = Operator::from(op);
      Vector *in = Vector::from(invec);
      Vector *out = Vector::from(outvec);

      if (!operator_) {
        return staticCeedError("Incorrect CeedOperator argument: op");
      }

      return operator_->applyAdd(in, out, request);
    }

    int Operator::ceedDestroy(CeedOperator op) {
      delete getOperator(op, false);
      return CEED_ERROR_SUCCESS;
    }
  }
}
