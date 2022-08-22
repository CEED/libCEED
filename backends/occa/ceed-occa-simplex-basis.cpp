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

#include "ceed-occa-kernels.hpp"
#include "ceed-occa-simplex-basis.hpp"

namespace ceed {
  namespace occa {
    SimplexBasis::SimplexBasis(CeedBasis basis,
                               CeedInt dim_,
                               CeedInt P_,
                               CeedInt Q_,
                               const CeedScalar *interp_,
                               const CeedScalar *grad_,
                               const CeedScalar *qWeight_) {
      setCeedFields(basis);

      dim = dim_;
      P = P_;
      Q = Q_;

      ::occa::device device = getDevice();

      interp  = device.malloc<CeedScalar>(P * Q, interp_);
      grad    = device.malloc<CeedScalar>(P * Q * dim, grad_);
      qWeight = device.malloc<CeedScalar>(Q, qWeight_);

      setKernelProperties();
    }

    SimplexBasis::~SimplexBasis() {}

    bool SimplexBasis::isTensorBasis() const {
      return false;
    }

    const char* SimplexBasis::getFunctionSource() const {
      // TODO: Add gpu function sources when split
      return occa_simplex_basis_cpu_function_source;
    }

    void SimplexBasis::setKernelProperties() {
      kernelProperties["defines/CeedInt"]    = ::occa::dtype::get<CeedInt>().name();
      kernelProperties["defines/CeedScalar"] = ::occa::dtype::get<CeedScalar>().name();
      kernelProperties["defines/DIM"]        = dim;
      kernelProperties["defines/Q"]          = Q;
      kernelProperties["defines/P"]          = P;
      kernelProperties["defines/MAX_PQ"]     = P > Q ? P : Q;
      kernelProperties["defines/BASIS_COMPONENT_COUNT"] = ceedComponentCount;
      if(usingGpuDevice()) {
        kernelProperties["defines/ELEMENTS_PER_BLOCK"] = (Q <= 1024) ? (1024 / Q) : 1;
      }
    }

    ::occa::kernel SimplexBasis::buildKernel(const std::string& kernelName) {
      std::string kernelSource;
      if (usingGpuDevice()) {
        kernelSource = occa_simplex_basis_gpu_source;
      } else {
        kernelSource = occa_simplex_basis_cpu_function_source;
        kernelSource += '\n';
        kernelSource += occa_simplex_basis_cpu_kernel_source;
      }

      return getDevice().buildKernelFromString(kernelSource,
                                               kernelName,
                                               kernelProperties);
    }

    int SimplexBasis::applyInterp(const CeedInt elementCount,
                                  const bool transpose,
                                  Vector &U,
                                  Vector &V) {
      if(transpose) {
        if(!interpTKernel.isInitialized()) {
          kernelProperties["defines/TRANSPOSE"] = transpose;
          interpTKernel = buildKernel("interp");
        }
        
        interpTKernel(elementCount,
                    interp,
                    U.getConstKernelArg(),
                    V.getKernelArg());
      } else {
        if(!interpKernel.isInitialized()) {
          kernelProperties["defines/TRANSPOSE"] = transpose;
          interpKernel = buildKernel("interp");
        }
        
        interpKernel(elementCount,
                    interp,
                    U.getConstKernelArg(),
                    V.getKernelArg());
      }
      return CEED_ERROR_SUCCESS;
    }

    int SimplexBasis::applyGrad(const CeedInt elementCount,
                                const bool transpose,
                                Vector &U,
                                Vector &V) {
      if(transpose) {
        if(!gradTKernel.isInitialized()) {
          kernelProperties["defines/TRANSPOSE"] = transpose;
          gradTKernel = buildKernel("grad");
        }

        gradTKernel(elementCount,
                  grad,
                  U.getConstKernelArg(),
                  V.getKernelArg());
      } else {
        if(!gradKernel.isInitialized()) {
          kernelProperties["defines/TRANSPOSE"] = transpose;
          gradKernel = buildKernel("grad");
        }

        gradKernel(elementCount,
                   grad,
                   U.getConstKernelArg(),
                   V.getKernelArg());
      }
      return CEED_ERROR_SUCCESS;
    }

    int SimplexBasis::applyWeight(const CeedInt elementCount,
                                  Vector &W) {
      if(!weightKernel.isInitialized()) {
        weightKernel = buildKernel("weight");
      }
      weightKernel(elementCount, qWeight, W.getKernelArg());

      return CEED_ERROR_SUCCESS;
    }

    int SimplexBasis::apply(const CeedInt elementCount,
                            CeedTransposeMode tmode,
                            CeedEvalMode emode,
                            Vector *U,
                            Vector *V) {
      const bool transpose = tmode == CEED_TRANSPOSE;

      if ((dim < 1) || (3 < dim)) {
        return ceedError("Backend only supports dimensions: 1, 2, and 3");
      }

      // Check arguments
      if (emode != CEED_EVAL_WEIGHT) {
        if (!U) {
          return ceedError("Incorrect CeedVector input: U");
        }
      }
      if (!V) {
        return ceedError("Incorrect CeedVector input: V");
      }

      try {
        // Apply kernel
        switch (emode) {
          case CEED_EVAL_INTERP:
            return applyInterp(elementCount, transpose, *U, *V);
          case CEED_EVAL_GRAD:
            return applyGrad(elementCount, transpose, *U, *V);
          case CEED_EVAL_WEIGHT:
            return applyWeight(elementCount, *V);
          default:
            return ceedError("Backend does not support given simplex eval mode");
        }
      } catch (::occa::exception &exc) {
        // Handle kernel build errors the CEED way
        CeedHandleOccaException(exc);
      }

      return CEED_ERROR_SUCCESS;
    }

    //---[ Ceed Callbacks ]-------------
    int SimplexBasis::ceedCreate(CeedElemTopology topology,
                                 CeedInt dim,
                                 CeedInt ndof,
                                 CeedInt nquad,
                                 const CeedScalar *interp,
                                 const CeedScalar *grad,
                                 const CeedScalar *qref,
                                 const CeedScalar *qWeight,
                                 CeedBasis basis) {
      int ierr;
      Ceed ceed;
      ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);

      SimplexBasis *basis_ = new SimplexBasis(basis,
                                              dim,
                                              ndof, nquad,
                                              interp, grad, qWeight);
      ierr = CeedBasisSetData(basis, basis_); CeedChk(ierr);

      CeedOccaRegisterFunction(basis, "Apply", Basis::ceedApply);
      CeedOccaRegisterFunction(basis, "Destroy", Basis::ceedDestroy);

      return CEED_ERROR_SUCCESS;
    }
  }
}
