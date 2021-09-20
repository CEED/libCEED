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

#include "ceed-occa-kernels.hpp"
#include "ceed-occa-tensor-basis.hpp"

namespace ceed {
  namespace occa {
    TensorBasis::TensorBasis(CeedBasis basis,
                             CeedInt dim_,
                             CeedInt P1D_,
                             CeedInt Q1D_,
                             const CeedScalar *interp1D_,
                             const CeedScalar *grad1D_,
                             const CeedScalar *qWeight1D_) :
        P1D(P1D_),
        Q1D(Q1D_),
        interpKernelBuilder(::occa::kernelBuilder(
          occa_tensor_basis_1d_gpu_source, "interp"
        )),
        gradKernelBuilder(::occa::kernelBuilder(
          occa_tensor_basis_1d_gpu_source, "grad"
        )),
        weightKernelBuilder(::occa::kernelBuilder(
          occa_tensor_basis_1d_gpu_source, "weight"
        )) {
      setCeedFields(basis);

      dim = dim_;

      P = P1D;
      Q = Q1D;
      for (int i = 1; i < dim; ++i) {
        P *= P1D;
        Q *= Q1D;
      }

      ::occa::device device = getDevice();

      interp1D  = device.malloc<CeedScalar>(P1D * Q1D, interp1D_);
      grad1D    = device.malloc<CeedScalar>(P1D * Q1D, grad1D_);
      qWeight1D = device.malloc<CeedScalar>(Q1D, qWeight1D_);

      setupKernelBuilders();
    }

    TensorBasis::~TensorBasis() {}

    bool TensorBasis::isTensorBasis() const {
      return true;
    }

    const char* TensorBasis::getFunctionSource() const {
      // TODO: Add gpu function sources when split
      const char *cpuFunctionSources[3] = {
        occa_tensor_basis_1d_cpu_function_source,
        occa_tensor_basis_2d_cpu_function_source,
        occa_tensor_basis_3d_cpu_function_source
      };
      return cpuFunctionSources[dim - 1];
    }

    void TensorBasis::setupKernelBuilders() {
      const char *cpuFunctionSources[3] = {
        occa_tensor_basis_1d_cpu_function_source,
        occa_tensor_basis_2d_cpu_function_source,
        occa_tensor_basis_3d_cpu_function_source
      };
      const char *cpuKernelSources[3] = {
        occa_tensor_basis_1d_cpu_kernel_source,
        occa_tensor_basis_2d_cpu_kernel_source,
        occa_tensor_basis_3d_cpu_kernel_source
      };
      const char *gpuKernelSources[3] = {
        occa_tensor_basis_1d_gpu_source,
        occa_tensor_basis_2d_gpu_source,
        occa_tensor_basis_3d_gpu_source
      };

      std::string kernelSource;
      if (usingGpuDevice()) {
        kernelSource = gpuKernelSources[dim - 1];
      } else {
        kernelSource = cpuFunctionSources[dim - 1];
        kernelSource += '\n';
        kernelSource += cpuKernelSources[dim - 1];
      }

      interpKernelBuilder = ::occa::kernelBuilder(
        kernelSource, "interp"
      );
      gradKernelBuilder = ::occa::kernelBuilder(
        kernelSource, "grad"
      );
      weightKernelBuilder = ::occa::kernelBuilder(
        kernelSource, "weight"
      );
    }

    int TensorBasis::applyInterp(const CeedInt elementCount,
                                 const bool transpose,
                                 Vector &U,
                                 Vector &V) {
      ::occa::kernel interp = (
        usingGpuDevice()
        ? getGpuInterpKernel(transpose)
        : getCpuInterpKernel(transpose)
      );

      interp(elementCount,
             interp1D,
             U.getConstKernelArg(),
             V.getKernelArg());

      return CEED_ERROR_SUCCESS;
    }

    ::occa::kernel TensorBasis::getCpuInterpKernel(const bool transpose) {
      return buildCpuEvalKernel(interpKernelBuilder,
                                transpose);
    }

    ::occa::kernel TensorBasis::getGpuInterpKernel(const bool transpose) {
      int elementsPerBlock;
      if (dim == 1) {
        elementsPerBlock = 32;
      } else if (dim == 2) {
        const CeedInt blocksByQ[7] = {0, 32, 8, 6, 4, 2, 8};
        if (Q1D < 7) {
          elementsPerBlock = blocksByQ[Q1D];
        } else {
          elementsPerBlock = 1;
        }
      } else {
        elementsPerBlock = 1;
      }

      return buildGpuEvalKernel(interpKernelBuilder,
                                transpose,
                                elementsPerBlock);
    }

    int TensorBasis::applyGrad(const CeedInt elementCount,
                               const bool transpose,
                               Vector &U,
                               Vector &V) {
      ::occa::kernel grad = (
        usingGpuDevice()
        ? getGpuGradKernel(transpose)
        : getCpuGradKernel(transpose)
      );

      grad(elementCount,
           interp1D, grad1D,
           U.getConstKernelArg(),
           V.getKernelArg());

      return CEED_ERROR_SUCCESS;
    }

    ::occa::kernel TensorBasis::getCpuGradKernel(const bool transpose) {
      return buildCpuEvalKernel(gradKernelBuilder,
                                transpose);
    }

    ::occa::kernel TensorBasis::getGpuGradKernel(const bool transpose) {
      int elementsPerBlock;
      if (dim == 1) {
        elementsPerBlock = 32;
      } else if (dim == 2) {
        const CeedInt blocksByQ[7] = {0, 32, 8, 6, 4, 2, 8};
        if (Q1D < 7) {
          elementsPerBlock = blocksByQ[Q1D];
        } else {
          elementsPerBlock = 1;
        }
      } else {
        elementsPerBlock = 1;
      }

      return buildGpuEvalKernel(gradKernelBuilder,
                                transpose,
                                elementsPerBlock);
    }

    int TensorBasis::applyWeight(const CeedInt elementCount,
                                 Vector &W) {
      ::occa::kernel weight = (
        usingGpuDevice()
        ? getGpuWeightKernel()
        : getCpuWeightKernel()
      );

      weight(elementCount, qWeight1D, W.getKernelArg());

      return CEED_ERROR_SUCCESS;
    }

    ::occa::kernel TensorBasis::getCpuWeightKernel() {
      return buildCpuEvalKernel(weightKernelBuilder,
                                false);
    }

    ::occa::kernel TensorBasis::getGpuWeightKernel() {
      int elementsPerBlock;
      if (dim == 1) {
        elementsPerBlock = 32 / Q1D;
      } else if (dim == 2) {
        if ((Q1D * Q1D) > 32) {
          elementsPerBlock = 1;
        } else {
          elementsPerBlock = 32 / (Q1D * Q1D);
        }
      } else {
        elementsPerBlock = Q1D;
      }

      return buildGpuEvalKernel(weightKernelBuilder,
                                false,
                                elementsPerBlock);
    }

    ::occa::kernel TensorBasis::buildCpuEvalKernel(::occa::kernelBuilder &kernelBuilder,
                                                   const bool transpose) {
      ::occa::json kernelProps;
      kernelProps["defines/CeedInt"]    = ::occa::dtype::get<CeedInt>().name();
      kernelProps["defines/CeedScalar"] = ::occa::dtype::get<CeedScalar>().name();
      kernelProps["defines/Q1D"] = Q1D;
      kernelProps["defines/P1D"] = P1D;
      kernelProps["defines/BASIS_COMPONENT_COUNT"] = ceedComponentCount;
      kernelProps["defines/TRANSPOSE"] = transpose;

      return kernelBuilder.getOrBuildKernel(::occa::scope({}, kernelProps));
    }

    ::occa::kernel TensorBasis::buildGpuEvalKernel(::occa::kernelBuilder &kernelBuilder,
                                                   const bool transpose,
                                                   const int elementsPerBlock) {
      ::occa::json kernelProps;
      kernelProps["defines/CeedInt"]    = ::occa::dtype::get<CeedInt>().name();
      kernelProps["defines/CeedScalar"] = ::occa::dtype::get<CeedScalar>().name();
      kernelProps["defines/Q1D"] = Q1D;
      kernelProps["defines/P1D"] = P1D;
      kernelProps["defines/BASIS_COMPONENT_COUNT"] = ceedComponentCount;
      kernelProps["defines/TRANSPOSE"]          = transpose;
      kernelProps["defines/MAX_PQ"]             = Q1D > P1D ? Q1D : P1D;
      kernelProps["defines/ELEMENTS_PER_BLOCK"] = elementsPerBlock;

      return kernelBuilder.getOrBuildKernel(::occa::scope({}, kernelProps));
    }

    int TensorBasis::apply(const CeedInt elementCount,
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
            return ceedError("Backend does not support given tensor eval mode");
        }
      } catch (::occa::exception &exc) {
        // Handle kernel build errors the CEED way
        CeedHandleOccaException(exc);
      }

      return CEED_ERROR_SUCCESS;
    }

    //---[ Ceed Callbacks ]-------------
    int TensorBasis::ceedCreate(CeedInt dim,
                                CeedInt P1D, CeedInt Q1D,
                                const CeedScalar *interp1D,
                                const CeedScalar *grad1D,
                                const CeedScalar *qref1D,
                                const CeedScalar *qWeight1D,
                                CeedBasis basis) {
      int ierr;
      Ceed ceed;
      ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);

      if (Q1D < P1D && Context::from(ceed)->usingGpuDevice()) {
        return staticCeedError("(OCCA) Backend does not implement underintegrated basis");
      }

      TensorBasis *basis_ = new TensorBasis(basis,
                                            dim,
                                            P1D, Q1D,
                                            interp1D, grad1D, qWeight1D);
      ierr = CeedBasisSetData(basis, basis_); CeedChk(ierr);

      CeedOccaRegisterFunction(basis, "Apply", Basis::ceedApply);
      CeedOccaRegisterFunction(basis, "Destroy", Basis::ceedDestroy);

      return CEED_ERROR_SUCCESS;
    }
  }
}
