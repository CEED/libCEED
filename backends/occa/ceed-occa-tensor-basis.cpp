// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-occa-tensor-basis.hpp"

#include "ceed-occa-kernels.hpp"

namespace ceed {
namespace occa {
TensorBasis::TensorBasis(CeedBasis basis, CeedInt dim_, CeedInt P1D_, CeedInt Q1D_, const CeedScalar *interp1D_, const CeedScalar *grad1D_,
                         const CeedScalar *qWeight1D_)
    : P1D(P1D_), Q1D(Q1D_) {
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

  setKernelProperties();
}

TensorBasis::~TensorBasis() {}

bool TensorBasis::isTensorBasis() const { return true; }

void TensorBasis::setKernelProperties() {
  kernelProperties["defines/CeedInt"]               = ::occa::dtype::get<CeedInt>().name();
  kernelProperties["defines/CeedScalar"]            = ::occa::dtype::get<CeedScalar>().name();
  kernelProperties["defines/Q1D"]                   = Q1D;
  kernelProperties["defines/P1D"]                   = P1D;
  kernelProperties["defines/BASIS_COMPONENT_COUNT"] = ceedComponentCount;
  if (usingGpuDevice()) {
    kernelProperties["defines/MAX_PQ"] = (Q1D > P1D) ? Q1D : P1D;
  }
}

const char *TensorBasis::getFunctionSource() const {
  // TODO: Add gpu function sources when split
  const char *cpuFunctionSources[3] = {occa_tensor_basis_1d_cpu_function_source, occa_tensor_basis_2d_cpu_function_source,
                                       occa_tensor_basis_3d_cpu_function_source};
  return cpuFunctionSources[dim - 1];
}

std::string TensorBasis::getKernelSource() const {
  const char *cpuFunctionSources[3] = {occa_tensor_basis_1d_cpu_function_source, occa_tensor_basis_2d_cpu_function_source,
                                       occa_tensor_basis_3d_cpu_function_source};
  const char *cpuKernelSources[3]   = {occa_tensor_basis_1d_cpu_kernel_source, occa_tensor_basis_2d_cpu_kernel_source,
                                       occa_tensor_basis_3d_cpu_kernel_source};
  const char *gpuKernelSources[3]   = {occa_tensor_basis_1d_gpu_source, occa_tensor_basis_2d_gpu_source, occa_tensor_basis_3d_gpu_source};

  std::string kernelSource;
  if (usingGpuDevice()) {
    kernelSource = gpuKernelSources[dim - 1];
  } else {
    kernelSource = cpuFunctionSources[dim - 1];
    kernelSource += '\n';
    kernelSource += cpuKernelSources[dim - 1];
  }
  return kernelSource;
}

::occa::kernel TensorBasis::buildKernel(const std::string &kernelName) {
  std::string kernelSource = getKernelSource();
  return getDevice().buildKernelFromString(kernelSource, kernelName, kernelProperties);
}

int TensorBasis::applyInterp(const CeedInt elementCount, const bool transpose, Vector &U, Vector &V) {
  if (transpose) {
    if (!interpTKernel.isInitialized()) {
      kernelProperties["defines/TRANSPOSE"]          = transpose;
      kernelProperties["defines/ELEMENTS_PER_BLOCK"] = elementsPerBlockInterp();
      interpTKernel                                  = buildKernel("interp");
    }
    interpTKernel(elementCount, interp1D, U.getConstKernelArg(), V.getKernelArg());
  } else {
    if (!interpKernel.isInitialized()) {
      kernelProperties["defines/TRANSPOSE"]          = transpose;
      kernelProperties["defines/ELEMENTS_PER_BLOCK"] = elementsPerBlockInterp();
      interpKernel                                   = buildKernel("interp");
    }
    interpKernel(elementCount, interp1D, U.getConstKernelArg(), V.getKernelArg());
  }
  return CEED_ERROR_SUCCESS;
}

int TensorBasis::elementsPerBlockInterp() const {
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
  return elementsPerBlock;
}

int TensorBasis::applyGrad(const CeedInt elementCount, const bool transpose, Vector &U, Vector &V) {
  if (transpose) {
    if (!gradTKernel.isInitialized()) {
      kernelProperties["defines/TRANSPOSE"]          = transpose;
      kernelProperties["defines/ELEMENTS_PER_BLOCK"] = elementsPerBlockGrad();
      gradTKernel                                    = buildKernel("grad");
    }
    gradTKernel(elementCount, interp1D, grad1D, U.getConstKernelArg(), V.getKernelArg());
  } else {
    if (!gradKernel.isInitialized()) {
      kernelProperties["defines/TRANSPOSE"]          = transpose;
      kernelProperties["defines/ELEMENTS_PER_BLOCK"] = elementsPerBlockGrad();
      gradKernel                                     = buildKernel("grad");
    }
    gradKernel(elementCount, interp1D, grad1D, U.getConstKernelArg(), V.getKernelArg());
  }
  return CEED_ERROR_SUCCESS;
}

int TensorBasis::elementsPerBlockGrad() const {
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
  return elementsPerBlock;
}

int TensorBasis::applyWeight(const CeedInt elementCount, Vector &W) {
  if (!weightKernel.isInitialized()) {
    kernelProperties["defines/ELEMENTS_PER_BLOCK"] = elementsPerBlockWeight();
    weightKernel                                   = buildKernel("weight");
  }
  weightKernel(elementCount, qWeight1D, W.getKernelArg());

  return CEED_ERROR_SUCCESS;
}

int TensorBasis::elementsPerBlockWeight() const {
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
  return elementsPerBlock;
}

int TensorBasis::apply(const CeedInt elementCount, CeedTransposeMode tmode, CeedEvalMode emode, Vector *U, Vector *V) {
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
int TensorBasis::ceedCreate(CeedInt dim, CeedInt P1D, CeedInt Q1D, const CeedScalar *interp1D, const CeedScalar *grad1D, const CeedScalar *qref1D,
                            const CeedScalar *qWeight1D, CeedBasis basis) {
  Ceed ceed;
  CeedCallBackend(CeedBasisGetCeed(basis, &ceed));

  if (Q1D < P1D && Context::from(ceed)->usingGpuDevice()) {
    return staticCeedError("(OCCA) Backend does not implement underintegrated basis");
  }

  TensorBasis *basis_ = new TensorBasis(basis, dim, P1D, Q1D, interp1D, grad1D, qWeight1D);
  CeedCallBackend(CeedBasisSetData(basis, basis_));

  CeedOccaRegisterFunction(basis, "Apply", Basis::ceedApply);
  CeedOccaRegisterFunction(basis, "Destroy", Basis::ceedDestroy);

  return CEED_ERROR_SUCCESS;
}
}  // namespace occa
}  // namespace ceed
