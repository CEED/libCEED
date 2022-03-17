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

#ifndef CEED_OCCA_KERNELS_TENSORBASIS_HEADER
#define CEED_OCCA_KERNELS_TENSORBASIS_HEADER

// Kernels are based on the cuda backend from LLNL and VT groups
//
// Expects the following types to be defined:
// - CeedInt
// - CeedScalar
//
// Expects the following constants to be defined:
// - Q1D                  : CeedInt
// - P1D                  : CeedInt
// - BASIS_COMPONENT_COUNT: CeedInt
// - ELEMENTS_PER_BLOCK   : CeedInt
// - SHARED_BUFFER_SIZE   : CeedInt
// - TRANSPOSE            : bool

extern const char *occa_tensor_basis_1d_cpu_function_source;
extern const char *occa_tensor_basis_1d_cpu_kernel_source;

extern const char *occa_tensor_basis_2d_cpu_function_source;
extern const char *occa_tensor_basis_2d_cpu_kernel_source;

extern const char *occa_tensor_basis_3d_cpu_function_source;
extern const char *occa_tensor_basis_3d_cpu_kernel_source;

extern const char *occa_tensor_basis_1d_gpu_source;
extern const char *occa_tensor_basis_2d_gpu_source;
extern const char *occa_tensor_basis_3d_gpu_source;

#endif
