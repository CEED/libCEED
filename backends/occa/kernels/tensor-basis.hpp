// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

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
