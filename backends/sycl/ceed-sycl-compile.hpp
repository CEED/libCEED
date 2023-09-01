// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_SYCL_COMPILE_HPP
#define CEED_SYCL_COMPILE_HPP

#include <ceed/backend.h>
#include <ceed/ceed.h>

#include <map>
#include <sycl/sycl.hpp>

using SyclModule_t = sycl::kernel_bundle<sycl::bundle_state::executable>;

CEED_INTERN int CeedBuildModule_Sycl(Ceed ceed, const std::string &kernel_source, SyclModule_t **sycl_module,
                                     const std::map<std::string, CeedInt> &constants = {});
CEED_INTERN int CeedGetKernel_Sycl(Ceed ceed, const SyclModule_t *sycl_module, const std::string &kernel_name, sycl::kernel **sycl_kernel);

CEED_INTERN int CeedRunKernelDimSharedSycl(Ceed ceed, sycl::kernel *kernel, const int grid_size, const int block_size_x, const int block_size_y,
                                           const int block_size_z, const int shared_mem_size, void **args);

#endif  // CEED_SYCL_COMPILE_HPP
