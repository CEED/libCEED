// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef _ceed_sycl_compile_hpp
#define _ceed_sycl_compile_hpp

#include <ceed/backend.h>
#include <ceed/ceed.h>

#include <map>
#include <sycl/sycl.hpp>

using SyclModule_t = sycl::kernel_bundle<sycl::bundle_state::executable>;

CEED_INTERN int CeedBuildModule_Sycl(Ceed ceed, const std::string &kernel_source, SyclModule_t **sycl_module,
                                     const std::map<std::string, CeedInt> &constants = {});
CEED_INTERN int CeedGetKernel_Sycl(Ceed ceed, const SyclModule_t *sycl_module, const std::string &kernel_name, sycl::kernel **sycl_kernel);

#endif  // _ceed_sycl_compile_h
