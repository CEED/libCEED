// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <sycl/sycl.hpp>
#include "magma_v2.h"

//////////////////////////////////////////////////////////////////////////////////////////
// NonTensor weight function
extern "C" void magma_weight_nontensor(magma_int_t grid, magma_int_t threads, magma_int_t nelem, magma_int_t Q, CeedScalar *dqweight, CeedScalar *dv,
                                       magma_queue_t queue) {
  sycl::queue *sycl_queue = magma_queue_get_sycl_stream(queue);
  sycl_queue
      ->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid * threads), sycl::range<1>(threads)), [=](sycl::nd_item<1> item_ct1) {
          const int tid = item_ct1.get_local_id(0);
          // TODO load qweight in shared memory if blockDim.z > 1?
          // Currently removed the loop that the CUDA/HIP version has, since we always set 1 element per block
          CeedInt elem       = item_ct1.get_group(0);
          dv[elem * Q + tid] = dqweight[tid];
        });
      })
      .wait();
}
