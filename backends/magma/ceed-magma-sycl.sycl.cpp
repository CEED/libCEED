// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <magma_v2.h>
#include <string>
#include <sycl/sycl.hpp>

#include "ceed-magma.h"
#include "ceed-magma-sycl.h"
#include "../sycl/ceed-sycl-common.hpp"

int CeedInitMagma_Sycl(Ceed ceed, int device_id) {

  Ceed_Magma *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  // This function is mostly a copy of CeedInit_Sycl for the main SYCL backends
  auto sycl_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  int  device_count = sycl_devices.size();

  if (0 == device_count) {
    return CeedError(ceed, CEED_ERROR_BACKEND, "No SYCL devices of GPU type are available");
  }

  // Validate the requested device_id
  if (device_count < device_id + 1) {
    return CeedError(ceed, CEED_ERROR_BACKEND, "Invalid SYCL device id requested");
  }

  sycl::device sycl_device{sycl_devices[device_id]};
  // Check that the device supports explicit device allocations
  if (!sycl_device.has(sycl::aspect::usm_device_allocations)) {
    return CeedError(ceed, CEED_ERROR_BACKEND,
                     "The requested SYCL device does not support explicit "
                     "device allocations.");
  }

  // Creating an asynchronous error handler
  sycl::async_handler sycl_async_handler = [&](sycl::exception_list exceptionList) {
    for (std::exception_ptr const &e : exceptionList) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception const &e) {
        std::ostringstream error_msg;
        error_msg << "SYCL asynchronous exception caught:\n";
        error_msg << e.what() << std::endl;
        return CeedError(ceed, CEED_ERROR_BACKEND, error_msg.str().c_str());
      }
    }
    return CEED_ERROR_SUCCESS;
  };

  sycl::context sycl_context{sycl_device.get_platform().get_devices()};
  sycl::queue *sycl_queue = new sycl::queue(sycl_context, sycl_device, sycl_async_handler,
		                            {sycl::property::queue::in_order()});

  data->device = device_id;
  magma_queue_create_from_sycl(data->device, sycl_queue, sycl_queue, sycl_queue, &(data->queue));

  return CEED_ERROR_SUCCESS;
}

// Set *handle to point to a sycl queue (C-friendly way to access the
// sycl::queue inside magma_queue_t)
CEED_INTERN 
int CeedMagmaGetSyclHandle(Ceed ceed, void **handle) { 
  Ceed_Magma *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  *handle = (void *) magma_queue_get_sycl_stream(data->queue);

  return CEED_ERROR_SUCCESS;
}

// Backend implementation of CeedSetStream
int CeedSetStream_Magma(Ceed ceed, void *handle) {
  Ceed_Magma *data;
  CeedCallBackend(CeedGetData(ceed, &data));

  sycl::queue *q = static_cast<sycl::queue *>(handle);

  // Check that the queue is in-order for now
  CeedCheck(q->is_in_order(), ceed, CEED_ERROR_BACKEND, "SYCL queue must be in order for MAGMA backend");

  // Ensure we are using the expected device
  auto sycl_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  CeedCheck(sycl_devices[data->device] == q->get_device(), ceed, CEED_ERROR_BACKEND, "Device mismatch between provided queue and ceed object");
  magma_queue_create_from_sycl(data->device, q, NULL, NULL, &(data->queue));

  // Set the sycl-ref delegate stream to match this one
  Ceed ceed_delegate = NULL;
  CeedCallBackend(CeedGetDelegate(ceed, &ceed_delegate));
  if (ceed_delegate) {
    Ceed_Sycl *delegate_data;
    CeedCallBackend(CeedGetData(ceed_delegate, &delegate_data));
    delegate_data->sycl_device  = q->get_device();
    delegate_data->sycl_context = q->get_context();
    delegate_data->sycl_queue   = *q;
  }

  return CEED_ERROR_SUCCESS;
}

CEED_INTERN void CeedMagmaQueueSync_Sycl(magma_queue_t queue) {
    magma_queue_get_sycl_stream(queue)->wait();
}

// C++ wrapper for MKL GEMM routine
CEED_INTERN int mkl_gemm_batched_strided(magma_trans_t transA, magma_trans_t transB,
		                         magma_int_t m, magma_int_t n, magma_int_t k,
					 CeedScalar alpha,
                                         const CeedScalar *dA, magma_int_t ldda, magma_int_t strideA,
					 const CeedScalar *dB, magma_int_t lddb, magma_int_t strideB,
                                         CeedScalar beta,
					 CeedScalar *dC, magma_int_t lddc, magma_int_t strideC,
                                         magma_int_t batchCount, magma_queue_t queue) {

	sycl::queue *sycl_queue = magma_queue_get_sycl_stream(queue);
     	oneapi::mkl::transpose transpose_ct1 = syclblas_trans_const(transA);
        oneapi::mkl::transpose transpose_ct2 = syclblas_trans_const(transB);
        oneapi::mkl::blas::column_major::gemm_batch(
                                 *sycl_queue, transpose_ct1, transpose_ct2,
                                 std::int64_t(m), std::int64_t(n), std::int64_t(k),
                                 alpha,
                                 dA, std::int64_t(ldda), std::int64_t(strideA),
                                 dB, std::int64_t(lddb), std::int64_t(strideB),
                                 beta,
                                 dC, std::int64_t(lddc), std::int64_t(strideC),
                                 std::int64_t(batchCount), {});
	return CEED_ERROR_SUCCESS;
}
