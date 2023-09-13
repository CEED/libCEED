// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other
// CEED contributors. All Rights Reserved. See the top-level LICENSE and NOTICE
// files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <ceed/jit-tools.h>

#include <string>
#include <sycl/sycl.hpp>

#include "../sycl/ceed-sycl-compile.hpp"
#include "ceed-sycl-ref.hpp"

class CeedElemRestrSyclStridedNT;
class CeedElemRestrSyclOffsetNT;
class CeedElemRestrSyclStridedT;
class CeedElemRestrSyclOffsetT;

//------------------------------------------------------------------------------
// Restriction Kernel : L-vector -> E-vector, strided
//------------------------------------------------------------------------------
static int CeedElemRestrictionStridedNoTranspose_Sycl(sycl::queue &sycl_queue, const CeedElemRestriction_Sycl *impl, const CeedScalar *u,
                                                      CeedScalar *v) {
  const CeedInt  elem_size    = impl->elem_size;
  const CeedInt  num_elem     = impl->num_elem;
  const CeedInt  num_comp     = impl->num_comp;
  const CeedInt  stride_nodes = impl->strides[0];
  const CeedInt  stride_comp  = impl->strides[1];
  const CeedInt  stride_elem  = impl->strides[2];
  sycl::range<1> kernel_range(num_elem * elem_size);

  // Order queue
  sycl::event e = sycl_queue.ext_oneapi_submit_barrier();
  sycl_queue.parallel_for<CeedElemRestrSyclStridedNT>(kernel_range, {e}, [=](sycl::id<1> node) {
    const CeedInt loc_node = node % elem_size;
    const CeedInt elem     = node / elem_size;

    for (CeedInt comp = 0; comp < num_comp; comp++) {
      v[loc_node + comp * elem_size * num_elem + elem * elem_size] = u[loc_node * stride_nodes + comp * stride_comp + elem * stride_elem];
    }
  });
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Restriction Kernel : L-vector -> E-vector, offsets provided
//------------------------------------------------------------------------------
static int CeedElemRestrictionOffsetNoTranspose_Sycl(sycl::queue &sycl_queue, const CeedElemRestriction_Sycl *impl, const CeedScalar *u,
                                                     CeedScalar *v) {
  const CeedInt  elem_size   = impl->elem_size;
  const CeedInt  num_elem    = impl->num_elem;
  const CeedInt  num_comp    = impl->num_comp;
  const CeedInt  comp_stride = impl->comp_stride;
  const CeedInt *indices     = impl->d_ind;

  sycl::range<1> kernel_range(num_elem * elem_size);

  // Order queue
  sycl::event e = sycl_queue.ext_oneapi_submit_barrier();
  sycl_queue.parallel_for<CeedElemRestrSyclOffsetNT>(kernel_range, {e}, [=](sycl::id<1> node) {
    const CeedInt ind      = indices[node];
    const CeedInt loc_node = node % elem_size;
    const CeedInt elem     = node / elem_size;

    for (CeedInt comp = 0; comp < num_comp; comp++) {
      v[loc_node + comp * elem_size * num_elem + elem * elem_size] = u[ind + comp * comp_stride];
    }
  });
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Kernel: E-vector -> L-vector, strided
//------------------------------------------------------------------------------
static int CeedElemRestrictionStridedTranspose_Sycl(sycl::queue &sycl_queue, const CeedElemRestriction_Sycl *impl, const CeedScalar *u,
                                                    CeedScalar *v) {
  const CeedInt elem_size    = impl->elem_size;
  const CeedInt num_elem     = impl->num_elem;
  const CeedInt num_comp     = impl->num_comp;
  const CeedInt stride_nodes = impl->strides[0];
  const CeedInt stride_comp  = impl->strides[1];
  const CeedInt stride_elem  = impl->strides[2];

  sycl::range<1> kernel_range(num_elem * elem_size);

  // Order queue
  sycl::event e = sycl_queue.ext_oneapi_submit_barrier();
  sycl_queue.parallel_for<CeedElemRestrSyclStridedT>(kernel_range, {e}, [=](sycl::id<1> node) {
    const CeedInt loc_node = node % elem_size;
    const CeedInt elem     = node / elem_size;

    for (CeedInt comp = 0; comp < num_comp; comp++) {
      v[loc_node * stride_nodes + comp * stride_comp + elem * stride_elem] += u[loc_node + comp * elem_size * num_elem + elem * elem_size];
    }
  });
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Kernel: E-vector -> L-vector, offsets provided
//------------------------------------------------------------------------------
static int CeedElemRestrictionOffsetTranspose_Sycl(sycl::queue &sycl_queue, const CeedElemRestriction_Sycl *impl, const CeedScalar *u,
                                                   CeedScalar *v) {
  const CeedInt  num_nodes     = impl->num_nodes;
  const CeedInt  elem_size     = impl->elem_size;
  const CeedInt  num_elem      = impl->num_elem;
  const CeedInt  num_comp      = impl->num_comp;
  const CeedInt  comp_stride   = impl->comp_stride;
  const CeedInt *l_vec_indices = impl->d_l_vec_indices;
  const CeedInt *t_offsets     = impl->d_t_offsets;
  const CeedInt *t_indices     = impl->d_t_indices;

  sycl::range<1> kernel_range(num_nodes * num_comp);

  // Order queue
  sycl::event e = sycl_queue.ext_oneapi_submit_barrier();
  sycl_queue.parallel_for<CeedElemRestrSyclOffsetT>(kernel_range, {e}, [=](sycl::id<1> id) {
    const CeedInt node    = id % num_nodes;
    const CeedInt comp    = id / num_nodes;
    const CeedInt ind     = l_vec_indices[node];
    const CeedInt range_1 = t_offsets[node];
    const CeedInt range_N = t_offsets[node + 1];
    CeedScalar    value   = 0.0;

    for (CeedInt j = range_1; j < range_N; j++) {
      const CeedInt t_ind    = t_indices[j];
      CeedInt       loc_node = t_ind % elem_size;
      CeedInt       elem     = t_ind / elem_size;

      value += u[loc_node + comp * elem_size * num_elem + elem * elem_size];
    }
    v[ind + comp * comp_stride] += value;
  });
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Apply restriction
//------------------------------------------------------------------------------
static int CeedElemRestrictionApply_Sycl(CeedElemRestriction r, CeedTransposeMode t_mode, CeedVector u, CeedVector v, CeedRequest *request) {
  Ceed                      ceed;
  Ceed_Sycl                *data;
  const CeedScalar         *d_u;
  CeedScalar               *d_v;
  CeedElemRestriction_Sycl *impl;

  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));
  CeedCallBackend(CeedGetData(ceed, &data));

  // Get vectors
  CeedCallBackend(CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u));
  if (t_mode == CEED_TRANSPOSE) {
    // Sum into for transpose mode, e-vec to l-vec
    CeedCallBackend(CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v));
  } else {
    // Overwrite for notranspose mode, l-vec to e-vec
    CeedCallBackend(CeedVectorGetArrayWrite(v, CEED_MEM_DEVICE, &d_v));
  }

  // Restrict
  if (t_mode == CEED_NOTRANSPOSE) {
    // L-vector -> E-vector
    if (impl->d_ind) {
      // -- Offsets provided
      CeedCallBackend(CeedElemRestrictionOffsetNoTranspose_Sycl(data->sycl_queue, impl, d_u, d_v));
    } else {
      // -- Strided restriction
      CeedCallBackend(CeedElemRestrictionStridedNoTranspose_Sycl(data->sycl_queue, impl, d_u, d_v));
    }
  } else {
    // E-vector -> L-vector
    if (impl->d_ind) {
      // -- Offsets provided
      CeedCallBackend(CeedElemRestrictionOffsetTranspose_Sycl(data->sycl_queue, impl, d_u, d_v));
    } else {
      // -- Strided restriction
      CeedCallBackend(CeedElemRestrictionStridedTranspose_Sycl(data->sycl_queue, impl, d_u, d_v));
    }
  }
  // Wait for queues to be completed. NOTE: This may not be necessary
  CeedCallSycl(ceed, data->sycl_queue.wait_and_throw());

  if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_ORDERED) *request = NULL;

  // Restore arrays
  CeedCallBackend(CeedVectorRestoreArrayRead(u, &d_u));
  CeedCallBackend(CeedVectorRestoreArray(v, &d_v));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Get offsets
//------------------------------------------------------------------------------
static int CeedElemRestrictionGetOffsets_Sycl(CeedElemRestriction r, CeedMemType m_type, const CeedInt **offsets) {
  Ceed                      ceed;
  CeedElemRestriction_Sycl *impl;

  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));

  switch (m_type) {
    case CEED_MEM_HOST:
      *offsets = impl->h_ind;
      break;
    case CEED_MEM_DEVICE:
      *offsets = impl->d_ind;
      break;
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Destroy restriction
//------------------------------------------------------------------------------
static int CeedElemRestrictionDestroy_Sycl(CeedElemRestriction r) {
  Ceed                      ceed;
  Ceed_Sycl                *data;
  CeedElemRestriction_Sycl *impl;

  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));
  CeedCallBackend(CeedGetData(ceed, &data));

  // Wait for all work to finish before freeing memory
  CeedCallSycl(ceed, data->sycl_queue.wait_and_throw());

  CeedCallBackend(CeedFree(&impl->h_ind_allocated));
  CeedCallSycl(ceed, sycl::free(impl->d_ind_allocated, data->sycl_context));
  CeedCallSycl(ceed, sycl::free(impl->d_t_offsets, data->sycl_context));
  CeedCallSycl(ceed, sycl::free(impl->d_t_indices, data->sycl_context));
  CeedCallSycl(ceed, sycl::free(impl->d_l_vec_indices, data->sycl_context));
  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create transpose offsets and indices
//------------------------------------------------------------------------------
static int CeedElemRestrictionOffset_Sycl(const CeedElemRestriction r, const CeedInt *indices) {
  Ceed                      ceed;
  Ceed_Sycl                *data;
  bool                     *is_node;
  CeedSize                  l_size;
  CeedInt                   num_elem, elem_size, num_comp, num_nodes = 0, *ind_to_offset, *l_vec_indices, *t_offsets, *t_indices;
  CeedElemRestriction_Sycl *impl;

  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));
  CeedCallBackend(CeedElemRestrictionGetNumElements(r, &num_elem));
  CeedCallBackend(CeedElemRestrictionGetElementSize(r, &elem_size));
  CeedCallBackend(CeedElemRestrictionGetLVectorSize(r, &l_size));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(r, &num_comp));

  // Count num_nodes
  CeedCallBackend(CeedCalloc(l_size, &is_node));
  const CeedInt size_indices = num_elem * elem_size;

  for (CeedInt i = 0; i < size_indices; i++) is_node[indices[i]] = 1;
  for (CeedInt i = 0; i < l_size; i++) num_nodes += is_node[i];
  impl->num_nodes = num_nodes;

  // L-vector offsets array
  CeedCallBackend(CeedCalloc(l_size, &ind_to_offset));
  CeedCallBackend(CeedCalloc(num_nodes, &l_vec_indices));
  for (CeedInt i = 0, j = 0; i < l_size; i++) {
    if (is_node[i]) {
      l_vec_indices[j] = i;
      ind_to_offset[i] = j++;
    }
  }
  CeedCallBackend(CeedFree(&is_node));

  // Compute transpose offsets and indices
  const CeedInt size_offsets = num_nodes + 1;

  CeedCallBackend(CeedCalloc(size_offsets, &t_offsets));
  CeedCallBackend(CeedMalloc(size_indices, &t_indices));
  // Count node multiplicity
  for (CeedInt e = 0; e < num_elem; ++e) {
    for (CeedInt i = 0; i < elem_size; ++i) ++t_offsets[ind_to_offset[indices[elem_size * e + i]] + 1];
  }
  // Convert to running sum
  for (CeedInt i = 1; i < size_offsets; ++i) t_offsets[i] += t_offsets[i - 1];
  // List all E-vec indices associated with L-vec node
  for (CeedInt e = 0; e < num_elem; ++e) {
    for (CeedInt i = 0; i < elem_size; ++i) {
      const CeedInt lid                          = elem_size * e + i;
      const CeedInt gid                          = indices[lid];
      t_indices[t_offsets[ind_to_offset[gid]]++] = lid;
    }
  }
  // Reset running sum
  for (int i = size_offsets - 1; i > 0; --i) t_offsets[i] = t_offsets[i - 1];
  t_offsets[0] = 0;

  // Copy data to device
  CeedCallBackend(CeedGetData(ceed, &data));

  // Order queue
  sycl::event e = data->sycl_queue.ext_oneapi_submit_barrier();

  // -- L-vector indices
  CeedCallSycl(ceed, impl->d_l_vec_indices = sycl::malloc_device<CeedInt>(num_nodes, data->sycl_device, data->sycl_context));
  sycl::event copy_lvec = data->sycl_queue.copy<CeedInt>(l_vec_indices, impl->d_l_vec_indices, num_nodes, {e});
  // -- Transpose offsets
  CeedCallSycl(ceed, impl->d_t_offsets = sycl::malloc_device<CeedInt>(size_offsets, data->sycl_device, data->sycl_context));
  sycl::event copy_offsets = data->sycl_queue.copy<CeedInt>(t_offsets, impl->d_t_offsets, size_offsets, {e});
  // -- Transpose indices
  CeedCallSycl(ceed, impl->d_t_indices = sycl::malloc_device<CeedInt>(size_indices, data->sycl_device, data->sycl_context));
  sycl::event copy_indices = data->sycl_queue.copy<CeedInt>(t_indices, impl->d_t_indices, size_indices, {e});

  // Wait for all copies to complete and handle exceptions
  CeedCallSycl(ceed, sycl::event::wait_and_throw({copy_lvec, copy_offsets, copy_indices}));

  // Cleanup
  CeedCallBackend(CeedFree(&ind_to_offset));
  CeedCallBackend(CeedFree(&l_vec_indices));
  CeedCallBackend(CeedFree(&t_offsets));
  CeedCallBackend(CeedFree(&t_indices));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Create restriction
//------------------------------------------------------------------------------
int CeedElemRestrictionCreate_Sycl(CeedMemType mem_type, CeedCopyMode copy_mode, const CeedInt *indices, const bool *orients,
                                   const CeedInt8 *curl_orients, CeedElemRestriction r) {
  Ceed                      ceed;
  Ceed_Sycl                *data;
  bool                      is_strided;
  CeedInt                   num_elem, num_comp, elem_size, comp_stride = 1;
  CeedRestrictionType       rstr_type;
  CeedElemRestriction_Sycl *impl;

  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));
  CeedCallBackend(CeedGetData(ceed, &data));
  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedElemRestrictionGetNumElements(r, &num_elem));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(r, &num_comp));
  CeedCallBackend(CeedElemRestrictionGetElementSize(r, &elem_size));
  CeedInt size       = num_elem * elem_size;
  CeedInt strides[3] = {1, size, elem_size};

  CeedCallBackend(CeedElemRestrictionGetType(r, &rstr_type));
  CeedCheck(rstr_type != CEED_RESTRICTION_ORIENTED && rstr_type != CEED_RESTRICTION_CURL_ORIENTED, ceed, CEED_ERROR_BACKEND,
            "Backend does not implement CeedElemRestrictionCreateOriented or CeedElemRestrictionCreateCurlOriented");

  // Stride data
  CeedCallBackend(CeedElemRestrictionIsStrided(r, &is_strided));
  if (is_strided) {
    bool has_backend_strides;

    CeedCallBackend(CeedElemRestrictionHasBackendStrides(r, &has_backend_strides));
    if (!has_backend_strides) {
      CeedCallBackend(CeedElemRestrictionGetStrides(r, &strides));
    }
  } else {
    CeedCallBackend(CeedElemRestrictionGetCompStride(r, &comp_stride));
  }

  impl->h_ind           = NULL;
  impl->h_ind_allocated = NULL;
  impl->d_ind           = NULL;
  impl->d_ind_allocated = NULL;
  impl->d_t_indices     = NULL;
  impl->d_t_offsets     = NULL;
  impl->num_nodes       = size;
  impl->num_elem        = num_elem;
  impl->num_comp        = num_comp;
  impl->elem_size       = elem_size;
  impl->comp_stride     = comp_stride;
  impl->strides[0]      = strides[0];
  impl->strides[1]      = strides[1];
  impl->strides[2]      = strides[2];
  CeedCallBackend(CeedElemRestrictionSetData(r, impl));
  CeedInt layout[3] = {1, elem_size * num_elem, elem_size};
  CeedCallBackend(CeedElemRestrictionSetELayout(r, layout));

  // Set up device indices/offset arrays
  if (mem_type == CEED_MEM_HOST) {
    switch (copy_mode) {
      case CEED_OWN_POINTER:
        impl->h_ind_allocated = (CeedInt *)indices;
        impl->h_ind           = (CeedInt *)indices;
        break;
      case CEED_USE_POINTER:
        impl->h_ind = (CeedInt *)indices;
        break;
      case CEED_COPY_VALUES:
        if (indices != NULL) {
          CeedCallBackend(CeedMalloc(elem_size * num_elem, &impl->h_ind_allocated));
          memcpy(impl->h_ind_allocated, indices, elem_size * num_elem * sizeof(CeedInt));
          impl->h_ind = impl->h_ind_allocated;
        }
        break;
    }
    if (indices != NULL) {
      CeedCallSycl(ceed, impl->d_ind = sycl::malloc_device<CeedInt>(size, data->sycl_device, data->sycl_context));
      impl->d_ind_allocated = impl->d_ind;  // We own the device memory
      // Order queue
      sycl::event e = data->sycl_queue.ext_oneapi_submit_barrier();
      // Copy from host to device
      sycl::event copy_event = data->sycl_queue.copy<CeedInt>(indices, impl->d_ind, size, {e});
      // Wait for copy to finish and handle exceptions
      CeedCallSycl(ceed, copy_event.wait_and_throw());
      CeedCallBackend(CeedElemRestrictionOffset_Sycl(r, indices));
    }
  } else if (mem_type == CEED_MEM_DEVICE) {
    switch (copy_mode) {
      case CEED_COPY_VALUES:
        if (indices != NULL) {
          CeedCallSycl(ceed, impl->d_ind = sycl::malloc_device<CeedInt>(size, data->sycl_device, data->sycl_context));
          impl->d_ind_allocated = impl->d_ind;  // We own the device memory
                                                // Copy from device to device
          // Order queue
          sycl::event e          = data->sycl_queue.ext_oneapi_submit_barrier();
          sycl::event copy_event = data->sycl_queue.copy<CeedInt>(indices, impl->d_ind, size, {e});
          // Wait for copy to finish and handle exceptions
          CeedCallSycl(ceed, copy_event.wait_and_throw());
        }
        break;
      case CEED_OWN_POINTER:
        impl->d_ind           = (CeedInt *)indices;
        impl->d_ind_allocated = impl->d_ind;
        break;
      case CEED_USE_POINTER:
        impl->d_ind = (CeedInt *)indices;
    }
    if (indices != NULL) {
      CeedCallBackend(CeedMalloc(elem_size * num_elem, &impl->h_ind_allocated));
      // Order queue
      sycl::event e = data->sycl_queue.ext_oneapi_submit_barrier();
      // Copy from device to host
      sycl::event copy_event = data->sycl_queue.copy<CeedInt>(impl->d_ind, impl->h_ind_allocated, elem_size * num_elem, {e});
      CeedCallSycl(ceed, copy_event.wait_and_throw());
      impl->h_ind = impl->h_ind_allocated;
      CeedCallBackend(CeedElemRestrictionOffset_Sycl(r, indices));
    }
  } else {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_BACKEND, "Only MemType = HOST or DEVICE supported");
    // LCOV_EXCL_STOP
  }

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "ElemRestriction", r, "Apply", CeedElemRestrictionApply_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "ElemRestriction", r, "ApplyUnsigned", CeedElemRestrictionApply_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "ElemRestriction", r, "ApplyUnoriented", CeedElemRestrictionApply_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "ElemRestriction", r, "GetOffsets", CeedElemRestrictionGetOffsets_Sycl));
  CeedCallBackend(CeedSetBackendFunctionCpp(ceed, "ElemRestriction", r, "Destroy", CeedElemRestrictionDestroy_Sycl));
  return CEED_ERROR_SUCCESS;
}
