// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "ceed-ref.h"

//------------------------------------------------------------------------------
// Core ElemRestriction Apply Code
//------------------------------------------------------------------------------
static inline int CeedElemRestrictionApplyStridedNoTranspose_Ref_Core(CeedElemRestriction rstr, const CeedInt num_comp, const CeedInt block_size,
                                                                      CeedInt start, CeedInt stop, CeedInt num_elem, CeedInt elem_size,
                                                                      CeedSize v_offset, const CeedScalar *__restrict__ uu,
                                                                      CeedScalar *__restrict__ vv) {
  // No offsets provided, identity restriction
  bool has_backend_strides;

  CeedCallBackend(CeedElemRestrictionHasBackendStrides(rstr, &has_backend_strides));
  if (has_backend_strides) {
    // CPU backend strides are {1, elem_size, elem_size*num_comp}
    // This if branch is left separate to allow better inlining
    for (CeedSize e = start * block_size; e < stop * block_size; e += block_size) {
      CeedPragmaSIMD for (CeedSize k = 0; k < num_comp; k++) {
        CeedPragmaSIMD for (CeedSize n = 0; n < elem_size; n++) {
          CeedPragmaSIMD for (CeedSize j = 0; j < block_size; j++) {
            vv[e * elem_size * num_comp + (k * elem_size + n) * block_size + j - v_offset] =
                uu[n + k * elem_size + CeedIntMin(e + j, num_elem - 1) * elem_size * (CeedSize)num_comp];
          }
        }
      }
    }
  } else {
    // User provided strides
    CeedInt strides[3];

    CeedCallBackend(CeedElemRestrictionGetStrides(rstr, strides));
    for (CeedSize e = start * block_size; e < stop * block_size; e += block_size) {
      CeedPragmaSIMD for (CeedSize k = 0; k < num_comp; k++) {
        CeedPragmaSIMD for (CeedSize n = 0; n < elem_size; n++) {
          CeedPragmaSIMD for (CeedSize j = 0; j < block_size; j++) {
            vv[e * elem_size * num_comp + (k * elem_size + n) * block_size + j - v_offset] =
                uu[n * strides[0] + k * strides[1] + CeedIntMin(e + j, num_elem - 1) * (CeedSize)strides[2]];
          }
        }
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

static inline int CeedElemRestrictionApplyOffsetNoTranspose_Ref_Core(CeedElemRestriction rstr, const CeedInt num_comp, const CeedInt block_size,
                                                                     const CeedInt comp_stride, CeedInt start, CeedInt stop, CeedInt num_elem,
                                                                     CeedInt elem_size, CeedSize v_offset, const CeedScalar *__restrict__ uu,
                                                                     CeedScalar *__restrict__ vv) {
  // Default restriction with offsets
  CeedElemRestriction_Ref *impl;

  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));
  for (CeedSize e = start * block_size; e < stop * block_size; e += block_size) {
    CeedPragmaSIMD for (CeedSize k = 0; k < num_comp; k++) {
      CeedPragmaSIMD for (CeedSize i = 0; i < elem_size * block_size; i++) {
        vv[elem_size * (k * block_size + e * num_comp) + i - v_offset] = uu[impl->offsets[i + e * elem_size] + k * comp_stride];
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

static inline int CeedElemRestrictionApplyOrientedNoTranspose_Ref_Core(CeedElemRestriction rstr, const CeedInt num_comp, const CeedInt block_size,
                                                                       const CeedInt comp_stride, CeedInt start, CeedInt stop, CeedInt num_elem,
                                                                       CeedInt elem_size, CeedSize v_offset, const CeedScalar *__restrict__ uu,
                                                                       CeedScalar *__restrict__ vv) {
  // Restriction with orientations
  CeedElemRestriction_Ref *impl;

  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));
  for (CeedSize e = start * block_size; e < stop * block_size; e += block_size) {
    CeedPragmaSIMD for (CeedSize k = 0; k < num_comp; k++) {
      CeedPragmaSIMD for (CeedSize i = 0; i < elem_size * block_size; i++) {
        vv[elem_size * (k * block_size + e * num_comp) + i - v_offset] =
            uu[impl->offsets[i + e * elem_size] + k * comp_stride] * (impl->orients[i + e * elem_size] ? -1.0 : 1.0);
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

static inline int CeedElemRestrictionApplyCurlOrientedNoTranspose_Ref_Core(CeedElemRestriction rstr, const CeedInt num_comp, const CeedInt block_size,
                                                                           const CeedInt comp_stride, CeedInt start, CeedInt stop, CeedInt num_elem,
                                                                           CeedInt elem_size, CeedSize v_offset, const CeedScalar *__restrict__ uu,
                                                                           CeedScalar *__restrict__ vv) {
  // Restriction with tridiagonal transformation
  CeedElemRestriction_Ref *impl;

  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));
  for (CeedSize e = start * block_size; e < stop * block_size; e += block_size) {
    CeedPragmaSIMD for (CeedSize k = 0; k < num_comp; k++) {
      CeedSize n = 0;

      CeedPragmaSIMD for (CeedSize j = 0; j < block_size; j++) {
        vv[e * elem_size * num_comp + (k * elem_size + n) * block_size + j - v_offset] =
            uu[impl->offsets[j + n * block_size + e * elem_size] + k * comp_stride] *
                impl->curl_orients[j + (3 * n + 1) * block_size + e * 3 * elem_size] +
            uu[impl->offsets[j + (n + 1) * block_size + e * elem_size] + k * comp_stride] *
                impl->curl_orients[j + (3 * n + 2) * block_size + e * 3 * elem_size];
      }
      CeedPragmaSIMD for (n = 1; n < elem_size - 1; n++) {
        CeedPragmaSIMD for (CeedSize j = 0; j < block_size; j++) {
          vv[e * elem_size * num_comp + (k * elem_size + n) * block_size + j - v_offset] =
              uu[impl->offsets[j + (n - 1) * block_size + e * elem_size] + k * comp_stride] *
                  impl->curl_orients[j + (3 * n + 0) * block_size + e * 3 * elem_size] +
              uu[impl->offsets[j + n * block_size + e * elem_size] + k * comp_stride] *
                  impl->curl_orients[j + (3 * n + 1) * block_size + e * 3 * elem_size] +
              uu[impl->offsets[j + (n + 1) * block_size + e * elem_size] + k * comp_stride] *
                  impl->curl_orients[j + (3 * n + 2) * block_size + e * 3 * elem_size];
        }
      }
      CeedPragmaSIMD for (CeedSize j = 0; j < block_size; j++) {
        vv[e * elem_size * num_comp + (k * elem_size + n) * block_size + j - v_offset] =
            uu[impl->offsets[j + (n - 1) * block_size + e * elem_size] + k * comp_stride] *
                impl->curl_orients[j + (3 * n + 0) * block_size + e * 3 * elem_size] +
            uu[impl->offsets[j + n * block_size + e * elem_size] + k * comp_stride] *
                impl->curl_orients[j + (3 * n + 1) * block_size + e * 3 * elem_size];
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

static inline int CeedElemRestrictionApplyCurlOrientedUnsignedNoTranspose_Ref_Core(CeedElemRestriction rstr, const CeedInt num_comp,
                                                                                   const CeedInt block_size, const CeedInt comp_stride, CeedInt start,
                                                                                   CeedInt stop, CeedInt num_elem, CeedInt elem_size,
                                                                                   CeedSize v_offset, const CeedScalar *__restrict__ uu,
                                                                                   CeedScalar *__restrict__ vv) {
  // Restriction with (unsigned) tridiagonal transformation
  CeedElemRestriction_Ref *impl;

  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));
  for (CeedSize e = start * block_size; e < stop * block_size; e += block_size) {
    CeedPragmaSIMD for (CeedSize k = 0; k < num_comp; k++) {
      CeedSize n = 0;

      CeedPragmaSIMD for (CeedSize j = 0; j < block_size; j++) {
        vv[e * elem_size * num_comp + (k * elem_size + n) * block_size + j - v_offset] =
            uu[impl->offsets[j + n * block_size + e * elem_size] + k * comp_stride] *
                abs(impl->curl_orients[j + (3 * n + 1) * block_size + e * 3 * elem_size]) +
            uu[impl->offsets[j + (n + 1) * block_size + e * elem_size] + k * comp_stride] *
                abs(impl->curl_orients[j + (3 * n + 2) * block_size + e * 3 * elem_size]);
      }
      CeedPragmaSIMD for (n = 1; n < elem_size - 1; n++) {
        CeedPragmaSIMD for (CeedSize j = 0; j < block_size; j++) {
          vv[e * elem_size * num_comp + (k * elem_size + n) * block_size + j - v_offset] =
              uu[impl->offsets[j + (n - 1) * block_size + e * elem_size] + k * comp_stride] *
                  abs(impl->curl_orients[j + (3 * n + 0) * block_size + e * 3 * elem_size]) +
              uu[impl->offsets[j + n * block_size + e * elem_size] + k * comp_stride] *
                  abs(impl->curl_orients[j + (3 * n + 1) * block_size + e * 3 * elem_size]) +
              uu[impl->offsets[j + (n + 1) * block_size + e * elem_size] + k * comp_stride] *
                  abs(impl->curl_orients[j + (3 * n + 2) * block_size + e * 3 * elem_size]);
        }
      }
      CeedPragmaSIMD for (CeedSize j = 0; j < block_size; j++) {
        vv[e * elem_size * num_comp + (k * elem_size + n) * block_size + j - v_offset] =
            uu[impl->offsets[j + (n - 1) * block_size + e * elem_size] + k * comp_stride] *
                abs(impl->curl_orients[j + (3 * n + 0) * block_size + e * 3 * elem_size]) +
            uu[impl->offsets[j + n * block_size + e * elem_size] + k * comp_stride] *
                abs(impl->curl_orients[j + (3 * n + 1) * block_size + e * 3 * elem_size]);
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

static inline int CeedElemRestrictionApplyStridedTranspose_Ref_Core(CeedElemRestriction rstr, const CeedInt num_comp, const CeedInt block_size,
                                                                    CeedInt start, CeedInt stop, CeedInt num_elem, CeedInt elem_size,
                                                                    CeedSize v_offset, const CeedScalar *__restrict__ uu,
                                                                    CeedScalar *__restrict__ vv) {
  // No offsets provided, identity restriction
  bool has_backend_strides;

  CeedCallBackend(CeedElemRestrictionHasBackendStrides(rstr, &has_backend_strides));
  if (has_backend_strides) {
    // CPU backend strides are {1, elem_size, elem_size*num_comp}
    // This if brach is left separate to allow better inlining
    for (CeedSize e = start * block_size; e < stop * block_size; e += block_size) {
      CeedPragmaSIMD for (CeedSize k = 0; k < num_comp; k++) {
        CeedPragmaSIMD for (CeedSize n = 0; n < elem_size; n++) {
          CeedPragmaSIMD for (CeedSize j = 0; j < CeedIntMin(block_size, num_elem - e); j++) {
            vv[n + k * elem_size + (e + j) * elem_size * num_comp] += uu[e * elem_size * num_comp + (k * elem_size + n) * block_size + j - v_offset];
          }
        }
      }
    }
  } else {
    // User provided strides
    CeedInt strides[3];

    CeedCallBackend(CeedElemRestrictionGetStrides(rstr, strides));
    for (CeedInt e = start * block_size; e < stop * block_size; e += block_size) {
      CeedPragmaSIMD for (CeedSize k = 0; k < num_comp; k++) {
        CeedPragmaSIMD for (CeedSize n = 0; n < elem_size; n++) {
          CeedPragmaSIMD for (CeedSize j = 0; j < CeedIntMin(block_size, num_elem - e); j++) {
            vv[n * strides[0] + k * strides[1] + (e + j) * strides[2]] +=
                uu[e * elem_size * num_comp + (k * elem_size + n) * block_size + j - v_offset];
          }
        }
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

static inline int CeedElemRestrictionApplyOffsetTranspose_Ref_Core(CeedElemRestriction rstr, const CeedInt num_comp, const CeedInt block_size,
                                                                   const CeedInt comp_stride, CeedInt start, CeedInt stop, CeedInt num_elem,
                                                                   CeedInt elem_size, CeedSize v_offset, const CeedScalar *__restrict__ uu,
                                                                   CeedScalar *__restrict__ vv) {
  // Default restriction with offsets
  CeedElemRestriction_Ref *impl;

  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));
  for (CeedSize e = start * block_size; e < stop * block_size; e += block_size) {
    for (CeedSize k = 0; k < num_comp; k++) {
      for (CeedSize i = 0; i < elem_size * block_size; i += block_size) {
        // Iteration bound set to discard padding elements
        for (CeedSize j = i; j < i + CeedIntMin(block_size, num_elem - e); j++) {
          CeedScalar vv_loc;

          vv_loc = uu[elem_size * (k * block_size + e * num_comp) + j - v_offset];
          CeedPragmaAtomic vv[impl->offsets[j + e * elem_size] + k * comp_stride] += vv_loc;
        }
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

static inline int CeedElemRestrictionApplyOrientedTranspose_Ref_Core(CeedElemRestriction rstr, const CeedInt num_comp, const CeedInt block_size,
                                                                     const CeedInt comp_stride, CeedInt start, CeedInt stop, CeedInt num_elem,
                                                                     CeedInt elem_size, CeedSize v_offset, const CeedScalar *__restrict__ uu,
                                                                     CeedScalar *__restrict__ vv) {
  // Restriction with orientations
  CeedElemRestriction_Ref *impl;

  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));
  for (CeedSize e = start * block_size; e < stop * block_size; e += block_size) {
    for (CeedSize k = 0; k < num_comp; k++) {
      for (CeedSize i = 0; i < elem_size * block_size; i += block_size) {
        // Iteration bound set to discard padding elements
        for (CeedSize j = i; j < i + CeedIntMin(block_size, num_elem - e); j++) {
          CeedScalar vv_loc;

          vv_loc = uu[elem_size * (k * block_size + e * num_comp) + j - v_offset] * (impl->orients[j + e * elem_size] ? -1.0 : 1.0);
          CeedPragmaAtomic vv[impl->offsets[j + e * elem_size] + k * comp_stride] += vv_loc;
        }
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

static inline int CeedElemRestrictionApplyCurlOrientedTranspose_Ref_Core(CeedElemRestriction rstr, const CeedInt num_comp, const CeedInt block_size,
                                                                         const CeedInt comp_stride, CeedInt start, CeedInt stop, CeedInt num_elem,
                                                                         CeedInt elem_size, CeedSize v_offset, const CeedScalar *__restrict__ uu,
                                                                         CeedScalar *__restrict__ vv) {
  // Restriction with tridiagonal transformation
  CeedElemRestriction_Ref *impl;
  CeedScalar               vv_loc[block_size];

  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));
  for (CeedSize e = start * block_size; e < stop * block_size; e += block_size) {
    for (CeedSize k = 0; k < num_comp; k++) {
      // Iteration bound set to discard padding elements
      const CeedSize block_end = CeedIntMin(block_size, num_elem - e);
      CeedSize       n         = 0;

      CeedPragmaSIMD for (CeedSize j = 0; j < block_end; j++) {
        vv_loc[j] = uu[e * elem_size * num_comp + (k * elem_size + n) * block_size + j - v_offset] *
                        impl->curl_orients[j + (3 * n + 1) * block_size + e * 3 * elem_size] +
                    uu[e * elem_size * num_comp + (k * elem_size + n + 1) * block_size + j - v_offset] *
                        impl->curl_orients[j + (3 * n + 3) * block_size + e * 3 * elem_size];
      }
      for (CeedSize j = 0; j < block_end; j++) {
        CeedPragmaAtomic vv[impl->offsets[j + n * block_size + e * elem_size] + k * comp_stride] += vv_loc[j];
      }
      for (n = 1; n < elem_size - 1; n++) {
        CeedPragmaSIMD for (CeedInt j = 0; j < block_end; j++) {
          vv_loc[j] = uu[e * elem_size * num_comp + (k * elem_size + n - 1) * block_size + j - v_offset] *
                          impl->curl_orients[j + (3 * n - 1) * block_size + e * 3 * elem_size] +
                      uu[e * elem_size * num_comp + (k * elem_size + n) * block_size + j - v_offset] *
                          impl->curl_orients[j + (3 * n + 1) * block_size + e * 3 * elem_size] +
                      uu[e * elem_size * num_comp + (k * elem_size + n + 1) * block_size + j - v_offset] *
                          impl->curl_orients[j + (3 * n + 3) * block_size + e * 3 * elem_size];
        }
        for (CeedSize j = 0; j < block_end; j++) {
          CeedPragmaAtomic vv[impl->offsets[j + n * block_size + e * elem_size] + k * comp_stride] += vv_loc[j];
        }
      }
      CeedPragmaSIMD for (CeedSize j = 0; j < block_end; j++) {
        vv_loc[j] = uu[e * elem_size * num_comp + (k * elem_size + n - 1) * block_size + j - v_offset] *
                        impl->curl_orients[j + (3 * n - 1) * block_size + e * 3 * elem_size] +
                    uu[e * elem_size * num_comp + (k * elem_size + n) * block_size + j - v_offset] *
                        impl->curl_orients[j + (3 * n + 1) * block_size + e * 3 * elem_size];
      }
      for (CeedSize j = 0; j < block_end; j++) {
        CeedPragmaAtomic vv[impl->offsets[j + n * block_size + e * elem_size] + k * comp_stride] += vv_loc[j];
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

static inline int CeedElemRestrictionApplyCurlOrientedUnsignedTranspose_Ref_Core(CeedElemRestriction rstr, const CeedInt num_comp,
                                                                                 const CeedInt block_size, const CeedInt comp_stride, CeedInt start,
                                                                                 CeedInt stop, CeedInt num_elem, CeedInt elem_size, CeedSize v_offset,
                                                                                 const CeedScalar *__restrict__ uu, CeedScalar *__restrict__ vv) {
  // Restriction with (unsigned) tridiagonal transformation
  CeedElemRestriction_Ref *impl;
  CeedScalar               vv_loc[block_size];

  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));
  for (CeedSize e = start * block_size; e < stop * block_size; e += block_size) {
    for (CeedSize k = 0; k < num_comp; k++) {
      // Iteration bound set to discard padding elements
      const CeedSize block_end = CeedIntMin(block_size, num_elem - e);
      CeedSize       n         = 0;

      CeedPragmaSIMD for (CeedSize j = 0; j < block_end; j++) {
        vv_loc[j] = uu[e * elem_size * num_comp + (k * elem_size + n) * block_size + j - v_offset] *
                        abs(impl->curl_orients[j + (3 * n + 1) * block_size + e * 3 * elem_size]) +
                    uu[e * elem_size * num_comp + (k * elem_size + n + 1) * block_size + j - v_offset] *
                        abs(impl->curl_orients[j + (3 * n + 3) * block_size + e * 3 * elem_size]);
      }
      for (CeedSize j = 0; j < block_end; j++) {
        CeedPragmaAtomic vv[impl->offsets[j + n * block_size + e * elem_size] + k * comp_stride] += vv_loc[j];
      }
      for (n = 1; n < elem_size - 1; n++) {
        CeedPragmaSIMD for (CeedSize j = 0; j < block_end; j++) {
          vv_loc[j] = uu[e * elem_size * num_comp + (k * elem_size + n - 1) * block_size + j - v_offset] *
                          abs(impl->curl_orients[j + (3 * n - 1) * block_size + e * 3 * elem_size]) +
                      uu[e * elem_size * num_comp + (k * elem_size + n) * block_size + j - v_offset] *
                          abs(impl->curl_orients[j + (3 * n + 1) * block_size + e * 3 * elem_size]) +
                      uu[e * elem_size * num_comp + (k * elem_size + n + 1) * block_size + j - v_offset] *
                          abs(impl->curl_orients[j + (3 * n + 3) * block_size + e * 3 * elem_size]);
        }
        for (CeedSize j = 0; j < block_end; j++) {
          CeedPragmaAtomic vv[impl->offsets[j + n * block_size + e * elem_size] + k * comp_stride] += vv_loc[j];
        }
      }
      CeedPragmaSIMD for (CeedSize j = 0; j < block_end; j++) {
        vv_loc[j] = uu[e * elem_size * num_comp + (k * elem_size + n - 1) * block_size + j - v_offset] *
                        abs(impl->curl_orients[j + (3 * n - 1) * block_size + e * 3 * elem_size]) +
                    uu[e * elem_size * num_comp + (k * elem_size + n) * block_size + j - v_offset] *
                        abs(impl->curl_orients[j + (3 * n + 1) * block_size + e * 3 * elem_size]);
      }
      for (CeedSize j = 0; j < block_end; j++) {
        CeedPragmaAtomic vv[impl->offsets[j + n * block_size + e * elem_size] + k * comp_stride] += vv_loc[j];
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

static inline int CeedElemRestrictionApplyAtPointsInElement_Ref_Core(CeedElemRestriction rstr, const CeedInt num_comp, CeedInt start, CeedInt stop,
                                                                     CeedTransposeMode t_mode, const CeedScalar *__restrict__ uu,
                                                                     CeedScalar *__restrict__ vv) {
  CeedInt                  num_points, l_vec_offset;
  CeedSize                 e_vec_offset = 0;
  CeedElemRestriction_Ref *impl;

  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));
  for (CeedInt e = start; e < stop; e++) {
    l_vec_offset = impl->offsets[e];
    CeedCallBackend(CeedElemRestrictionGetNumPointsInElement(rstr, e, &num_points));
    if (t_mode == CEED_NOTRANSPOSE) {
      for (CeedSize i = 0; i < num_points; i++) {
        for (CeedSize j = 0; j < num_comp; j++) vv[j * num_points + i + e_vec_offset] = uu[impl->offsets[i + l_vec_offset] * num_comp + j];
      }
    } else {
      for (CeedSize i = 0; i < num_points; i++) {
        for (CeedSize j = 0; j < num_comp; j++) vv[impl->offsets[i + l_vec_offset] * num_comp + j] = uu[j * num_points + i + e_vec_offset];
      }
    }
    e_vec_offset += num_points * (CeedSize)num_comp;
  }
  return CEED_ERROR_SUCCESS;
}

static inline int CeedElemRestrictionApply_Ref_Core(CeedElemRestriction rstr, const CeedInt num_comp, const CeedInt block_size,
                                                    const CeedInt comp_stride, CeedInt start, CeedInt stop, CeedTransposeMode t_mode, bool use_signs,
                                                    bool use_orients, CeedVector u, CeedVector v, CeedRequest *request) {
  CeedInt             num_elem, elem_size;
  CeedSize            v_offset = 0;
  CeedRestrictionType rstr_type;
  const CeedScalar   *uu;
  CeedScalar         *vv;

  CeedCallBackend(CeedElemRestrictionGetNumElements(rstr, &num_elem));
  CeedCallBackend(CeedElemRestrictionGetElementSize(rstr, &elem_size));
  v_offset = start * block_size * elem_size * (CeedSize)num_comp;
  CeedCallBackend(CeedElemRestrictionGetType(rstr, &rstr_type));
  CeedCallBackend(CeedVectorGetArrayRead(u, CEED_MEM_HOST, &uu));

  if (t_mode == CEED_TRANSPOSE) {
    // Sum into for transpose mode, E-vector to L-vector
    CeedCallBackend(CeedVectorGetArray(v, CEED_MEM_HOST, &vv));
  } else {
    // Overwrite for notranspose mode, L-vector to E-vector
    CeedCallBackend(CeedVectorGetArrayWrite(v, CEED_MEM_HOST, &vv));
  }
  if (t_mode == CEED_TRANSPOSE) {
    // Restriction from E-vector to L-vector
    // Performing v += r^T * u
    // uu has shape [elem_size, num_comp, num_elem], row-major
    // vv has shape [nnodes, num_comp]
    // Sum into for transpose mode
    switch (rstr_type) {
      case CEED_RESTRICTION_STRIDED:
        CeedCallBackend(
            CeedElemRestrictionApplyStridedTranspose_Ref_Core(rstr, num_comp, block_size, start, stop, num_elem, elem_size, v_offset, uu, vv));
        break;
      case CEED_RESTRICTION_STANDARD:
        CeedCallBackend(CeedElemRestrictionApplyOffsetTranspose_Ref_Core(rstr, num_comp, block_size, comp_stride, start, stop, num_elem, elem_size,
                                                                         v_offset, uu, vv));
        break;
      case CEED_RESTRICTION_ORIENTED:
        if (use_signs) {
          CeedCallBackend(CeedElemRestrictionApplyOrientedTranspose_Ref_Core(rstr, num_comp, block_size, comp_stride, start, stop, num_elem,
                                                                             elem_size, v_offset, uu, vv));
        } else {
          CeedCallBackend(CeedElemRestrictionApplyOffsetTranspose_Ref_Core(rstr, num_comp, block_size, comp_stride, start, stop, num_elem, elem_size,
                                                                           v_offset, uu, vv));
        }
        break;
      case CEED_RESTRICTION_CURL_ORIENTED:
        if (use_signs && use_orients) {
          CeedCallBackend(CeedElemRestrictionApplyCurlOrientedTranspose_Ref_Core(rstr, num_comp, block_size, comp_stride, start, stop, num_elem,
                                                                                 elem_size, v_offset, uu, vv));
        } else if (use_orients) {
          CeedCallBackend(CeedElemRestrictionApplyCurlOrientedUnsignedTranspose_Ref_Core(rstr, num_comp, block_size, comp_stride, start, stop,
                                                                                         num_elem, elem_size, v_offset, uu, vv));
        } else {
          CeedCallBackend(CeedElemRestrictionApplyOffsetTranspose_Ref_Core(rstr, num_comp, block_size, comp_stride, start, stop, num_elem, elem_size,
                                                                           v_offset, uu, vv));
        }
        break;
      case CEED_RESTRICTION_POINTS:
        CeedCallBackend(CeedElemRestrictionApplyAtPointsInElement_Ref_Core(rstr, num_comp, start, stop, t_mode, uu, vv));
        break;
    }
  } else {
    // Restriction from L-vector to E-vector
    // Perform: v = r * u
    // vv has shape [elem_size, num_comp, num_elem], row-major
    // uu has shape [nnodes, num_comp]
    // Overwrite for notranspose mode
    switch (rstr_type) {
      case CEED_RESTRICTION_STRIDED:
        CeedCallBackend(
            CeedElemRestrictionApplyStridedNoTranspose_Ref_Core(rstr, num_comp, block_size, start, stop, num_elem, elem_size, v_offset, uu, vv));
        break;
      case CEED_RESTRICTION_STANDARD:
        CeedCallBackend(CeedElemRestrictionApplyOffsetNoTranspose_Ref_Core(rstr, num_comp, block_size, comp_stride, start, stop, num_elem, elem_size,
                                                                           v_offset, uu, vv));
        break;
      case CEED_RESTRICTION_ORIENTED:
        if (use_signs) {
          CeedCallBackend(CeedElemRestrictionApplyOrientedNoTranspose_Ref_Core(rstr, num_comp, block_size, comp_stride, start, stop, num_elem,
                                                                               elem_size, v_offset, uu, vv));
        } else {
          CeedCallBackend(CeedElemRestrictionApplyOffsetNoTranspose_Ref_Core(rstr, num_comp, block_size, comp_stride, start, stop, num_elem,
                                                                             elem_size, v_offset, uu, vv));
        }
        break;
      case CEED_RESTRICTION_CURL_ORIENTED:
        if (use_signs && use_orients) {
          CeedCallBackend(CeedElemRestrictionApplyCurlOrientedNoTranspose_Ref_Core(rstr, num_comp, block_size, comp_stride, start, stop, num_elem,
                                                                                   elem_size, v_offset, uu, vv));
        } else if (use_orients) {
          CeedCallBackend(CeedElemRestrictionApplyCurlOrientedUnsignedNoTranspose_Ref_Core(rstr, num_comp, block_size, comp_stride, start, stop,
                                                                                           num_elem, elem_size, v_offset, uu, vv));
        } else {
          CeedCallBackend(CeedElemRestrictionApplyOffsetNoTranspose_Ref_Core(rstr, num_comp, block_size, comp_stride, start, stop, num_elem,
                                                                             elem_size, v_offset, uu, vv));
        }
        break;
      case CEED_RESTRICTION_POINTS:
        CeedCallBackend(CeedElemRestrictionApplyAtPointsInElement_Ref_Core(rstr, num_comp, start, stop, t_mode, uu, vv));
        break;
    }
  }
  CeedCallBackend(CeedVectorRestoreArrayRead(u, &uu));
  CeedCallBackend(CeedVectorRestoreArray(v, &vv));
  if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_ORDERED) *request = NULL;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// ElemRestriction Apply - Common Sizes
//------------------------------------------------------------------------------
static int CeedElemRestrictionApply_Ref_110(CeedElemRestriction rstr, const CeedInt num_comp, const CeedInt block_size, const CeedInt comp_stride,
                                            CeedInt start, CeedInt stop, CeedTransposeMode t_mode, bool use_signs, bool use_orients, CeedVector u,
                                            CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(rstr, 1, 1, comp_stride, start, stop, t_mode, use_signs, use_orients, u, v, request);
}

static int CeedElemRestrictionApply_Ref_111(CeedElemRestriction rstr, const CeedInt num_comp, const CeedInt block_size, const CeedInt comp_stride,
                                            CeedInt start, CeedInt stop, CeedTransposeMode t_mode, bool use_signs, bool use_orients, CeedVector u,
                                            CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(rstr, 1, 1, 1, start, stop, t_mode, use_signs, use_orients, u, v, request);
}

static int CeedElemRestrictionApply_Ref_180(CeedElemRestriction rstr, const CeedInt num_comp, const CeedInt block_size, const CeedInt comp_stride,
                                            CeedInt start, CeedInt stop, CeedTransposeMode t_mode, bool use_signs, bool use_orients, CeedVector u,
                                            CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(rstr, 1, 8, comp_stride, start, stop, t_mode, use_signs, use_orients, u, v, request);
}

static int CeedElemRestrictionApply_Ref_181(CeedElemRestriction rstr, const CeedInt num_comp, const CeedInt block_size, const CeedInt comp_stride,
                                            CeedInt start, CeedInt stop, CeedTransposeMode t_mode, bool use_signs, bool use_orients, CeedVector u,
                                            CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(rstr, 1, 8, 1, start, stop, t_mode, use_signs, use_orients, u, v, request);
}

static int CeedElemRestrictionApply_Ref_310(CeedElemRestriction rstr, const CeedInt num_comp, const CeedInt block_size, const CeedInt comp_stride,
                                            CeedInt start, CeedInt stop, CeedTransposeMode t_mode, bool use_signs, bool use_orients, CeedVector u,
                                            CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(rstr, 3, 1, comp_stride, start, stop, t_mode, use_signs, use_orients, u, v, request);
}

static int CeedElemRestrictionApply_Ref_311(CeedElemRestriction rstr, const CeedInt num_comp, const CeedInt block_size, const CeedInt comp_stride,
                                            CeedInt start, CeedInt stop, CeedTransposeMode t_mode, bool use_signs, bool use_orients, CeedVector u,
                                            CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(rstr, 3, 1, 1, start, stop, t_mode, use_signs, use_orients, u, v, request);
}

static int CeedElemRestrictionApply_Ref_380(CeedElemRestriction rstr, const CeedInt num_comp, const CeedInt block_size, const CeedInt comp_stride,
                                            CeedInt start, CeedInt stop, CeedTransposeMode t_mode, bool use_signs, bool use_orients, CeedVector u,
                                            CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(rstr, 3, 8, comp_stride, start, stop, t_mode, use_signs, use_orients, u, v, request);
}

static int CeedElemRestrictionApply_Ref_381(CeedElemRestriction rstr, const CeedInt num_comp, const CeedInt block_size, const CeedInt comp_stride,
                                            CeedInt start, CeedInt stop, CeedTransposeMode t_mode, bool use_signs, bool use_orients, CeedVector u,
                                            CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(rstr, 3, 8, 1, start, stop, t_mode, use_signs, use_orients, u, v, request);
}

// LCOV_EXCL_START
static int CeedElemRestrictionApply_Ref_510(CeedElemRestriction rstr, const CeedInt num_comp, const CeedInt block_size, const CeedInt comp_stride,
                                            CeedInt start, CeedInt stop, CeedTransposeMode t_mode, bool use_signs, bool use_orients, CeedVector u,
                                            CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(rstr, 5, 1, comp_stride, start, stop, t_mode, use_signs, use_orients, u, v, request);
}
// LCOV_EXCL_STOP

static int CeedElemRestrictionApply_Ref_511(CeedElemRestriction rstr, const CeedInt num_comp, const CeedInt block_size, const CeedInt comp_stride,
                                            CeedInt start, CeedInt stop, CeedTransposeMode t_mode, bool use_signs, bool use_orients, CeedVector u,
                                            CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(rstr, 5, 1, 1, start, stop, t_mode, use_signs, use_orients, u, v, request);
}

// LCOV_EXCL_START
static int CeedElemRestrictionApply_Ref_580(CeedElemRestriction rstr, const CeedInt num_comp, const CeedInt block_size, const CeedInt comp_stride,
                                            CeedInt start, CeedInt stop, CeedTransposeMode t_mode, bool use_signs, bool use_orients, CeedVector u,
                                            CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(rstr, 5, 8, comp_stride, start, stop, t_mode, use_signs, use_orients, u, v, request);
}
// LCOV_EXCL_STOP

static int CeedElemRestrictionApply_Ref_581(CeedElemRestriction rstr, const CeedInt num_comp, const CeedInt block_size, const CeedInt comp_stride,
                                            CeedInt start, CeedInt stop, CeedTransposeMode t_mode, bool use_signs, bool use_orients, CeedVector u,
                                            CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(rstr, 5, 8, 1, start, stop, t_mode, use_signs, use_orients, u, v, request);
}

//------------------------------------------------------------------------------
// ElemRestriction Apply
//------------------------------------------------------------------------------
static int CeedElemRestrictionApply_Ref(CeedElemRestriction rstr, CeedTransposeMode t_mode, CeedVector u, CeedVector v, CeedRequest *request) {
  CeedInt                  num_block, block_size, num_comp, comp_stride;
  CeedElemRestriction_Ref *impl;

  CeedCallBackend(CeedElemRestrictionGetNumBlocks(rstr, &num_block));
  CeedCallBackend(CeedElemRestrictionGetBlockSize(rstr, &block_size));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(rstr, &num_comp));
  CeedCallBackend(CeedElemRestrictionGetCompStride(rstr, &comp_stride));
  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));
  CeedCallBackend(impl->Apply(rstr, num_comp, block_size, comp_stride, 0, num_block, t_mode, true, true, u, v, request));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// ElemRestriction Apply Unsigned
//------------------------------------------------------------------------------
static int CeedElemRestrictionApplyUnsigned_Ref(CeedElemRestriction rstr, CeedTransposeMode t_mode, CeedVector u, CeedVector v,
                                                CeedRequest *request) {
  CeedInt                  num_block, block_size, num_comp, comp_stride;
  CeedElemRestriction_Ref *impl;

  CeedCallBackend(CeedElemRestrictionGetNumBlocks(rstr, &num_block));
  CeedCallBackend(CeedElemRestrictionGetBlockSize(rstr, &block_size));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(rstr, &num_comp));
  CeedCallBackend(CeedElemRestrictionGetCompStride(rstr, &comp_stride));
  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));
  CeedCallBackend(impl->Apply(rstr, num_comp, block_size, comp_stride, 0, num_block, t_mode, false, true, u, v, request));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// ElemRestriction Apply Unoriented
//------------------------------------------------------------------------------
static int CeedElemRestrictionApplyUnoriented_Ref(CeedElemRestriction rstr, CeedTransposeMode t_mode, CeedVector u, CeedVector v,
                                                  CeedRequest *request) {
  CeedInt                  num_block, block_size, num_comp, comp_stride;
  CeedElemRestriction_Ref *impl;

  CeedCallBackend(CeedElemRestrictionGetNumBlocks(rstr, &num_block));
  CeedCallBackend(CeedElemRestrictionGetBlockSize(rstr, &block_size));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(rstr, &num_comp));
  CeedCallBackend(CeedElemRestrictionGetCompStride(rstr, &comp_stride));
  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));
  CeedCallBackend(impl->Apply(rstr, num_comp, block_size, comp_stride, 0, num_block, t_mode, false, false, u, v, request));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// ElemRestriction Apply Points
//------------------------------------------------------------------------------
static int CeedElemRestrictionApplyAtPointsInElement_Ref(CeedElemRestriction rstr, CeedInt elem, CeedTransposeMode t_mode, CeedVector u, CeedVector v,
                                                         CeedRequest *request) {
  CeedInt                  num_comp;
  CeedElemRestriction_Ref *impl;

  CeedCallBackend(CeedElemRestrictionGetNumComponents(rstr, &num_comp));
  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));
  return impl->Apply(rstr, num_comp, 0, 1, elem, elem + 1, t_mode, false, false, u, v, request);
}

//------------------------------------------------------------------------------
// ElemRestriction Apply Block
//------------------------------------------------------------------------------
static int CeedElemRestrictionApplyBlock_Ref(CeedElemRestriction rstr, CeedInt block, CeedTransposeMode t_mode, CeedVector u, CeedVector v,
                                             CeedRequest *request) {
  CeedInt                  block_size, num_comp, comp_stride;
  CeedElemRestriction_Ref *impl;

  CeedCallBackend(CeedElemRestrictionGetBlockSize(rstr, &block_size));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(rstr, &num_comp));
  CeedCallBackend(CeedElemRestrictionGetCompStride(rstr, &comp_stride));
  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));
  CeedCallBackend(impl->Apply(rstr, num_comp, block_size, comp_stride, block, block + 1, t_mode, true, true, u, v, request));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// ElemRestriction Get Offsets
//------------------------------------------------------------------------------
static int CeedElemRestrictionGetOffsets_Ref(CeedElemRestriction rstr, CeedMemType mem_type, const CeedInt **offsets) {
  CeedElemRestriction_Ref *impl;

  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));

  CeedCheck(mem_type == CEED_MEM_HOST, CeedElemRestrictionReturnCeed(rstr), CEED_ERROR_BACKEND, "Can only provide to HOST memory");

  *offsets = impl->offsets;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// ElemRestriction Get Orientations
//------------------------------------------------------------------------------
static int CeedElemRestrictionGetOrientations_Ref(CeedElemRestriction rstr, CeedMemType mem_type, const bool **orients) {
  CeedElemRestriction_Ref *impl;

  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));

  CeedCheck(mem_type == CEED_MEM_HOST, CeedElemRestrictionReturnCeed(rstr), CEED_ERROR_BACKEND, "Can only provide to HOST memory");

  *orients = impl->orients;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// ElemRestriction Get Curl-Conforming Orientations
//------------------------------------------------------------------------------
static int CeedElemRestrictionGetCurlOrientations_Ref(CeedElemRestriction rstr, CeedMemType mem_type, const CeedInt8 **curl_orients) {
  CeedElemRestriction_Ref *impl;

  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));

  CeedCheck(mem_type == CEED_MEM_HOST, CeedElemRestrictionReturnCeed(rstr), CEED_ERROR_BACKEND, "Can only provide to HOST memory");

  *curl_orients = impl->curl_orients;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// ElemRestriction Destroy
//------------------------------------------------------------------------------
static int CeedElemRestrictionDestroy_Ref(CeedElemRestriction rstr) {
  CeedElemRestriction_Ref *impl;

  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));
  CeedCallBackend(CeedFree(&impl->offsets_owned));
  CeedCallBackend(CeedFree(&impl->orients_owned));
  CeedCallBackend(CeedFree(&impl->curl_orients_owned));
  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// ElemRestriction Create
//------------------------------------------------------------------------------
int CeedElemRestrictionCreate_Ref(CeedMemType mem_type, CeedCopyMode copy_mode, const CeedInt *offsets, const bool *orients,
                                  const CeedInt8 *curl_orients, CeedElemRestriction rstr) {
  Ceed                     ceed;
  CeedInt                  num_elem, elem_size, num_block, block_size, num_comp, comp_stride, num_points = 0, num_offsets;
  CeedRestrictionType      rstr_type;
  CeedElemRestriction_Ref *impl;

  CeedCallBackend(CeedElemRestrictionGetCeed(rstr, &ceed));
  CeedCallBackend(CeedElemRestrictionGetNumElements(rstr, &num_elem));
  CeedCallBackend(CeedElemRestrictionGetElementSize(rstr, &elem_size));
  CeedCallBackend(CeedElemRestrictionGetNumBlocks(rstr, &num_block));
  CeedCallBackend(CeedElemRestrictionGetBlockSize(rstr, &block_size));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(rstr, &num_comp));
  CeedCallBackend(CeedElemRestrictionGetCompStride(rstr, &comp_stride));
  CeedCallBackend(CeedElemRestrictionGetType(rstr, &rstr_type));

  CeedCheck(mem_type == CEED_MEM_HOST, ceed, CEED_ERROR_BACKEND, "Only MemType = HOST supported");

  CeedCallBackend(CeedCalloc(1, &impl));
  CeedCallBackend(CeedElemRestrictionSetData(rstr, impl));

  // Set layouts
  {
    bool    has_backend_strides;
    CeedInt layout[3] = {1, elem_size, elem_size * num_comp};

    CeedCallBackend(CeedElemRestrictionSetELayout(rstr, layout));
    if (rstr_type == CEED_RESTRICTION_STRIDED) {
      CeedCallBackend(CeedElemRestrictionHasBackendStrides(rstr, &has_backend_strides));
      if (has_backend_strides) {
        CeedCallBackend(CeedElemRestrictionSetLLayout(rstr, layout));
      }
    }
  }

  // Offsets data
  if (rstr_type != CEED_RESTRICTION_STRIDED) {
    const char *resource;

    // Check indices for ref or memcheck backends
    {
      Ceed current = ceed, parent = NULL;

      CeedCallBackend(CeedGetParent(current, &parent));
      while (current != parent) {
        current = parent;
        CeedCallBackend(CeedGetParent(current, &parent));
      }
      CeedCallBackend(CeedGetResource(parent, &resource));
    }
    if (!strcmp(resource, "/cpu/self/ref/serial") || !strcmp(resource, "/cpu/self/ref/blocked")) {
      CeedSize l_size;

      CeedCallBackend(CeedElemRestrictionGetLVectorSize(rstr, &l_size));
      for (CeedInt i = 0; i < num_elem * elem_size; i++) {
        CeedCheck(offsets[i] >= 0 && offsets[i] + (num_comp - 1) * comp_stride < l_size, ceed, CEED_ERROR_BACKEND,
                  "Restriction offset %" CeedInt_FMT " (%" CeedInt_FMT ") out of range [0, %" CeedInt_FMT "]", i, offsets[i], l_size);
      }
    }

    // Copy data
    if (rstr_type == CEED_RESTRICTION_POINTS) CeedCallBackend(CeedElemRestrictionGetNumPoints(rstr, &num_points));
    num_offsets = rstr_type == CEED_RESTRICTION_POINTS ? (num_elem + 1 + num_points) : (num_elem * elem_size);
    CeedCallBackend(CeedSetHostCeedIntArray(offsets, copy_mode, num_offsets, &impl->offsets_owned, &impl->offsets_borrowed, &impl->offsets));

    // Orientation data
    if (rstr_type == CEED_RESTRICTION_ORIENTED) {
      CeedCheck(orients != NULL, ceed, CEED_ERROR_BACKEND, "No orients array provided for oriented restriction");
      CeedCallBackend(CeedSetHostBoolArray(orients, copy_mode, num_offsets, &impl->orients_owned, &impl->orients_borrowed, &impl->orients));
    } else if (rstr_type == CEED_RESTRICTION_CURL_ORIENTED) {
      CeedCheck(curl_orients != NULL, ceed, CEED_ERROR_BACKEND, "No curl_orients array provided for oriented restriction");
      CeedCallBackend(CeedSetHostCeedInt8Array(curl_orients, copy_mode, 3 * num_offsets, &impl->curl_orients_owned, &impl->curl_orients_borrowed,
                                               &impl->curl_orients));
    }
  }

  // Set apply function based upon num_comp, block_size, and comp_stride
  CeedInt index = -1;

  if (block_size < 10) index = 100 * num_comp + 10 * block_size + (comp_stride == 1);
  switch (index) {
    case 110:
      impl->Apply = CeedElemRestrictionApply_Ref_110;
      break;
    case 111:
      impl->Apply = CeedElemRestrictionApply_Ref_111;
      break;
    case 180:
      impl->Apply = CeedElemRestrictionApply_Ref_180;
      break;
    case 181:
      impl->Apply = CeedElemRestrictionApply_Ref_181;
      break;
    case 310:
      impl->Apply = CeedElemRestrictionApply_Ref_310;
      break;
    case 311:
      impl->Apply = CeedElemRestrictionApply_Ref_311;
      break;
    case 380:
      impl->Apply = CeedElemRestrictionApply_Ref_380;
      break;
    case 381:
      impl->Apply = CeedElemRestrictionApply_Ref_381;
      break;
    // LCOV_EXCL_START
    case 510:
      impl->Apply = CeedElemRestrictionApply_Ref_510;
      break;
    // LCOV_EXCL_STOP
    case 511:
      impl->Apply = CeedElemRestrictionApply_Ref_511;
      break;
    // LCOV_EXCL_START
    case 580:
      impl->Apply = CeedElemRestrictionApply_Ref_580;
      break;
    // LCOV_EXCL_STOP
    case 581:
      impl->Apply = CeedElemRestrictionApply_Ref_581;
      break;
    default:
      impl->Apply = CeedElemRestrictionApply_Ref_Core;
      break;
  }

  // Register backend functions
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "Apply", CeedElemRestrictionApply_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "ApplyUnsigned", CeedElemRestrictionApplyUnsigned_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "ApplyUnoriented", CeedElemRestrictionApplyUnoriented_Ref));
  if (rstr_type == CEED_RESTRICTION_POINTS) {
    CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "ApplyAtPointsInElement", CeedElemRestrictionApplyAtPointsInElement_Ref));
  }
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "ApplyBlock", CeedElemRestrictionApplyBlock_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "GetOffsets", CeedElemRestrictionGetOffsets_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "GetOrientations", CeedElemRestrictionGetOrientations_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "GetCurlOrientations", CeedElemRestrictionGetCurlOrientations_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", rstr, "Destroy", CeedElemRestrictionDestroy_Ref));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
