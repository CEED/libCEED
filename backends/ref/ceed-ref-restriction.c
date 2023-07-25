// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
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
static inline int CeedElemRestrictionApplyStridedNoTranspose_Ref_Core(CeedElemRestriction r, const CeedInt num_comp, const CeedInt blk_size,
                                                                      CeedInt start, CeedInt stop, CeedInt num_elem, CeedInt elem_size,
                                                                      CeedInt v_offset, const CeedScalar *uu, CeedScalar *vv) {
  // No offsets provided, identity restriction
  bool has_backend_strides;
  CeedCallBackend(CeedElemRestrictionHasBackendStrides(r, &has_backend_strides));
  if (has_backend_strides) {
    // CPU backend strides are {1, elem_size, elem_size*num_comp}
    // This if branch is left separate to allow better inlining
    for (CeedInt e = start * blk_size; e < stop * blk_size; e += blk_size) {
      CeedPragmaSIMD for (CeedInt k = 0; k < num_comp; k++) {
        CeedPragmaSIMD for (CeedInt n = 0; n < elem_size; n++) {
          CeedPragmaSIMD for (CeedInt j = 0; j < blk_size; j++) {
            vv[e * elem_size * num_comp + (k * elem_size + n) * blk_size + j - v_offset] =
                uu[n + k * elem_size + CeedIntMin(e + j, num_elem - 1) * elem_size * num_comp];
          }
        }
      }
    }
  } else {
    // User provided strides
    CeedInt strides[3];
    CeedCallBackend(CeedElemRestrictionGetStrides(r, &strides));
    for (CeedInt e = start * blk_size; e < stop * blk_size; e += blk_size) {
      CeedPragmaSIMD for (CeedInt k = 0; k < num_comp; k++) {
        CeedPragmaSIMD for (CeedInt n = 0; n < elem_size; n++) {
          CeedPragmaSIMD for (CeedInt j = 0; j < blk_size; j++) {
            vv[e * elem_size * num_comp + (k * elem_size + n) * blk_size + j - v_offset] =
                uu[n * strides[0] + k * strides[1] + CeedIntMin(e + j, num_elem - 1) * strides[2]];
          }
        }
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

static inline int CeedElemRestrictionApplyStandardNoTranspose_Ref_Core(CeedElemRestriction r, const CeedInt num_comp, const CeedInt blk_size,
                                                                       const CeedInt comp_stride, CeedInt start, CeedInt stop, CeedInt num_elem,
                                                                       CeedInt elem_size, CeedInt v_offset, const CeedScalar *uu, CeedScalar *vv) {
  // Default restriction with offsets
  CeedElemRestriction_Ref *impl;
  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));
  for (CeedInt e = start * blk_size; e < stop * blk_size; e += blk_size) {
    CeedPragmaSIMD for (CeedInt k = 0; k < num_comp; k++) {
      CeedPragmaSIMD for (CeedInt i = 0; i < elem_size * blk_size; i++) {
        vv[elem_size * (k * blk_size + e * num_comp) + i - v_offset] = uu[impl->offsets[i + e * elem_size] + k * comp_stride];
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

static inline int CeedElemRestrictionApplyOrientedNoTranspose_Ref_Core(CeedElemRestriction r, const CeedInt num_comp, const CeedInt blk_size,
                                                                       const CeedInt comp_stride, CeedInt start, CeedInt stop, CeedInt num_elem,
                                                                       CeedInt elem_size, CeedInt v_offset, const CeedScalar *uu, CeedScalar *vv) {
  // Restriction with orientations
  CeedElemRestriction_Ref *impl;
  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));
  for (CeedInt e = start * blk_size; e < stop * blk_size; e += blk_size) {
    CeedPragmaSIMD for (CeedInt k = 0; k < num_comp; k++) {
      CeedPragmaSIMD for (CeedInt i = 0; i < elem_size * blk_size; i++) {
        vv[elem_size * (k * blk_size + e * num_comp) + i - v_offset] =
            uu[impl->offsets[i + e * elem_size] + k * comp_stride] * (impl->orients[i + e * elem_size] ? -1.0 : 1.0);
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

static inline int CeedElemRestrictionApplyCurlOrientedNoTranspose_Ref_Core(CeedElemRestriction r, const CeedInt num_comp, const CeedInt blk_size,
                                                                           const CeedInt comp_stride, CeedInt start, CeedInt stop, CeedInt num_elem,
                                                                           CeedInt elem_size, CeedInt v_offset, const CeedScalar *uu,
                                                                           CeedScalar *vv) {
  // Restriction with tridiagonal transformation
  CeedElemRestriction_Ref *impl;
  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));
  for (CeedInt e = start * blk_size; e < stop * blk_size; e += blk_size) {
    CeedPragmaSIMD for (CeedInt k = 0; k < num_comp; k++) {
      CeedInt n = 0;
      CeedPragmaSIMD for (CeedInt j = 0; j < blk_size; j++) {
        vv[e * elem_size * num_comp + (k * elem_size + n) * blk_size + j - v_offset] =
            uu[impl->offsets[j + n * blk_size + e * elem_size] + k * comp_stride] *
                impl->curl_orients[j + (3 * n + 1) * blk_size + e * 3 * elem_size] +
            uu[impl->offsets[j + (n + 1) * blk_size + e * elem_size] + k * comp_stride] *
                impl->curl_orients[j + (3 * n + 2) * blk_size + e * 3 * elem_size];
      }
      for (n = 1; n < elem_size - 1; n++) {
        CeedPragmaSIMD for (CeedInt j = 0; j < blk_size; j++) {
          vv[e * elem_size * num_comp + (k * elem_size + n) * blk_size + j - v_offset] =
              uu[impl->offsets[j + (n - 1) * blk_size + e * elem_size] + k * comp_stride] *
                  impl->curl_orients[j + (3 * n + 0) * blk_size + e * 3 * elem_size] +
              uu[impl->offsets[j + n * blk_size + e * elem_size] + k * comp_stride] *
                  impl->curl_orients[j + (3 * n + 1) * blk_size + e * 3 * elem_size] +
              uu[impl->offsets[j + (n + 1) * blk_size + e * elem_size] + k * comp_stride] *
                  impl->curl_orients[j + (3 * n + 2) * blk_size + e * 3 * elem_size];
        }
      }
      CeedPragmaSIMD for (CeedInt j = 0; j < blk_size; j++) {
        vv[e * elem_size * num_comp + (k * elem_size + n) * blk_size + j - v_offset] =
            uu[impl->offsets[j + (n - 1) * blk_size + e * elem_size] + k * comp_stride] *
                impl->curl_orients[j + (3 * n + 0) * blk_size + e * 3 * elem_size] +
            uu[impl->offsets[j + n * blk_size + e * elem_size] + k * comp_stride] *
                impl->curl_orients[j + (3 * n + 1) * blk_size + e * 3 * elem_size];
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

static inline int CeedElemRestrictionApplyCurlOrientedUnsignedNoTranspose_Ref_Core(CeedElemRestriction r, const CeedInt num_comp,
                                                                                   const CeedInt blk_size, const CeedInt comp_stride, CeedInt start,
                                                                                   CeedInt stop, CeedInt num_elem, CeedInt elem_size,
                                                                                   CeedInt v_offset, const CeedScalar *uu, CeedScalar *vv) {
  // Restriction with (unsigned) tridiagonal transformation
  CeedElemRestriction_Ref *impl;
  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));
  for (CeedInt e = start * blk_size; e < stop * blk_size; e += blk_size) {
    CeedPragmaSIMD for (CeedInt k = 0; k < num_comp; k++) {
      CeedInt n = 0;
      CeedPragmaSIMD for (CeedInt j = 0; j < blk_size; j++) {
        vv[e * elem_size * num_comp + (k * elem_size + n) * blk_size + j - v_offset] =
            uu[impl->offsets[j + n * blk_size + e * elem_size] + k * comp_stride] *
                abs(impl->curl_orients[j + (3 * n + 1) * blk_size + e * 3 * elem_size]) +
            uu[impl->offsets[j + (n + 1) * blk_size + e * elem_size] + k * comp_stride] *
                abs(impl->curl_orients[j + (3 * n + 2) * blk_size + e * 3 * elem_size]);
      }
      for (n = 1; n < elem_size - 1; n++) {
        CeedPragmaSIMD for (CeedInt j = 0; j < blk_size; j++) {
          vv[e * elem_size * num_comp + (k * elem_size + n) * blk_size + j - v_offset] =
              uu[impl->offsets[j + (n - 1) * blk_size + e * elem_size] + k * comp_stride] *
                  abs(impl->curl_orients[j + (3 * n + 0) * blk_size + e * 3 * elem_size]) +
              uu[impl->offsets[j + n * blk_size + e * elem_size] + k * comp_stride] *
                  abs(impl->curl_orients[j + (3 * n + 1) * blk_size + e * 3 * elem_size]) +
              uu[impl->offsets[j + (n + 1) * blk_size + e * elem_size] + k * comp_stride] *
                  abs(impl->curl_orients[j + (3 * n + 2) * blk_size + e * 3 * elem_size]);
        }
      }
      CeedPragmaSIMD for (CeedInt j = 0; j < blk_size; j++) {
        vv[e * elem_size * num_comp + (k * elem_size + n) * blk_size + j - v_offset] =
            uu[impl->offsets[j + (n - 1) * blk_size + e * elem_size] + k * comp_stride] *
                abs(impl->curl_orients[j + (3 * n + 0) * blk_size + e * 3 * elem_size]) +
            uu[impl->offsets[j + n * blk_size + e * elem_size] + k * comp_stride] *
                abs(impl->curl_orients[j + (3 * n + 1) * blk_size + e * 3 * elem_size]);
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

static inline int CeedElemRestrictionApplyStridedTranspose_Ref_Core(CeedElemRestriction r, const CeedInt num_comp, const CeedInt blk_size,
                                                                    CeedInt start, CeedInt stop, CeedInt num_elem, CeedInt elem_size,
                                                                    CeedInt v_offset, const CeedScalar *uu, CeedScalar *vv) {
  // No offsets provided, identity restriction
  bool has_backend_strides;
  CeedCallBackend(CeedElemRestrictionHasBackendStrides(r, &has_backend_strides));
  if (has_backend_strides) {
    // CPU backend strides are {1, elem_size, elem_size*num_comp}
    // This if brach is left separate to allow better inlining
    for (CeedInt e = start * blk_size; e < stop * blk_size; e += blk_size) {
      CeedPragmaSIMD for (CeedInt k = 0; k < num_comp; k++) {
        CeedPragmaSIMD for (CeedInt n = 0; n < elem_size; n++) {
          CeedPragmaSIMD for (CeedInt j = 0; j < CeedIntMin(blk_size, num_elem - e); j++) {
            vv[n + k * elem_size + (e + j) * elem_size * num_comp] += uu[e * elem_size * num_comp + (k * elem_size + n) * blk_size + j - v_offset];
          }
        }
      }
    }
  } else {
    // User provided strides
    CeedInt strides[3];
    CeedCallBackend(CeedElemRestrictionGetStrides(r, &strides));
    for (CeedInt e = start * blk_size; e < stop * blk_size; e += blk_size) {
      CeedPragmaSIMD for (CeedInt k = 0; k < num_comp; k++) {
        CeedPragmaSIMD for (CeedInt n = 0; n < elem_size; n++) {
          CeedPragmaSIMD for (CeedInt j = 0; j < CeedIntMin(blk_size, num_elem - e); j++) {
            vv[n * strides[0] + k * strides[1] + (e + j) * strides[2]] +=
                uu[e * elem_size * num_comp + (k * elem_size + n) * blk_size + j - v_offset];
          }
        }
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

static inline int CeedElemRestrictionApplyStandardTranspose_Ref_Core(CeedElemRestriction r, const CeedInt num_comp, const CeedInt blk_size,
                                                                     const CeedInt comp_stride, CeedInt start, CeedInt stop, CeedInt num_elem,
                                                                     CeedInt elem_size, CeedInt v_offset, const CeedScalar *uu, CeedScalar *vv) {
  // Default restriction with offsets
  CeedElemRestriction_Ref *impl;
  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));
  for (CeedInt e = start * blk_size; e < stop * blk_size; e += blk_size) {
    for (CeedInt k = 0; k < num_comp; k++) {
      for (CeedInt i = 0; i < elem_size * blk_size; i += blk_size) {
        // Iteration bound set to discard padding elements
        for (CeedInt j = i; j < i + CeedIntMin(blk_size, num_elem - e); j++) {
          vv[impl->offsets[j + e * elem_size] + k * comp_stride] += uu[elem_size * (k * blk_size + e * num_comp) + j - v_offset];
        }
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

static inline int CeedElemRestrictionApplyOrientedTranspose_Ref_Core(CeedElemRestriction r, const CeedInt num_comp, const CeedInt blk_size,
                                                                     const CeedInt comp_stride, CeedInt start, CeedInt stop, CeedInt num_elem,
                                                                     CeedInt elem_size, CeedInt v_offset, const CeedScalar *uu, CeedScalar *vv) {
  // Restriction with orientations
  CeedElemRestriction_Ref *impl;
  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));
  for (CeedInt e = start * blk_size; e < stop * blk_size; e += blk_size) {
    for (CeedInt k = 0; k < num_comp; k++) {
      for (CeedInt i = 0; i < elem_size * blk_size; i += blk_size) {
        // Iteration bound set to discard padding elements
        for (CeedInt j = i; j < i + CeedIntMin(blk_size, num_elem - e); j++) {
          vv[impl->offsets[j + e * elem_size] + k * comp_stride] +=
              uu[elem_size * (k * blk_size + e * num_comp) + j - v_offset] * (impl->orients[j + e * elem_size] ? -1.0 : 1.0);
        }
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

static inline int CeedElemRestrictionApplyCurlOrientedTranspose_Ref_Core(CeedElemRestriction r, const CeedInt num_comp, const CeedInt blk_size,
                                                                         const CeedInt comp_stride, CeedInt start, CeedInt stop, CeedInt num_elem,
                                                                         CeedInt elem_size, CeedInt v_offset, const CeedScalar *uu, CeedScalar *vv) {
  // Restriction with tridiagonal transformation
  CeedElemRestriction_Ref *impl;
  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));
  for (CeedInt e = start * blk_size; e < stop * blk_size; e += blk_size) {
    for (CeedInt k = 0; k < num_comp; k++) {
      // Iteration bound set to discard padding elements
      CeedInt blk_end = CeedIntMin(blk_size, num_elem - e), n = 0;
      for (CeedInt j = 0; j < blk_end; j++) {
        vv[impl->offsets[j + n * blk_size + e * elem_size] + k * comp_stride] +=
            uu[e * elem_size * num_comp + (k * elem_size + n) * blk_size + j - v_offset] *
                impl->curl_orients[j + (3 * n + 1) * blk_size + e * 3 * elem_size] +
            uu[e * elem_size * num_comp + (k * elem_size + n + 1) * blk_size + j - v_offset] *
                impl->curl_orients[j + (3 * n + 3) * blk_size + e * 3 * elem_size];
      }
      for (n = 1; n < elem_size - 1; n++) {
        for (CeedInt j = 0; j < blk_end; j++) {
          vv[impl->offsets[j + n * blk_size + e * elem_size] + k * comp_stride] +=
              uu[e * elem_size * num_comp + (k * elem_size + n - 1) * blk_size + j - v_offset] *
                  impl->curl_orients[j + (3 * n - 1) * blk_size + e * 3 * elem_size] +
              uu[e * elem_size * num_comp + (k * elem_size + n) * blk_size + j - v_offset] *
                  impl->curl_orients[j + (3 * n + 1) * blk_size + e * 3 * elem_size] +
              uu[e * elem_size * num_comp + (k * elem_size + n + 1) * blk_size + j - v_offset] *
                  impl->curl_orients[j + (3 * n + 3) * blk_size + e * 3 * elem_size];
        }
      }
      for (CeedInt j = 0; j < blk_end; j++) {
        vv[impl->offsets[j + n * blk_size + e * elem_size] + k * comp_stride] +=
            uu[e * elem_size * num_comp + (k * elem_size + n - 1) * blk_size + j - v_offset] *
                impl->curl_orients[j + (3 * n - 1) * blk_size + e * 3 * elem_size] +
            uu[e * elem_size * num_comp + (k * elem_size + n) * blk_size + j - v_offset] *
                impl->curl_orients[j + (3 * n + 1) * blk_size + e * 3 * elem_size];
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

static inline int CeedElemRestrictionApplyCurlOrientedUnsignedTranspose_Ref_Core(CeedElemRestriction r, const CeedInt num_comp,
                                                                                 const CeedInt blk_size, const CeedInt comp_stride, CeedInt start,
                                                                                 CeedInt stop, CeedInt num_elem, CeedInt elem_size, CeedInt v_offset,
                                                                                 const CeedScalar *uu, CeedScalar *vv) {
  // Restriction with (unsigned) tridiagonal transformation
  CeedElemRestriction_Ref *impl;
  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));
  for (CeedInt e = start * blk_size; e < stop * blk_size; e += blk_size) {
    for (CeedInt k = 0; k < num_comp; k++) {
      // Iteration bound set to discard padding elements
      CeedInt blk_end = CeedIntMin(blk_size, num_elem - e), n = 0;
      for (CeedInt j = 0; j < blk_end; j++) {
        vv[impl->offsets[j + n * blk_size + e * elem_size] + k * comp_stride] +=
            uu[e * elem_size * num_comp + (k * elem_size + n) * blk_size + j - v_offset] *
                abs(impl->curl_orients[j + (3 * n + 1) * blk_size + e * 3 * elem_size]) +
            uu[e * elem_size * num_comp + (k * elem_size + n + 1) * blk_size + j - v_offset] *
                abs(impl->curl_orients[j + (3 * n + 3) * blk_size + e * 3 * elem_size]);
      }
      for (n = 1; n < elem_size - 1; n++) {
        for (CeedInt j = 0; j < blk_end; j++) {
          vv[impl->offsets[j + n * blk_size + e * elem_size] + k * comp_stride] +=
              uu[e * elem_size * num_comp + (k * elem_size + n - 1) * blk_size + j - v_offset] *
                  abs(impl->curl_orients[j + (3 * n - 1) * blk_size + e * 3 * elem_size]) +
              uu[e * elem_size * num_comp + (k * elem_size + n) * blk_size + j - v_offset] *
                  abs(impl->curl_orients[j + (3 * n + 1) * blk_size + e * 3 * elem_size]) +
              uu[e * elem_size * num_comp + (k * elem_size + n + 1) * blk_size + j - v_offset] *
                  abs(impl->curl_orients[j + (3 * n + 3) * blk_size + e * 3 * elem_size]);
        }
      }
      for (CeedInt j = 0; j < blk_end; j++) {
        vv[impl->offsets[j + n * blk_size + e * elem_size] + k * comp_stride] +=
            uu[e * elem_size * num_comp + (k * elem_size + n - 1) * blk_size + j - v_offset] *
                abs(impl->curl_orients[j + (3 * n - 1) * blk_size + e * 3 * elem_size]) +
            uu[e * elem_size * num_comp + (k * elem_size + n) * blk_size + j - v_offset] *
                abs(impl->curl_orients[j + (3 * n + 1) * blk_size + e * 3 * elem_size]);
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

static inline int CeedElemRestrictionApply_Ref_Core(CeedElemRestriction r, const CeedInt num_comp, const CeedInt blk_size, const CeedInt comp_stride,
                                                    CeedInt start, CeedInt stop, CeedTransposeMode t_mode, bool use_signs, bool use_orients,
                                                    CeedVector u, CeedVector v, CeedRequest *request) {
  const CeedScalar *uu;
  CeedScalar       *vv;
  CeedInt           num_elem, elem_size, v_offset;
  CeedCallBackend(CeedElemRestrictionGetNumElements(r, &num_elem));
  CeedCallBackend(CeedElemRestrictionGetElementSize(r, &elem_size));
  v_offset = start * blk_size * elem_size * num_comp;
  CeedRestrictionType rstr_type;
  CeedCallBackend(CeedElemRestrictionGetType(r, &rstr_type));

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
        CeedElemRestrictionApplyStridedTranspose_Ref_Core(r, num_comp, blk_size, start, stop, num_elem, elem_size, v_offset, uu, vv);
        break;
      case CEED_RESTRICTION_STANDARD:
        CeedElemRestrictionApplyStandardTranspose_Ref_Core(r, num_comp, blk_size, comp_stride, start, stop, num_elem, elem_size, v_offset, uu, vv);
        break;
      case CEED_RESTRICTION_ORIENTED:
        if (use_signs) {
          CeedElemRestrictionApplyOrientedTranspose_Ref_Core(r, num_comp, blk_size, comp_stride, start, stop, num_elem, elem_size, v_offset, uu, vv);
        } else {
          CeedElemRestrictionApplyStandardTranspose_Ref_Core(r, num_comp, blk_size, comp_stride, start, stop, num_elem, elem_size, v_offset, uu, vv);
        }
        break;
      case CEED_RESTRICTION_CURL_ORIENTED:
        if (use_signs && use_orients) {
          CeedElemRestrictionApplyCurlOrientedTranspose_Ref_Core(r, num_comp, blk_size, comp_stride, start, stop, num_elem, elem_size, v_offset, uu,
                                                                 vv);
        } else if (use_orients) {
          CeedElemRestrictionApplyCurlOrientedUnsignedTranspose_Ref_Core(r, num_comp, blk_size, comp_stride, start, stop, num_elem, elem_size,
                                                                         v_offset, uu, vv);
        } else {
          CeedElemRestrictionApplyStandardTranspose_Ref_Core(r, num_comp, blk_size, comp_stride, start, stop, num_elem, elem_size, v_offset, uu, vv);
        }
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
        CeedElemRestrictionApplyStridedNoTranspose_Ref_Core(r, num_comp, blk_size, start, stop, num_elem, elem_size, v_offset, uu, vv);
        break;
      case CEED_RESTRICTION_STANDARD:
        CeedElemRestrictionApplyStandardNoTranspose_Ref_Core(r, num_comp, blk_size, comp_stride, start, stop, num_elem, elem_size, v_offset, uu, vv);
        break;
      case CEED_RESTRICTION_ORIENTED:
        if (use_signs) {
          CeedElemRestrictionApplyOrientedNoTranspose_Ref_Core(r, num_comp, blk_size, comp_stride, start, stop, num_elem, elem_size, v_offset, uu,
                                                               vv);
        } else {
          CeedElemRestrictionApplyStandardNoTranspose_Ref_Core(r, num_comp, blk_size, comp_stride, start, stop, num_elem, elem_size, v_offset, uu,
                                                               vv);
        }
        break;
      case CEED_RESTRICTION_CURL_ORIENTED:
        if (use_signs && use_orients) {
          CeedElemRestrictionApplyCurlOrientedNoTranspose_Ref_Core(r, num_comp, blk_size, comp_stride, start, stop, num_elem, elem_size, v_offset, uu,
                                                                   vv);
        } else if (use_orients) {
          CeedElemRestrictionApplyCurlOrientedUnsignedNoTranspose_Ref_Core(r, num_comp, blk_size, comp_stride, start, stop, num_elem, elem_size,
                                                                           v_offset, uu, vv);
        } else {
          CeedElemRestrictionApplyStandardNoTranspose_Ref_Core(r, num_comp, blk_size, comp_stride, start, stop, num_elem, elem_size, v_offset, uu,
                                                               vv);
        }
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
static int CeedElemRestrictionApply_Ref_110(CeedElemRestriction r, const CeedInt num_comp, const CeedInt blk_size, const CeedInt comp_stride,
                                            CeedInt start, CeedInt stop, CeedTransposeMode t_mode, bool use_signs, bool use_orients, CeedVector u,
                                            CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 1, 1, comp_stride, start, stop, t_mode, use_signs, use_orients, u, v, request);
}

static int CeedElemRestrictionApply_Ref_111(CeedElemRestriction r, const CeedInt num_comp, const CeedInt blk_size, const CeedInt comp_stride,
                                            CeedInt start, CeedInt stop, CeedTransposeMode t_mode, bool use_signs, bool use_orients, CeedVector u,
                                            CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 1, 1, 1, start, stop, t_mode, use_signs, use_orients, u, v, request);
}

static int CeedElemRestrictionApply_Ref_180(CeedElemRestriction r, const CeedInt num_comp, const CeedInt blk_size, const CeedInt comp_stride,
                                            CeedInt start, CeedInt stop, CeedTransposeMode t_mode, bool use_signs, bool use_orients, CeedVector u,
                                            CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 1, 8, comp_stride, start, stop, t_mode, use_signs, use_orients, u, v, request);
}

static int CeedElemRestrictionApply_Ref_181(CeedElemRestriction r, const CeedInt num_comp, const CeedInt blk_size, const CeedInt comp_stride,
                                            CeedInt start, CeedInt stop, CeedTransposeMode t_mode, bool use_signs, bool use_orients, CeedVector u,
                                            CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 1, 8, 1, start, stop, t_mode, use_signs, use_orients, u, v, request);
}

static int CeedElemRestrictionApply_Ref_310(CeedElemRestriction r, const CeedInt num_comp, const CeedInt blk_size, const CeedInt comp_stride,
                                            CeedInt start, CeedInt stop, CeedTransposeMode t_mode, bool use_signs, bool use_orients, CeedVector u,
                                            CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 3, 1, comp_stride, start, stop, t_mode, use_signs, use_orients, u, v, request);
}

static int CeedElemRestrictionApply_Ref_311(CeedElemRestriction r, const CeedInt num_comp, const CeedInt blk_size, const CeedInt comp_stride,
                                            CeedInt start, CeedInt stop, CeedTransposeMode t_mode, bool use_signs, bool use_orients, CeedVector u,
                                            CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 3, 1, 1, start, stop, t_mode, use_signs, use_orients, u, v, request);
}

static int CeedElemRestrictionApply_Ref_380(CeedElemRestriction r, const CeedInt num_comp, const CeedInt blk_size, const CeedInt comp_stride,
                                            CeedInt start, CeedInt stop, CeedTransposeMode t_mode, bool use_signs, bool use_orients, CeedVector u,
                                            CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 3, 8, comp_stride, start, stop, t_mode, use_signs, use_orients, u, v, request);
}

static int CeedElemRestrictionApply_Ref_381(CeedElemRestriction r, const CeedInt num_comp, const CeedInt blk_size, const CeedInt comp_stride,
                                            CeedInt start, CeedInt stop, CeedTransposeMode t_mode, bool use_signs, bool use_orients, CeedVector u,
                                            CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 3, 8, 1, start, stop, t_mode, use_signs, use_orients, u, v, request);
}

// LCOV_EXCL_START
static int CeedElemRestrictionApply_Ref_510(CeedElemRestriction r, const CeedInt num_comp, const CeedInt blk_size, const CeedInt comp_stride,
                                            CeedInt start, CeedInt stop, CeedTransposeMode t_mode, bool use_signs, bool use_orients, CeedVector u,
                                            CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 5, 1, comp_stride, start, stop, t_mode, use_signs, use_orients, u, v, request);
}
// LCOV_EXCL_STOP

static int CeedElemRestrictionApply_Ref_511(CeedElemRestriction r, const CeedInt num_comp, const CeedInt blk_size, const CeedInt comp_stride,
                                            CeedInt start, CeedInt stop, CeedTransposeMode t_mode, bool use_signs, bool use_orients, CeedVector u,
                                            CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 5, 1, 1, start, stop, t_mode, use_signs, use_orients, u, v, request);
}

// LCOV_EXCL_START
static int CeedElemRestrictionApply_Ref_580(CeedElemRestriction r, const CeedInt num_comp, const CeedInt blk_size, const CeedInt comp_stride,
                                            CeedInt start, CeedInt stop, CeedTransposeMode t_mode, bool use_signs, bool use_orients, CeedVector u,
                                            CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 5, 8, comp_stride, start, stop, t_mode, use_signs, use_orients, u, v, request);
}
// LCOV_EXCL_STOP

static int CeedElemRestrictionApply_Ref_581(CeedElemRestriction r, const CeedInt num_comp, const CeedInt blk_size, const CeedInt comp_stride,
                                            CeedInt start, CeedInt stop, CeedTransposeMode t_mode, bool use_signs, bool use_orients, CeedVector u,
                                            CeedVector v, CeedRequest *request) {
  return CeedElemRestrictionApply_Ref_Core(r, 5, 8, 1, start, stop, t_mode, use_signs, use_orients, u, v, request);
}

//------------------------------------------------------------------------------
// ElemRestriction Apply
//------------------------------------------------------------------------------
static int CeedElemRestrictionApply_Ref(CeedElemRestriction r, CeedTransposeMode t_mode, CeedVector u, CeedVector v, CeedRequest *request) {
  CeedInt num_blk, blk_size, num_comp, comp_stride;
  CeedCallBackend(CeedElemRestrictionGetNumBlocks(r, &num_blk));
  CeedCallBackend(CeedElemRestrictionGetBlockSize(r, &blk_size));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(r, &num_comp));
  CeedCallBackend(CeedElemRestrictionGetCompStride(r, &comp_stride));
  CeedElemRestriction_Ref *impl;
  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));

  return impl->Apply(r, num_comp, blk_size, comp_stride, 0, num_blk, t_mode, true, true, u, v, request);
}

//------------------------------------------------------------------------------
// ElemRestriction Apply Unsigned
//------------------------------------------------------------------------------
static int CeedElemRestrictionApplyUnsigned_Ref(CeedElemRestriction r, CeedTransposeMode t_mode, CeedVector u, CeedVector v, CeedRequest *request) {
  CeedInt num_blk, blk_size, num_comp, comp_stride;
  CeedCallBackend(CeedElemRestrictionGetNumBlocks(r, &num_blk));
  CeedCallBackend(CeedElemRestrictionGetBlockSize(r, &blk_size));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(r, &num_comp));
  CeedCallBackend(CeedElemRestrictionGetCompStride(r, &comp_stride));
  CeedElemRestriction_Ref *impl;
  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));

  return impl->Apply(r, num_comp, blk_size, comp_stride, 0, num_blk, t_mode, false, true, u, v, request);
}

//------------------------------------------------------------------------------
// ElemRestriction Apply Unoriented
//------------------------------------------------------------------------------
static int CeedElemRestrictionApplyUnoriented_Ref(CeedElemRestriction r, CeedTransposeMode t_mode, CeedVector u, CeedVector v, CeedRequest *request) {
  CeedInt num_blk, blk_size, num_comp, comp_stride;
  CeedCallBackend(CeedElemRestrictionGetNumBlocks(r, &num_blk));
  CeedCallBackend(CeedElemRestrictionGetBlockSize(r, &blk_size));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(r, &num_comp));
  CeedCallBackend(CeedElemRestrictionGetCompStride(r, &comp_stride));
  CeedElemRestriction_Ref *impl;
  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));

  return impl->Apply(r, num_comp, blk_size, comp_stride, 0, num_blk, t_mode, false, false, u, v, request);
}

//------------------------------------------------------------------------------
// ElemRestriction Apply Block
//------------------------------------------------------------------------------
static int CeedElemRestrictionApplyBlock_Ref(CeedElemRestriction r, CeedInt block, CeedTransposeMode t_mode, CeedVector u, CeedVector v,
                                             CeedRequest *request) {
  CeedInt blk_size, num_comp, comp_stride;
  CeedCallBackend(CeedElemRestrictionGetBlockSize(r, &blk_size));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(r, &num_comp));
  CeedCallBackend(CeedElemRestrictionGetCompStride(r, &comp_stride));
  CeedElemRestriction_Ref *impl;
  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));

  return impl->Apply(r, num_comp, blk_size, comp_stride, block, block + 1, t_mode, true, true, u, v, request);
}

//------------------------------------------------------------------------------
// ElemRestriction Get Offsets
//------------------------------------------------------------------------------
static int CeedElemRestrictionGetOffsets_Ref(CeedElemRestriction rstr, CeedMemType mem_type, const CeedInt **offsets) {
  CeedElemRestriction_Ref *impl;
  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(rstr, &ceed));

  CeedCheck(mem_type == CEED_MEM_HOST, ceed, CEED_ERROR_BACKEND, "Can only provide to HOST memory");

  *offsets = impl->offsets;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// ElemRestriction Get Orientations
//------------------------------------------------------------------------------
static int CeedElemRestrictionGetOrientations_Ref(CeedElemRestriction rstr, CeedMemType mem_type, const bool **orients) {
  CeedElemRestriction_Ref *impl;
  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(rstr, &ceed));

  CeedCheck(mem_type == CEED_MEM_HOST, ceed, CEED_ERROR_BACKEND, "Can only provide to HOST memory");

  *orients = impl->orients;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// ElemRestriction Get Curl-Conforming Orientations
//------------------------------------------------------------------------------
static int CeedElemRestrictionGetCurlOrientations_Ref(CeedElemRestriction rstr, CeedMemType mem_type, const CeedInt8 **curl_orients) {
  CeedElemRestriction_Ref *impl;
  CeedCallBackend(CeedElemRestrictionGetData(rstr, &impl));
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(rstr, &ceed));

  CeedCheck(mem_type == CEED_MEM_HOST, ceed, CEED_ERROR_BACKEND, "Can only provide to HOST memory");

  *curl_orients = impl->curl_orients;
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// ElemRestriction Destroy
//------------------------------------------------------------------------------
static int CeedElemRestrictionDestroy_Ref(CeedElemRestriction r) {
  CeedElemRestriction_Ref *impl;
  CeedCallBackend(CeedElemRestrictionGetData(r, &impl));

  CeedCallBackend(CeedFree(&impl->offsets_allocated));
  CeedCallBackend(CeedFree(&impl->orients_allocated));
  CeedCallBackend(CeedFree(&impl->curl_orients_allocated));
  CeedCallBackend(CeedFree(&impl));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// ElemRestriction Create
//------------------------------------------------------------------------------
int CeedElemRestrictionCreate_Ref(CeedMemType mem_type, CeedCopyMode copy_mode, const CeedInt *offsets, const bool *orients,
                                  const CeedInt8 *curl_orients, CeedElemRestriction r) {
  CeedElemRestriction_Ref *impl;
  CeedInt                  num_elem, elem_size, num_blk, blk_size, num_comp, comp_stride;
  CeedCallBackend(CeedElemRestrictionGetNumElements(r, &num_elem));
  CeedCallBackend(CeedElemRestrictionGetElementSize(r, &elem_size));
  CeedCallBackend(CeedElemRestrictionGetNumBlocks(r, &num_blk));
  CeedCallBackend(CeedElemRestrictionGetBlockSize(r, &blk_size));
  CeedCallBackend(CeedElemRestrictionGetNumComponents(r, &num_comp));
  CeedCallBackend(CeedElemRestrictionGetCompStride(r, &comp_stride));
  Ceed ceed;
  CeedCallBackend(CeedElemRestrictionGetCeed(r, &ceed));

  CeedCheck(mem_type == CEED_MEM_HOST, ceed, CEED_ERROR_BACKEND, "Only MemType = HOST supported");
  CeedCallBackend(CeedCalloc(1, &impl));

  // Offsets data
  CeedRestrictionType rstr_type;
  CeedCallBackend(CeedElemRestrictionGetType(r, &rstr_type));
  if (rstr_type != CEED_RESTRICTION_STRIDED) {
    // Check indices for ref or memcheck backends
    Ceed parent_ceed = ceed, curr_ceed = NULL;
    while (parent_ceed != curr_ceed) {
      curr_ceed = parent_ceed;
      CeedCallBackend(CeedGetParent(curr_ceed, &parent_ceed));
    }
    const char *resource;
    CeedCallBackend(CeedGetResource(parent_ceed, &resource));
    if (!strcmp(resource, "/cpu/self/ref/serial") || !strcmp(resource, "/cpu/self/ref/blocked") || !strcmp(resource, "/cpu/self/memcheck/serial") ||
        !strcmp(resource, "/cpu/self/memcheck/blocked")) {
      CeedSize l_size;
      CeedCallBackend(CeedElemRestrictionGetLVectorSize(r, &l_size));

      for (CeedInt i = 0; i < num_elem * elem_size; i++) {
        CeedCheck(offsets[i] >= 0 && offsets[i] + (num_comp - 1) * comp_stride < l_size, ceed, CEED_ERROR_BACKEND,
                  "Restriction offset %" CeedInt_FMT " (%" CeedInt_FMT ") out of range [0, %" CeedInt_FMT "]", i, offsets[i], l_size);
      }
    }

    // Copy data
    switch (copy_mode) {
      case CEED_COPY_VALUES:
        CeedCallBackend(CeedMalloc(num_elem * elem_size, &impl->offsets_allocated));
        memcpy(impl->offsets_allocated, offsets, num_elem * elem_size * sizeof(offsets[0]));
        impl->offsets = impl->offsets_allocated;
        break;
      case CEED_OWN_POINTER:
        impl->offsets_allocated = (CeedInt *)offsets;
        impl->offsets           = impl->offsets_allocated;
        break;
      case CEED_USE_POINTER:
        impl->offsets = offsets;
    }

    // Orientation data
    if (rstr_type == CEED_RESTRICTION_ORIENTED) {
      CeedCheck(orients != NULL, ceed, CEED_ERROR_BACKEND, "No orients array provided for oriented restriction");
      switch (copy_mode) {
        case CEED_COPY_VALUES:
          CeedCallBackend(CeedMalloc(num_elem * elem_size, &impl->orients_allocated));
          memcpy(impl->orients_allocated, orients, num_elem * elem_size * sizeof(orients[0]));
          impl->orients = impl->orients_allocated;
          break;
        case CEED_OWN_POINTER:
          impl->orients_allocated = (bool *)orients;
          impl->orients           = impl->orients_allocated;
          break;
        case CEED_USE_POINTER:
          impl->orients = orients;
      }
    } else if (rstr_type == CEED_RESTRICTION_CURL_ORIENTED) {
      CeedCheck(curl_orients != NULL, ceed, CEED_ERROR_BACKEND, "No curl_orients array provided for oriented restriction");
      switch (copy_mode) {
        case CEED_COPY_VALUES:
          CeedCallBackend(CeedMalloc(num_elem * 3 * elem_size, &impl->curl_orients_allocated));
          memcpy(impl->curl_orients_allocated, curl_orients, num_elem * 3 * elem_size * sizeof(curl_orients[0]));
          impl->curl_orients = impl->curl_orients_allocated;
          break;
        case CEED_OWN_POINTER:
          impl->curl_orients_allocated = (CeedInt8 *)curl_orients;
          impl->curl_orients           = impl->curl_orients_allocated;
          break;
        case CEED_USE_POINTER:
          impl->curl_orients = curl_orients;
      }
    }
  }

  CeedCallBackend(CeedElemRestrictionSetData(r, impl));
  CeedInt layout[3] = {1, elem_size, elem_size * num_comp};
  CeedCallBackend(CeedElemRestrictionSetELayout(r, layout));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", r, "Apply", CeedElemRestrictionApply_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", r, "ApplyUnsigned", CeedElemRestrictionApplyUnsigned_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", r, "ApplyUnoriented", CeedElemRestrictionApplyUnoriented_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", r, "ApplyBlock", CeedElemRestrictionApplyBlock_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", r, "GetOffsets", CeedElemRestrictionGetOffsets_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", r, "GetOrientations", CeedElemRestrictionGetOrientations_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", r, "GetCurlOrientations", CeedElemRestrictionGetCurlOrientations_Ref));
  CeedCallBackend(CeedSetBackendFunction(ceed, "ElemRestriction", r, "Destroy", CeedElemRestrictionDestroy_Ref));

  // Set apply function based upon num_comp, blk_size, and comp_stride
  CeedInt idx = -1;
  if (blk_size < 10) idx = 100 * num_comp + 10 * blk_size + (comp_stride == 1);
  switch (idx) {
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

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
