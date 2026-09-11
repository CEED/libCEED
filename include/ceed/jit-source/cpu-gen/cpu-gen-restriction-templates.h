// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CPU JiT backend ElemRestriction templates
#pragma once

#include <ceed/types.h>
#include "cpu-gen-utils.h"

#include <math.h>

//------------------------------------------------------------------------------
// Strided
//------------------------------------------------------------------------------

template <CeedInt BLK_SIZE, CeedInt NUM_COMP, CeedInt ELEM_SIZE, CeedInt NUM_ELEM, CeedInt STRIDES_NODE, CeedInt STRIDES_COMP, CeedInt STRIDES_ELEM>
static inline int CeedElemRestriction_Apply_NoTranspose_Strided(const CeedInt block, const CeedScalar *__restrict__ uu, CeedScalar *__restrict__ vv) {
  const CeedInt e = block * BLK_SIZE;

  for (CeedSize k = 0; k < NUM_COMP; k++) {
    for (CeedSize n = 0; n < ELEM_SIZE; n++) {
      CeedPragmaSIMD for (CeedSize j = 0; j < BLK_SIZE; j++) {
        vv[(k * ELEM_SIZE + n) * BLK_SIZE + j] = uu[n * STRIDES_NODE + k * STRIDES_COMP + CeedIntMin(e + j, NUM_ELEM - 1) * (CeedSize)STRIDES_ELEM];
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLK_SIZE, CeedInt NUM_COMP, CeedInt ELEM_SIZE, CeedInt NUM_ELEM, CeedInt STRIDES_NODE, CeedInt STRIDES_COMP, CeedInt STRIDES_ELEM>
static inline int CeedElemRestriction_ApplyAdd_Transpose_Strided(const CeedInt block, const CeedScalar *__restrict__ uu,
                                                                 CeedScalar *__restrict__ vv) {
  const CeedInt e = block * BLK_SIZE;

  for (CeedSize k = 0; k < NUM_COMP; k++) {
    for (CeedSize n = 0; n < ELEM_SIZE; n++) {
      CeedPragmaSIMD for (CeedSize j = 0; j < CeedIntMin(BLK_SIZE, NUM_ELEM - e); j++) {
        vv[n * STRIDES_NODE + k * STRIDES_COMP + (e + j) * STRIDES_ELEM] += uu[(k * ELEM_SIZE + n) * BLK_SIZE + j];
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Offset
//------------------------------------------------------------------------------

template <CeedInt BLK_SIZE, CeedInt NUM_COMP, CeedInt ELEM_SIZE, CeedInt NUM_ELEM, CeedInt COMP_STRIDE>
static inline int CeedElemRestriction_Apply_NoTranspose_Offset(const CeedInt block, const CeedInt *offsets, const CeedScalar *__restrict__ uu,
                                                               CeedScalar *__restrict__ vv) {
  const CeedInt e = block * BLK_SIZE;

  for (CeedSize k = 0; k < NUM_COMP; k++) {
    CeedPragmaSIMD for (CeedSize i = 0; i < ELEM_SIZE * BLK_SIZE; i++) {
      vv[ELEM_SIZE * (k * BLK_SIZE) + i] = uu[offsets[i + e * ELEM_SIZE] + k * COMP_STRIDE];
    }
  }
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLK_SIZE, CeedInt NUM_COMP, CeedInt ELEM_SIZE, CeedInt NUM_ELEM, CeedInt COMP_STRIDE>
static inline int CeedElemRestriction_ApplyAdd_Transpose_Offset(const CeedInt block, const CeedInt *offsets, const CeedScalar *__restrict__ uu,
                                                                CeedScalar *__restrict__ vv) {
  const CeedInt e = block * BLK_SIZE;

  for (CeedSize k = 0; k < NUM_COMP; k++) {
    for (CeedSize i = 0; i < ELEM_SIZE * BLK_SIZE; i += BLK_SIZE) {
      // Iteration bound set to discard padding elements
      for (CeedSize j = i; j < i + CeedIntMin(BLK_SIZE, NUM_ELEM - e); j++) {
        vv[offsets[j + e * ELEM_SIZE] + k * COMP_STRIDE] += uu[ELEM_SIZE * (k * BLK_SIZE) + j];
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Oriented
//------------------------------------------------------------------------------

template <CeedInt BLK_SIZE, CeedInt NUM_COMP, CeedInt ELEM_SIZE, CeedInt NUM_ELEM, CeedInt COMP_STRIDE>
static inline int CeedElemRestriction_Apply_NoTranspose_Oriented(const CeedInt block, const CeedInt *offsets, const bool *orients,
                                                                 const CeedScalar *__restrict__ uu, CeedScalar *__restrict__ vv) {
  const CeedInt e = block * BLK_SIZE;

  for (CeedSize k = 0; k < NUM_COMP; k++) {
    CeedPragmaSIMD for (CeedSize i = 0; i < ELEM_SIZE * BLK_SIZE; i++) {
      const CeedScalar orient = orients[i + e * ELEM_SIZE] ? -1.0 : 1.0;

      vv[ELEM_SIZE * (k * BLK_SIZE) + i] = orient * uu[offsets[i + e * ELEM_SIZE] + k * COMP_STRIDE];
    }
  }
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLK_SIZE, CeedInt NUM_COMP, CeedInt ELEM_SIZE, CeedInt NUM_ELEM, CeedInt COMP_STRIDE>
static inline int CeedElemRestriction_ApplyAdd_Transpose_Oriented(const CeedInt block, const CeedInt *offsets, const bool *orients,
                                                                  const CeedScalar *__restrict__ uu, CeedScalar *__restrict__ vv) {
  const CeedInt e = block * BLK_SIZE;

  for (CeedSize k = 0; k < NUM_COMP; k++) {
    for (CeedSize i = 0; i < ELEM_SIZE * BLK_SIZE; i += BLK_SIZE) {
      // Iteration bound set to discard padding elements
      for (CeedSize j = i; j < i + CeedIntMin(BLK_SIZE, NUM_ELEM - e); j++) {
        const CeedScalar orient = orients[j + e * ELEM_SIZE] ? -1.0 : 1.0;

        vv[offsets[j + e * ELEM_SIZE] + k * COMP_STRIDE] += orient * uu[ELEM_SIZE * (k * BLK_SIZE) + j];
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Curl Oriented
//------------------------------------------------------------------------------

template <CeedInt BLK_SIZE, CeedInt NUM_COMP, CeedInt ELEM_SIZE, CeedInt NUM_ELEM, CeedInt COMP_STRIDE>
static inline int CeedElemRestriction_Apply_NoTranspose_CurlOriented(const CeedInt block, const CeedInt *offsets, const CeedInt8 *curl_orients,
                                                                     const CeedScalar *__restrict__ uu, CeedScalar *__restrict__ vv) {
  const CeedInt e = block * BLK_SIZE;

  for (CeedSize k = 0; k < NUM_COMP; k++) {
    CeedSize n = 0;

    CeedPragmaSIMD for (CeedSize j = 0; j < BLK_SIZE; j++) {
      vv[(k * ELEM_SIZE + n) * BLK_SIZE + j] =
          uu[offsets[j + n * BLK_SIZE + e * ELEM_SIZE] + k * COMP_STRIDE] * curl_orients[j + (3 * n + 1) * BLK_SIZE + e * 3 * ELEM_SIZE] +
          uu[offsets[j + (n + 1) * BLK_SIZE + e * ELEM_SIZE] + k * COMP_STRIDE] * curl_orients[j + (3 * n + 2) * BLK_SIZE + e * 3 * ELEM_SIZE];
    }
    for (n = 1; n < ELEM_SIZE - 1; n++) {
      CeedPragmaSIMD for (CeedSize j = 0; j < BLK_SIZE; j++) {
        vv[(k * ELEM_SIZE + n) * BLK_SIZE + j] =
            uu[offsets[j + (n - 1) * BLK_SIZE + e * ELEM_SIZE] + k * COMP_STRIDE] * curl_orients[j + (3 * n + 0) * BLK_SIZE + e * 3 * ELEM_SIZE] +
            uu[offsets[j + n * BLK_SIZE + e * ELEM_SIZE] + k * COMP_STRIDE] * curl_orients[j + (3 * n + 1) * BLK_SIZE + e * 3 * ELEM_SIZE] +
            uu[offsets[j + (n + 1) * BLK_SIZE + e * ELEM_SIZE] + k * COMP_STRIDE] * curl_orients[j + (3 * n + 2) * BLK_SIZE + e * 3 * ELEM_SIZE];
      }
    }
    CeedPragmaSIMD for (CeedSize j = 0; j < BLK_SIZE; j++) {
      vv[(k * ELEM_SIZE + n) * BLK_SIZE + j] =
          uu[offsets[j + (n - 1) * BLK_SIZE + e * ELEM_SIZE] + k * COMP_STRIDE] * curl_orients[j + (3 * n + 0) * BLK_SIZE + e * 3 * ELEM_SIZE] +
          uu[offsets[j + n * BLK_SIZE + e * ELEM_SIZE] + k * COMP_STRIDE] * curl_orients[j + (3 * n + 1) * BLK_SIZE + e * 3 * ELEM_SIZE];
    }
  }
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLK_SIZE, CeedInt NUM_COMP, CeedInt ELEM_SIZE, CeedInt NUM_ELEM, CeedInt COMP_STRIDE>
static inline int CeedElemRestriction_Apply_NoTranspose_CurlOrientedUnsigned(const CeedInt block, const CeedInt *offsets,
                                                                             const CeedInt8 *curl_orients, const CeedScalar *__restrict__ uu,
                                                                             CeedScalar *__restrict__ vv) {
  const CeedInt e = block * BLK_SIZE;

  for (CeedSize k = 0; k < NUM_COMP; k++) {
    CeedSize n = 0;

    CeedPragmaSIMD for (CeedSize j = 0; j < BLK_SIZE; j++) {
      vv[(k * ELEM_SIZE + n) * BLK_SIZE + j] =
          uu[offsets[j + n * BLK_SIZE + e * ELEM_SIZE] + k * COMP_STRIDE] * abs(curl_orients[j + (3 * n + 1) * BLK_SIZE + e * 3 * ELEM_SIZE]) +
          uu[offsets[j + (n + 1) * BLK_SIZE + e * ELEM_SIZE] + k * COMP_STRIDE] * abs(curl_orients[j + (3 * n + 2) * BLK_SIZE + e * 3 * ELEM_SIZE]);
    }
    for (n = 1; n < ELEM_SIZE - 1; n++) {
      CeedPragmaSIMD for (CeedSize j = 0; j < BLK_SIZE; j++) {
        vv[(k * ELEM_SIZE + n) * BLK_SIZE + j] =
            uu[offsets[j + (n - 1) * BLK_SIZE + e * ELEM_SIZE] + k * COMP_STRIDE] *
                abs(curl_orients[j + (3 * n + 0) * BLK_SIZE + e * 3 * ELEM_SIZE]) +
            uu[offsets[j + n * BLK_SIZE + e * ELEM_SIZE] + k * COMP_STRIDE] * abs(curl_orients[j + (3 * n + 1) * BLK_SIZE + e * 3 * ELEM_SIZE]) +
            uu[offsets[j + (n + 1) * BLK_SIZE + e * ELEM_SIZE] + k * COMP_STRIDE] * abs(curl_orients[j + (3 * n + 2) * BLK_SIZE + e * 3 * ELEM_SIZE]);
      }
    }
    CeedPragmaSIMD for (CeedSize j = 0; j < BLK_SIZE; j++) {
      vv[(k * ELEM_SIZE + n) * BLK_SIZE + j] =
          uu[offsets[j + (n - 1) * BLK_SIZE + e * ELEM_SIZE] + k * COMP_STRIDE] * abs(curl_orients[j + (3 * n + 0) * BLK_SIZE + e * 3 * ELEM_SIZE]) +
          uu[offsets[j + n * BLK_SIZE + e * ELEM_SIZE] + k * COMP_STRIDE] * abs(curl_orients[j + (3 * n + 1) * BLK_SIZE + e * 3 * ELEM_SIZE]);
    }
  }
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLK_SIZE, CeedInt NUM_COMP, CeedInt ELEM_SIZE, CeedInt NUM_ELEM, CeedInt COMP_STRIDE>
static inline int CeedElemRestriction_ApplyAdd_Transpose_CurlOriented(const CeedInt block, const CeedInt *offsets, const CeedInt8 *curl_orients,
                                                                      const CeedScalar *__restrict__ uu, CeedScalar *__restrict__ vv) {
  CeedScalar vv_loc[BLK_SIZE];

  const CeedInt e = block * BLK_SIZE;

  for (CeedSize k = 0; k < NUM_COMP; k++) {
    // Iteration bound set to discard padding elements
    const CeedSize block_end = CeedIntMin(BLK_SIZE, NUM_ELEM - e);
    CeedSize       n         = 0;

    CeedPragmaSIMD for (CeedSize j = 0; j < block_end; j++) {
      vv_loc[j] = uu[(k * ELEM_SIZE + n) * BLK_SIZE + j] * curl_orients[j + (3 * n + 1) * BLK_SIZE + e * 3 * ELEM_SIZE] +
                  uu[(k * ELEM_SIZE + n + 1) * BLK_SIZE + j] * curl_orients[j + (3 * n + 3) * BLK_SIZE + e * 3 * ELEM_SIZE];
    }
    for (CeedSize j = 0; j < block_end; j++) {
      vv[offsets[j + n * BLK_SIZE + e * ELEM_SIZE] + k * COMP_STRIDE] += vv_loc[j];
    }
    for (n = 1; n < ELEM_SIZE - 1; n++) {
      CeedPragmaSIMD for (CeedInt j = 0; j < block_end; j++) {
        vv_loc[j] = uu[(k * ELEM_SIZE + n - 1) * BLK_SIZE + j] * curl_orients[j + (3 * n - 1) * BLK_SIZE + e * 3 * ELEM_SIZE] +
                    uu[(k * ELEM_SIZE + n) * BLK_SIZE + j] * curl_orients[j + (3 * n + 1) * BLK_SIZE + e * 3 * ELEM_SIZE] +
                    uu[(k * ELEM_SIZE + n + 1) * BLK_SIZE + j] * curl_orients[j + (3 * n + 3) * BLK_SIZE + e * 3 * ELEM_SIZE];
      }
      for (CeedSize j = 0; j < block_end; j++) {
        vv[offsets[j + n * BLK_SIZE + e * ELEM_SIZE] + k * COMP_STRIDE] += vv_loc[j];
      }
    }
    CeedPragmaSIMD for (CeedSize j = 0; j < block_end; j++) {
      vv_loc[j] = uu[(k * ELEM_SIZE + n - 1) * BLK_SIZE + j] * curl_orients[j + (3 * n - 1) * BLK_SIZE + e * 3 * ELEM_SIZE] +
                  uu[(k * ELEM_SIZE + n) * BLK_SIZE + j] * curl_orients[j + (3 * n + 1) * BLK_SIZE + e * 3 * ELEM_SIZE];
    }
    for (CeedSize j = 0; j < block_end; j++) {
      vv[offsets[j + n * BLK_SIZE + e * ELEM_SIZE] + k * COMP_STRIDE] += vv_loc[j];
    }
  }
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLK_SIZE, CeedInt NUM_COMP, CeedInt ELEM_SIZE, CeedInt NUM_ELEM, CeedInt COMP_STRIDE>
static inline int CeedElemRestriction_ApplyAdd_Transpose_CurlOrientedUnsigned(const CeedInt block, const CeedInt *offsets,
                                                                              const CeedInt8 *curl_orients, const CeedScalar *__restrict__ uu,
                                                                              CeedScalar *__restrict__ vv) {
  CeedScalar vv_loc[BLK_SIZE];

  const CeedInt e = block * BLK_SIZE;

  for (CeedSize k = 0; k < NUM_COMP; k++) {
    // Iteration bound set to discard padding elements
    const CeedSize block_end = CeedIntMin(BLK_SIZE, NUM_ELEM - e);
    CeedSize       n         = 0;

    CeedPragmaSIMD for (CeedSize j = 0; j < block_end; j++) {
      vv_loc[j] = uu[(k * ELEM_SIZE + n) * BLK_SIZE + j] * abs(curl_orients[j + (3 * n + 1) * BLK_SIZE + e * 3 * ELEM_SIZE]) +
                  uu[(k * ELEM_SIZE + n + 1) * BLK_SIZE + j] * abs(curl_orients[j + (3 * n + 3) * BLK_SIZE + e * 3 * ELEM_SIZE]);
    }
    for (CeedSize j = 0; j < block_end; j++) {
      vv[offsets[j + n * BLK_SIZE + e * ELEM_SIZE] + k * COMP_STRIDE] += vv_loc[j];
    }
    for (n = 1; n < ELEM_SIZE - 1; n++) {
      CeedPragmaSIMD for (CeedSize j = 0; j < block_end; j++) {
        vv_loc[j] = uu[(k * ELEM_SIZE + n - 1) * BLK_SIZE + j] * abs(curl_orients[j + (3 * n - 1) * BLK_SIZE + e * 3 * ELEM_SIZE]) +
                    uu[(k * ELEM_SIZE + n) * BLK_SIZE + j] * abs(curl_orients[j + (3 * n + 1) * BLK_SIZE + e * 3 * ELEM_SIZE]) +
                    uu[(k * ELEM_SIZE + n + 1) * BLK_SIZE + j] * abs(curl_orients[j + (3 * n + 3) * BLK_SIZE + e * 3 * ELEM_SIZE]);
      }
      for (CeedSize j = 0; j < block_end; j++) {
        vv[offsets[j + n * BLK_SIZE + e * ELEM_SIZE] + k * COMP_STRIDE] += vv_loc[j];
      }
    }
    CeedPragmaSIMD for (CeedSize j = 0; j < block_end; j++) {
      vv_loc[j] = uu[(k * ELEM_SIZE + n - 1) * BLK_SIZE + j] * abs(curl_orients[j + (3 * n - 1) * BLK_SIZE + e * 3 * ELEM_SIZE]) +
                  uu[(k * ELEM_SIZE + n) * BLK_SIZE + j] * abs(curl_orients[j + (3 * n + 1) * BLK_SIZE + e * 3 * ELEM_SIZE]);
    }
    for (CeedSize j = 0; j < block_end; j++) {
      vv[offsets[j + n * BLK_SIZE + e * ELEM_SIZE] + k * COMP_STRIDE] += vv_loc[j];
    }
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// AtPoints
//------------------------------------------------------------------------------

template <CeedInt BLK_SIZE, CeedInt NUM_COMP>
static inline int CeedElemRestriction_Apply_NoTranspose_AtPoints(const CeedInt block, const CeedInt *offsets, const CeedScalar *__restrict__ uu,
                                                                 CeedScalar *__restrict__ vv) {
  CeedSize e_vec_offset = 0;

  for (CeedSize e = block * BLK_SIZE; e < (block + 1) * BLK_SIZE; e += 1) {
    const CeedInt num_points = offsets[e + 1] - offsets[e];

    for (CeedSize i = 0; i < num_points; i++) {
      for (CeedSize j = 0; j < NUM_COMP; j++) vv[j * num_points + i + e_vec_offset] = uu[offsets[i + offsets[e]] * NUM_COMP + j];
    }
    e_vec_offset += num_points * (CeedSize)NUM_COMP;
  }
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLK_SIZE, CeedInt NUM_COMP>
static inline int CeedElemRestriction_Apply_Transpose_AtPoints(const CeedInt block, const CeedInt *offsets, const CeedScalar *__restrict__ uu,
                                                               CeedScalar *__restrict__ vv) {
  CeedSize e_vec_offset = 0;

  for (CeedSize e = block * BLK_SIZE; e < (block + 1) * BLK_SIZE; e += 1) {
    const CeedInt num_points = offsets[e + 1] - offsets[e];

    for (CeedSize i = 0; i < num_points; i++) {
      for (CeedSize j = 0; j < NUM_COMP; j++) vv[offsets[i + offsets[e]] * NUM_COMP + j] = uu[j * num_points + i + e_vec_offset];
    }
    e_vec_offset += num_points * (CeedSize)NUM_COMP;
  }
  return CEED_ERROR_SUCCESS;
}

template <CeedInt BLK_SIZE, CeedInt NUM_COMP>
static inline int CeedElemRestriction_ApplyAdd_Transpose_AtPoints(const CeedInt block, const CeedInt *offsets, const CeedScalar *__restrict__ uu,
                                                                  CeedScalar *__restrict__ vv) {
  CeedSize e_vec_offset = 0;

  for (CeedSize e = block * BLK_SIZE; e < (block + 1) * BLK_SIZE; e += 1) {
    const CeedInt num_points = offsets[e + 1] - offsets[e];

    for (CeedSize i = 0; i < num_points; i++) {
      for (CeedSize j = 0; j < NUM_COMP; j++) vv[offsets[i + offsets[e]] * NUM_COMP + j] += uu[j * num_points + i + e_vec_offset];
    }
    e_vec_offset += num_points * (CeedSize)NUM_COMP;
  }
  return CEED_ERROR_SUCCESS;
}
