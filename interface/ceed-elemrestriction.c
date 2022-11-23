// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed-impl.h>
#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <stdbool.h>
#include <stdio.h>

/// @file
/// Implementation of CeedElemRestriction interfaces

/// ----------------------------------------------------------------------------
/// CeedElemRestriction Library Internal Functions
/// ----------------------------------------------------------------------------
/// @addtogroup CeedElemRestrictionDeveloper
/// @{

/**
  @brief Permute and pad offsets for a blocked restriction

  @param offsets     Array of shape [@a num_elem, @a elem_size]. Row i holds the
                       ordered list of the offsets (into the input CeedVector)
                       for the unknowns corresponding to element i, where
                       0 <= i < @a num_elem. All offsets must be in the range
                       [0, @a l_size - 1].
  @param blk_offsets Array of permuted and padded offsets of
                       shape [@a num_blk, @a elem_size, @a blk_size].
  @param num_blk     Number of blocks
  @param num_elem    Number of elements
  @param blk_size    Number of elements in a block
  @param elem_size   Size of each element

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedPermutePadOffsets(const CeedInt *offsets, CeedInt *blk_offsets, CeedInt num_blk, CeedInt num_elem, CeedInt blk_size, CeedInt elem_size) {
  for (CeedInt e = 0; e < num_blk * blk_size; e += blk_size) {
    for (CeedInt j = 0; j < blk_size; j++) {
      for (CeedInt k = 0; k < elem_size; k++) {
        blk_offsets[e * elem_size + k * blk_size + j] = offsets[CeedIntMin(e + j, num_elem - 1) * elem_size + k];
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

/// @}

/// ----------------------------------------------------------------------------
/// CeedElemRestriction Backend API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedElemRestrictionBackend
/// @{

/**

  @brief Get the strides of a strided CeedElemRestriction

  @param rstr          CeedElemRestriction
  @param[out] strides  Variable to store strides array

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionGetStrides(CeedElemRestriction rstr, CeedInt (*strides)[3]) {
  if (!rstr->strides) {
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, CEED_ERROR_MINOR, "ElemRestriction has no stride data");
    // LCOV_EXCL_STOP
  }

  for (CeedInt i = 0; i < 3; i++) (*strides)[i] = rstr->strides[i];
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get read-only access to a CeedElemRestriction offsets array by memtype

  @param rstr          CeedElemRestriction to retrieve offsets
  @param mem_type      Memory type on which to access the array.  If the backend
                         uses a different memory type, this will perform a copy
                         (possibly cached).
  @param[out] offsets  Array on memory type mem_type

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionGetOffsets(CeedElemRestriction rstr, CeedMemType mem_type, const CeedInt **offsets) {
  if (!rstr->GetOffsets) {
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support GetOffsets");
    // LCOV_EXCL_STOP
  }

  CeedCall(rstr->GetOffsets(rstr, mem_type, offsets));
  rstr->num_readers++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restore an offsets array obtained using CeedElemRestrictionGetOffsets()

  @param rstr     CeedElemRestriction to restore
  @param offsets  Array of offset data

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionRestoreOffsets(CeedElemRestriction rstr, const CeedInt **offsets) {
  *offsets = NULL;
  rstr->num_readers--;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the strided status of a CeedElemRestriction

  @param rstr             CeedElemRestriction
  @param[out] is_strided  Variable to store strided status, 1 if strided else 0

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionIsStrided(CeedElemRestriction rstr, bool *is_strided) {
  *is_strided = rstr->strides ? true : false;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get oriented status of a CeedElemRestriction

  @param rstr             CeedElemRestriction
  @param[out] is_oriented  Variable to store oriented status, 1 if oriented else 0

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionIsOriented(CeedElemRestriction rstr, bool *is_oriented) {
  *is_oriented = rstr->is_oriented;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the backend stride status of a CeedElemRestriction

  @param rstr                      CeedElemRestriction
  @param[out] has_backend_strides  Variable to store stride status

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionHasBackendStrides(CeedElemRestriction rstr, bool *has_backend_strides) {
  if (!rstr->strides) {
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, CEED_ERROR_MINOR, "ElemRestriction has no stride data");
    // LCOV_EXCL_STOP
  }

  *has_backend_strides = ((rstr->strides[0] == CEED_STRIDES_BACKEND[0]) && (rstr->strides[1] == CEED_STRIDES_BACKEND[1]) &&
                          (rstr->strides[2] == CEED_STRIDES_BACKEND[2]));
  return CEED_ERROR_SUCCESS;
}

/**

  @brief Get the E-vector layout of a CeedElemRestriction

  @param rstr         CeedElemRestriction
  @param[out] layout  Variable to store layout array,
                        stored as [nodes, components, elements].
                        The data for node i, component j, element k in the
                        E-vector is given by
                        i*layout[0] + j*layout[1] + k*layout[2]

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionGetELayout(CeedElemRestriction rstr, CeedInt (*layout)[3]) {
  if (!rstr->layout[0]) {
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, CEED_ERROR_MINOR, "ElemRestriction has no layout data");
    // LCOV_EXCL_STOP
  }

  for (CeedInt i = 0; i < 3; i++) (*layout)[i] = rstr->layout[i];
  return CEED_ERROR_SUCCESS;
}

/**

  @brief Set the E-vector layout of a CeedElemRestriction

  @param rstr    CeedElemRestriction
  @param layout  Variable to containing layout array,
                   stored as [nodes, components, elements].
                   The data for node i, component j, element k in the
                   E-vector is given by
                   i*layout[0] + j*layout[1] + k*layout[2]

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionSetELayout(CeedElemRestriction rstr, CeedInt layout[3]) {
  for (CeedInt i = 0; i < 3; i++) rstr->layout[i] = layout[i];
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the backend data of a CeedElemRestriction

  @param rstr       CeedElemRestriction
  @param[out] data  Variable to store data

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionGetData(CeedElemRestriction rstr, void *data) {
  *(void **)data = rstr->data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set the backend data of a CeedElemRestriction

  @param[out] rstr  CeedElemRestriction
  @param data       Data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionSetData(CeedElemRestriction rstr, void *data) {
  rstr->data = data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Increment the reference counter for a CeedElemRestriction

  @param rstr  ElemRestriction to increment the reference counter

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionReference(CeedElemRestriction rstr) {
  rstr->ref_count++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Estimate number of FLOPs required to apply CeedElemRestriction in t_mode

  @param rstr   ElemRestriction to estimate FLOPs for
  @param t_mode Apply restriction or transpose
  @param flops  Address of variable to hold FLOPs estimate

  @ref Backend
**/
int CeedElemRestrictionGetFlopsEstimate(CeedElemRestriction rstr, CeedTransposeMode t_mode, CeedSize *flops) {
  bool    is_oriented;
  CeedInt e_size = rstr->num_blk * rstr->blk_size * rstr->elem_size * rstr->num_comp, scale = 0;

  CeedCall(CeedElemRestrictionIsOriented(rstr, &is_oriented));
  switch (t_mode) {
    case CEED_NOTRANSPOSE:
      scale = is_oriented ? 1 : 0;
      break;
    case CEED_TRANSPOSE:
      scale = is_oriented ? 2 : 1;
      break;
  }
  *flops = e_size * scale;

  return CEED_ERROR_SUCCESS;
}

/// @}

/// @cond DOXYGEN_SKIP
static struct CeedElemRestriction_private ceed_elemrestriction_none;
/// @endcond

/// ----------------------------------------------------------------------------
/// CeedElemRestriction Public API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedElemRestrictionUser
/// @{

/// Indicate that the stride is determined by the backend
const CeedInt CEED_STRIDES_BACKEND[3] = {0};

/// Indicate that no CeedElemRestriction is provided by the user
const CeedElemRestriction CEED_ELEMRESTRICTION_NONE = &ceed_elemrestriction_none;

/**
  @brief Create a CeedElemRestriction

  @param ceed         A Ceed object where the CeedElemRestriction will be created
  @param num_elem     Number of elements described in the @a offsets array
  @param elem_size    Size (number of "nodes") per element
  @param num_comp     Number of field components per interpolation node
                        (1 for scalar fields)
  @param comp_stride  Stride between components for the same L-vector "node".
                        Data for node i, component j, element k can be found in
                        the L-vector at index
                        offsets[i + k*elem_size] + j*comp_stride.
  @param l_size       The size of the L-vector. This vector may be larger than
                        the elements and fields given by this restriction.
  @param mem_type     Memory type of the @a offsets array, see CeedMemType
  @param copy_mode    Copy mode for the @a offsets array, see CeedCopyMode
  @param offsets      Array of shape [@a num_elem, @a elem_size]. Row i holds the
                        ordered list of the offsets (into the input CeedVector)
                        for the unknowns corresponding to element i, where
                        0 <= i < @a num_elem. All offsets must be in the range
                        [0, @a l_size - 1].
  @param[out] rstr    Address of the variable where the newly created
                        CeedElemRestriction will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionCreate(Ceed ceed, CeedInt num_elem, CeedInt elem_size, CeedInt num_comp, CeedInt comp_stride, CeedSize l_size,
                              CeedMemType mem_type, CeedCopyMode copy_mode, const CeedInt *offsets, CeedElemRestriction *rstr) {
  if (!ceed->ElemRestrictionCreate) {
    Ceed delegate;
    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "ElemRestriction"));

    if (!delegate) {
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support ElemRestrictionCreate");
      // LCOV_EXCL_STOP
    }

    CeedCall(CeedElemRestrictionCreate(delegate, num_elem, elem_size, num_comp, comp_stride, l_size, mem_type, copy_mode, offsets, rstr));
    return CEED_ERROR_SUCCESS;
  }

  if (elem_size < 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "Element size must be at least 1");
    // LCOV_EXCL_STOP
  }

  if (num_comp < 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "ElemRestriction must have at least 1 component");
    // LCOV_EXCL_STOP
  }

  if (num_comp > 1 && comp_stride < 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "ElemRestriction component stride must be at least 1");
    // LCOV_EXCL_STOP
  }

  CeedCall(CeedCalloc(1, rstr));
  (*rstr)->ceed = ceed;
  CeedCall(CeedReference(ceed));
  (*rstr)->ref_count   = 1;
  (*rstr)->num_elem    = num_elem;
  (*rstr)->elem_size   = elem_size;
  (*rstr)->num_comp    = num_comp;
  (*rstr)->comp_stride = comp_stride;
  (*rstr)->l_size      = l_size;
  (*rstr)->num_blk     = num_elem;
  (*rstr)->blk_size    = 1;
  (*rstr)->is_oriented = 0;
  CeedCall(ceed->ElemRestrictionCreate(mem_type, copy_mode, offsets, *rstr));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a CeedElemRestriction with orientation sign

  @param ceed         A Ceed object where the CeedElemRestriction will be created
  @param num_elem     Number of elements described in the @a offsets array
  @param elem_size    Size (number of "nodes") per element
  @param num_comp     Number of field components per interpolation node
                        (1 for scalar fields)
  @param comp_stride  Stride between components for the same L-vector "node".
                        Data for node i, component j, element k can be found in
                        the L-vector at index
                        offsets[i + k*elem_size] + j*comp_stride.
  @param l_size       The size of the L-vector. This vector may be larger than
                        the elements and fields given by this restriction.
  @param mem_type     Memory type of the @a offsets array, see CeedMemType
  @param copy_mode    Copy mode for the @a offsets array, see CeedCopyMode
  @param offsets      Array of shape [@a num_elem, @a elem_size]. Row i holds the
                        ordered list of the offsets (into the input CeedVector)
                        for the unknowns corresponding to element i, where
                        0 <= i < @a num_elem. All offsets must be in the range
                        [0, @a l_size - 1].
  @param orient       Array of shape [@a num_elem, @a elem_size] with bool false
                        for positively oriented and true to flip the orientation.
  @param[out] rstr    Address of the variable where the newly created
                        CeedElemRestriction will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionCreateOriented(Ceed ceed, CeedInt num_elem, CeedInt elem_size, CeedInt num_comp, CeedInt comp_stride, CeedSize l_size,
                                      CeedMemType mem_type, CeedCopyMode copy_mode, const CeedInt *offsets, const bool *orient,
                                      CeedElemRestriction *rstr) {
  if (!ceed->ElemRestrictionCreateOriented) {
    Ceed delegate;
    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "ElemRestriction"));

    if (!delegate) {
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Backend does not implement ElemRestrictionCreateOriented");
      // LCOV_EXCL_STOP
    }

    CeedCall(
        CeedElemRestrictionCreateOriented(delegate, num_elem, elem_size, num_comp, comp_stride, l_size, mem_type, copy_mode, offsets, orient, rstr));
    return CEED_ERROR_SUCCESS;
  }

  if (elem_size < 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "Element size must be at least 1");
    // LCOV_EXCL_STOP
  }

  if (num_comp < 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "ElemRestriction must have at least 1 component");
    // LCOV_EXCL_STOP
  }

  if (num_comp > 1 && comp_stride < 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "ElemRestriction component stride must be at least 1");
    // LCOV_EXCL_STOP
  }

  CeedCall(CeedCalloc(1, rstr));
  (*rstr)->ceed = ceed;
  CeedCall(CeedReference(ceed));
  (*rstr)->ref_count   = 1;
  (*rstr)->num_elem    = num_elem;
  (*rstr)->elem_size   = elem_size;
  (*rstr)->num_comp    = num_comp;
  (*rstr)->comp_stride = comp_stride;
  (*rstr)->l_size      = l_size;
  (*rstr)->num_blk     = num_elem;
  (*rstr)->blk_size    = 1;
  (*rstr)->is_oriented = 1;
  CeedCall(ceed->ElemRestrictionCreateOriented(mem_type, copy_mode, offsets, orient, *rstr));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a strided CeedElemRestriction

  @param ceed       A Ceed object where the CeedElemRestriction will be created
  @param num_elem   Number of elements described by the restriction
  @param elem_size  Size (number of "nodes") per element
  @param num_comp   Number of field components per interpolation "node"
                      (1 for scalar fields)
  @param l_size     The size of the L-vector. This vector may be larger than
                      the elements and fields given by this restriction.
  @param strides    Array for strides between [nodes, components, elements].
                      Data for node i, component j, element k can be found in
                      the L-vector at index
                      i*strides[0] + j*strides[1] + k*strides[2].
                      @a CEED_STRIDES_BACKEND may be used with vectors created
                      by a Ceed backend.
  @param rstr       Address of the variable where the newly created
                      CeedElemRestriction will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionCreateStrided(Ceed ceed, CeedInt num_elem, CeedInt elem_size, CeedInt num_comp, CeedSize l_size, const CeedInt strides[3],
                                     CeedElemRestriction *rstr) {
  if (!ceed->ElemRestrictionCreate) {
    Ceed delegate;
    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "ElemRestriction"));

    if (!delegate) {
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support ElemRestrictionCreate");
      // LCOV_EXCL_STOP
    }

    CeedCall(CeedElemRestrictionCreateStrided(delegate, num_elem, elem_size, num_comp, l_size, strides, rstr));
    return CEED_ERROR_SUCCESS;
  }

  if (elem_size < 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "Element size must be at least 1");
    // LCOV_EXCL_STOP
  }

  if (num_comp < 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "ElemRestriction must have at least 1 component");
    // LCOV_EXCL_STOP
  }

  CeedCall(CeedCalloc(1, rstr));
  (*rstr)->ceed = ceed;
  CeedCall(CeedReference(ceed));
  (*rstr)->ref_count   = 1;
  (*rstr)->num_elem    = num_elem;
  (*rstr)->elem_size   = elem_size;
  (*rstr)->num_comp    = num_comp;
  (*rstr)->l_size      = l_size;
  (*rstr)->num_blk     = num_elem;
  (*rstr)->blk_size    = 1;
  (*rstr)->is_oriented = 0;
  CeedCall(CeedMalloc(3, &(*rstr)->strides));
  for (CeedInt i = 0; i < 3; i++) (*rstr)->strides[i] = strides[i];
  CeedCall(ceed->ElemRestrictionCreate(CEED_MEM_HOST, CEED_OWN_POINTER, NULL, *rstr));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a blocked CeedElemRestriction, typically only called by backends

  @param ceed         A Ceed object where the CeedElemRestriction will be created.
  @param num_elem     Number of elements described in the @a offsets array.
  @param elem_size    Size (number of unknowns) per element
  @param blk_size     Number of elements in a block
  @param num_comp     Number of field components per interpolation node
                        (1 for scalar fields)
  @param comp_stride  Stride between components for the same L-vector "node".
                        Data for node i, component j, element k can be found in
                        the L-vector at index
                        offsets[i + k*elem_size] + j*comp_stride.
  @param l_size       The size of the L-vector. This vector may be larger than
                        the elements and fields given by this restriction.
  @param mem_type     Memory type of the @a offsets array, see CeedMemType
  @param copy_mode    Copy mode for the @a offsets array, see CeedCopyMode
  @param offsets      Array of shape [@a num_elem, @a elem_size]. Row i holds the
                        ordered list of the offsets (into the input CeedVector)
                        for the unknowns corresponding to element i, where
                        0 <= i < @a num_elem. All offsets must be in the range
                        [0, @a l_size - 1]. The backend will permute and pad this
                        array to the desired ordering for the blocksize, which is
                        typically given by the backend. The default reordering is
                        to interlace elements.
  @param rstr         Address of the variable where the newly created
                        CeedElemRestriction will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
 **/
int CeedElemRestrictionCreateBlocked(Ceed ceed, CeedInt num_elem, CeedInt elem_size, CeedInt blk_size, CeedInt num_comp, CeedInt comp_stride,
                                     CeedSize l_size, CeedMemType mem_type, CeedCopyMode copy_mode, const CeedInt *offsets,
                                     CeedElemRestriction *rstr) {
  CeedInt *blk_offsets;
  CeedInt  num_blk = (num_elem / blk_size) + !!(num_elem % blk_size);

  if (!ceed->ElemRestrictionCreateBlocked) {
    Ceed delegate;
    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "ElemRestriction"));

    if (!delegate) {
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support ElemRestrictionCreateBlocked");
      // LCOV_EXCL_STOP
    }

    CeedCall(
        CeedElemRestrictionCreateBlocked(delegate, num_elem, elem_size, blk_size, num_comp, comp_stride, l_size, mem_type, copy_mode, offsets, rstr));
    return CEED_ERROR_SUCCESS;
  }

  if (elem_size < 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "Element size must be at least 1");
    // LCOV_EXCL_STOP
  }

  if (blk_size < 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "Block size must be at least 1");
    // LCOV_EXCL_STOP
  }

  if (num_comp < 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "ElemRestriction must have at least 1 component");
    // LCOV_EXCL_STOP
  }

  if (num_comp > 1 && comp_stride < 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "ElemRestriction component stride must be at least 1");
    // LCOV_EXCL_STOP
  }

  CeedCall(CeedCalloc(1, rstr));

  CeedCall(CeedCalloc(num_blk * blk_size * elem_size, &blk_offsets));
  CeedCall(CeedPermutePadOffsets(offsets, blk_offsets, num_blk, num_elem, blk_size, elem_size));

  (*rstr)->ceed = ceed;
  CeedCall(CeedReference(ceed));
  (*rstr)->ref_count   = 1;
  (*rstr)->num_elem    = num_elem;
  (*rstr)->elem_size   = elem_size;
  (*rstr)->num_comp    = num_comp;
  (*rstr)->comp_stride = comp_stride;
  (*rstr)->l_size      = l_size;
  (*rstr)->num_blk     = num_blk;
  (*rstr)->blk_size    = blk_size;
  (*rstr)->is_oriented = 0;
  CeedCall(ceed->ElemRestrictionCreateBlocked(CEED_MEM_HOST, CEED_OWN_POINTER, (const CeedInt *)blk_offsets, *rstr));
  if (copy_mode == CEED_OWN_POINTER) {
    CeedCall(CeedFree(&offsets));
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a blocked strided CeedElemRestriction

  @param ceed       A Ceed object where the CeedElemRestriction will be created
  @param num_elem   Number of elements described by the restriction
  @param elem_size  Size (number of "nodes") per element
  @param blk_size   Number of elements in a block
  @param num_comp   Number of field components per interpolation node
                      (1 for scalar fields)
  @param l_size     The size of the L-vector. This vector may be larger than
                      the elements and fields given by this restriction.
  @param strides    Array for strides between [nodes, components, elements].
                      Data for node i, component j, element k can be found in
                      the L-vector at index
                      i*strides[0] + j*strides[1] + k*strides[2].
                      @a CEED_STRIDES_BACKEND may be used with vectors created
                      by a Ceed backend.
  @param rstr       Address of the variable where the newly created
                      CeedElemRestriction will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionCreateBlockedStrided(Ceed ceed, CeedInt num_elem, CeedInt elem_size, CeedInt blk_size, CeedInt num_comp, CeedSize l_size,
                                            const CeedInt strides[3], CeedElemRestriction *rstr) {
  CeedInt num_blk = (num_elem / blk_size) + !!(num_elem % blk_size);

  if (!ceed->ElemRestrictionCreateBlocked) {
    Ceed delegate;
    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "ElemRestriction"));

    if (!delegate) {
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support ElemRestrictionCreateBlocked");
      // LCOV_EXCL_STOP
    }

    CeedCall(CeedElemRestrictionCreateBlockedStrided(delegate, num_elem, elem_size, blk_size, num_comp, l_size, strides, rstr));
    return CEED_ERROR_SUCCESS;
  }

  if (elem_size < 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "Element size must be at least 1");
    // LCOV_EXCL_STOP
  }

  if (blk_size < 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "Block size must be at least 1");
    // LCOV_EXCL_STOP
  }

  if (num_comp < 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "ElemRestriction must have at least 1 component");
    // LCOV_EXCL_STOP
  }

  CeedCall(CeedCalloc(1, rstr));

  (*rstr)->ceed = ceed;
  CeedCall(CeedReference(ceed));
  (*rstr)->ref_count   = 1;
  (*rstr)->num_elem    = num_elem;
  (*rstr)->elem_size   = elem_size;
  (*rstr)->num_comp    = num_comp;
  (*rstr)->l_size      = l_size;
  (*rstr)->num_blk     = num_blk;
  (*rstr)->blk_size    = blk_size;
  (*rstr)->is_oriented = 0;
  CeedCall(CeedMalloc(3, &(*rstr)->strides));
  for (CeedInt i = 0; i < 3; i++) (*rstr)->strides[i] = strides[i];
  CeedCall(ceed->ElemRestrictionCreateBlocked(CEED_MEM_HOST, CEED_OWN_POINTER, NULL, *rstr));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Copy the pointer to a CeedElemRestriction. Both pointers should
           be destroyed with `CeedElemRestrictionDestroy()`;
           Note: If `*rstr_copy` is non-NULL, then it is assumed that
           `*rstr_copy` is a pointer to a CeedElemRestriction. This
           CeedElemRestriction will be destroyed if `*rstr_copy` is the
           only reference to this CeedElemRestriction.

  @param rstr            CeedElemRestriction to copy reference to
  @param[out] rstr_copy  Variable to store copied reference

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionReferenceCopy(CeedElemRestriction rstr, CeedElemRestriction *rstr_copy) {
  CeedCall(CeedElemRestrictionReference(rstr));
  CeedCall(CeedElemRestrictionDestroy(rstr_copy));
  *rstr_copy = rstr;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create CeedVectors associated with a CeedElemRestriction

  @param rstr   CeedElemRestriction
  @param l_vec  The address of the L-vector to be created, or NULL
  @param e_vec  The address of the E-vector to be created, or NULL

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionCreateVector(CeedElemRestriction rstr, CeedVector *l_vec, CeedVector *e_vec) {
  CeedSize e_size, l_size;
  l_size = rstr->l_size;
  e_size = rstr->num_blk * rstr->blk_size * rstr->elem_size * rstr->num_comp;
  if (l_vec) CeedCall(CeedVectorCreate(rstr->ceed, l_size, l_vec));
  if (e_vec) CeedCall(CeedVectorCreate(rstr->ceed, e_size, e_vec));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restrict an L-vector to an E-vector or apply its transpose

  @param rstr    CeedElemRestriction
  @param t_mode  Apply restriction or transpose
  @param u       Input vector (of size @a l_size when t_mode=@ref CEED_NOTRANSPOSE)
  @param ru      Output vector (of shape [@a num_elem * @a elem_size] when
                   t_mode=@ref CEED_NOTRANSPOSE). Ordering of the e-vector is decided
                   by the backend.
  @param request Request or @ref CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionApply(CeedElemRestriction rstr, CeedTransposeMode t_mode, CeedVector u, CeedVector ru, CeedRequest *request) {
  CeedInt m, n;

  if (t_mode == CEED_NOTRANSPOSE) {
    m = rstr->num_blk * rstr->blk_size * rstr->elem_size * rstr->num_comp;
    n = rstr->l_size;
  } else {
    m = rstr->l_size;
    n = rstr->num_blk * rstr->blk_size * rstr->elem_size * rstr->num_comp;
  }
  if (n != u->length) {
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, CEED_ERROR_DIMENSION,
                     "Input vector size %" CeedInt_FMT " not compatible with element restriction (%" CeedInt_FMT ", %" CeedInt_FMT ")", u->length, m,
                     n);
    // LCOV_EXCL_STOP
  }
  if (m != ru->length) {
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, CEED_ERROR_DIMENSION,
                     "Output vector size %" CeedInt_FMT " not compatible with element restriction (%" CeedInt_FMT ", %" CeedInt_FMT ")", ru->length,
                     m, n);
    // LCOV_EXCL_STOP
  }
  if (rstr->num_elem > 0) CeedCall(rstr->Apply(rstr, t_mode, u, ru, request));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restrict an L-vector to a block of an E-vector or apply its transpose

  @param rstr    CeedElemRestriction
  @param block   Block number to restrict to/from, i.e. block=0 will handle
                   elements [0 : blk_size] and block=3 will handle elements
                   [3*blk_size : 4*blk_size]
  @param t_mode  Apply restriction or transpose
  @param u       Input vector (of size @a l_size when t_mode=@ref CEED_NOTRANSPOSE)
  @param ru      Output vector (of shape [@a blk_size * @a elem_size] when
                   t_mode=@ref CEED_NOTRANSPOSE). Ordering of the e-vector is decided
                   by the backend.
  @param request Request or @ref CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionApplyBlock(CeedElemRestriction rstr, CeedInt block, CeedTransposeMode t_mode, CeedVector u, CeedVector ru,
                                  CeedRequest *request) {
  CeedInt m, n;

  if (t_mode == CEED_NOTRANSPOSE) {
    m = rstr->blk_size * rstr->elem_size * rstr->num_comp;
    n = rstr->l_size;
  } else {
    m = rstr->l_size;
    n = rstr->blk_size * rstr->elem_size * rstr->num_comp;
  }
  if (n != u->length) {
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, CEED_ERROR_DIMENSION,
                     "Input vector size %" CeedInt_FMT " not compatible with element restriction (%" CeedInt_FMT ", %" CeedInt_FMT ")", u->length, m,
                     n);
    // LCOV_EXCL_STOP
  }
  if (m != ru->length) {
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, CEED_ERROR_DIMENSION,
                     "Output vector size %" CeedInt_FMT " not compatible with element restriction (%" CeedInt_FMT ", %" CeedInt_FMT ")", ru->length,
                     m, n);
    // LCOV_EXCL_STOP
  }
  if (rstr->blk_size * block > rstr->num_elem) {
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, CEED_ERROR_DIMENSION,
                     "Cannot retrieve block %" CeedInt_FMT ", element %" CeedInt_FMT " > total elements %" CeedInt_FMT "", block,
                     rstr->blk_size * block, rstr->num_elem);
    // LCOV_EXCL_STOP
  }
  CeedCall(rstr->ApplyBlock(rstr, block, t_mode, u, ru, request));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the Ceed associated with a CeedElemRestriction

  @param rstr       CeedElemRestriction
  @param[out] ceed  Variable to store Ceed

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedElemRestrictionGetCeed(CeedElemRestriction rstr, Ceed *ceed) {
  *ceed = rstr->ceed;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the L-vector component stride

  @param rstr              CeedElemRestriction
  @param[out] comp_stride  Variable to store component stride

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedElemRestrictionGetCompStride(CeedElemRestriction rstr, CeedInt *comp_stride) {
  *comp_stride = rstr->comp_stride;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the total number of elements in the range of a CeedElemRestriction

  @param rstr           CeedElemRestriction
  @param[out] num_elem  Variable to store number of elements

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedElemRestrictionGetNumElements(CeedElemRestriction rstr, CeedInt *num_elem) {
  *num_elem = rstr->num_elem;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the size of elements in the CeedElemRestriction

  @param rstr            CeedElemRestriction
  @param[out] elem_size  Variable to store size of elements

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedElemRestrictionGetElementSize(CeedElemRestriction rstr, CeedInt *elem_size) {
  *elem_size = rstr->elem_size;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the size of the l-vector for a CeedElemRestriction

  @param rstr         CeedElemRestriction
  @param[out] l_size  Variable to store number of nodes

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedElemRestrictionGetLVectorSize(CeedElemRestriction rstr, CeedSize *l_size) {
  *l_size = rstr->l_size;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the number of components in the elements of a
         CeedElemRestriction

  @param rstr           CeedElemRestriction
  @param[out] num_comp  Variable to store number of components

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedElemRestrictionGetNumComponents(CeedElemRestriction rstr, CeedInt *num_comp) {
  *num_comp = rstr->num_comp;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the number of blocks in a CeedElemRestriction

  @param rstr            CeedElemRestriction
  @param[out] num_block  Variable to store number of blocks

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedElemRestrictionGetNumBlocks(CeedElemRestriction rstr, CeedInt *num_block) {
  *num_block = rstr->num_blk;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the size of blocks in the CeedElemRestriction

  @param rstr           CeedElemRestriction
  @param[out] blk_size  Variable to store size of blocks

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedElemRestrictionGetBlockSize(CeedElemRestriction rstr, CeedInt *blk_size) {
  *blk_size = rstr->blk_size;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the multiplicity of nodes in a CeedElemRestriction

  @param rstr       CeedElemRestriction
  @param[out] mult  Vector to store multiplicity (of size l_size)

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionGetMultiplicity(CeedElemRestriction rstr, CeedVector mult) {
  CeedVector e_vec;

  // Create e_vec to hold intermediate computation in E^T (E 1)
  CeedCall(CeedElemRestrictionCreateVector(rstr, NULL, &e_vec));

  // Compute e_vec = E * 1
  CeedCall(CeedVectorSetValue(mult, 1.0));
  CeedCall(CeedElemRestrictionApply(rstr, CEED_NOTRANSPOSE, mult, e_vec, CEED_REQUEST_IMMEDIATE));
  // Compute multiplicity, mult = E^T * e_vec = E^T (E 1)
  CeedCall(CeedVectorSetValue(mult, 0.0));
  CeedCall(CeedElemRestrictionApply(rstr, CEED_TRANSPOSE, e_vec, mult, CEED_REQUEST_IMMEDIATE));
  // Cleanup
  CeedCall(CeedVectorDestroy(&e_vec));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief View a CeedElemRestriction

  @param[in] rstr    CeedElemRestriction to view
  @param[in] stream  Stream to write; typically stdout/stderr or a file

  @return Error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionView(CeedElemRestriction rstr, FILE *stream) {
  char stridesstr[500];
  if (rstr->strides) {
    sprintf(stridesstr, "[%" CeedInt_FMT ", %" CeedInt_FMT ", %" CeedInt_FMT "]", rstr->strides[0], rstr->strides[1], rstr->strides[2]);
  } else {
    sprintf(stridesstr, "%" CeedInt_FMT, rstr->comp_stride);
  }

  fprintf(stream, "%sCeedElemRestriction from (%td, %" CeedInt_FMT ") to %" CeedInt_FMT " elements with %" CeedInt_FMT " nodes each and %s %s\n",
          rstr->blk_size > 1 ? "Blocked " : "", rstr->l_size, rstr->num_comp, rstr->num_elem, rstr->elem_size,
          rstr->strides ? "strides" : "component stride", stridesstr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Destroy a CeedElemRestriction

  @param rstr  CeedElemRestriction to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionDestroy(CeedElemRestriction *rstr) {
  if (!*rstr || --(*rstr)->ref_count > 0) return CEED_ERROR_SUCCESS;
  if ((*rstr)->num_readers) {
    // LCOV_EXCL_START
    return CeedError((*rstr)->ceed, CEED_ERROR_ACCESS, "Cannot destroy CeedElemRestriction, a process has read access to the offset data");
    // LCOV_EXCL_STOP
  }
  if ((*rstr)->Destroy) CeedCall((*rstr)->Destroy(*rstr));
  CeedCall(CeedFree(&(*rstr)->strides));
  CeedCall(CeedDestroy(&(*rstr)->ceed));
  CeedCall(CeedFree(rstr));
  return CEED_ERROR_SUCCESS;
}

/// @}
