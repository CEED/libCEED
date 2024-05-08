// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed-impl.h>
#include <ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

/// @file
/// Implementation of CeedElemRestriction interfaces

/// ----------------------------------------------------------------------------
/// CeedElemRestriction Library Internal Functions
/// ----------------------------------------------------------------------------
/// @addtogroup CeedElemRestrictionDeveloper
/// @{

/**
  @brief Permute and pad offsets for a blocked `CeedElemRestriction`

  @param[in]  offsets       Array of shape `[num_elem, elem_size]`
  @param[out] block_offsets Array of permuted and padded array values of shape `[num_block, elem_size, block_size]`
  @param[in]  num_block     Number of blocks
  @param[in]  num_elem      Number of elements
  @param[in]  block_size    Number of elements in a block
  @param[in]  elem_size     Size of each element

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedPermutePadOffsets(const CeedInt *offsets, CeedInt *block_offsets, CeedInt num_block, CeedInt num_elem, CeedInt block_size,
                          CeedInt elem_size) {
  for (CeedInt e = 0; e < num_block * block_size; e += block_size) {
    for (CeedInt j = 0; j < block_size; j++) {
      for (CeedInt k = 0; k < elem_size; k++) {
        block_offsets[e * elem_size + k * block_size + j] = offsets[CeedIntMin(e + j, num_elem - 1) * elem_size + k];
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Permute and pad orientations for a blocked `CeedElemRestriction`

  @param[in]  orients       Array of shape `[num_elem, elem_size]`
  @param[out] block_orients Array of permuted and padded array values of shape `[num_block, elem_size, block_size]`
  @param[in]  num_block     Number of blocks
  @param[in]  num_elem      Number of elements
  @param[in]  block_size    Number of elements in a block
  @param[in]  elem_size     Size of each element

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedPermutePadOrients(const bool *orients, bool *block_orients, CeedInt num_block, CeedInt num_elem, CeedInt block_size, CeedInt elem_size) {
  for (CeedInt e = 0; e < num_block * block_size; e += block_size) {
    for (CeedInt j = 0; j < block_size; j++) {
      for (CeedInt k = 0; k < elem_size; k++) {
        block_orients[e * elem_size + k * block_size + j] = orients[CeedIntMin(e + j, num_elem - 1) * elem_size + k];
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Permute and pad curl-conforming orientations for a blocked `CeedElemRestriction`

  @param[in]  curl_orients       Array of shape `[num_elem, elem_size]`
  @param[out] block_curl_orients Array of permuted and padded array values of shape `[num_block, elem_size, block_size]`
  @param[in]  num_block          Number of blocks
  @param[in]  num_elem           Number of elements
  @param[in]  block_size         Number of elements in a block
  @param[in]  elem_size          Size of each element

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedPermutePadCurlOrients(const CeedInt8 *curl_orients, CeedInt8 *block_curl_orients, CeedInt num_block, CeedInt num_elem, CeedInt block_size,
                              CeedInt elem_size) {
  for (CeedInt e = 0; e < num_block * block_size; e += block_size) {
    for (CeedInt j = 0; j < block_size; j++) {
      for (CeedInt k = 0; k < elem_size; k++) {
        block_curl_orients[e * elem_size + k * block_size + j] = curl_orients[CeedIntMin(e + j, num_elem - 1) * elem_size + k];
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
  @brief Get the type of a `CeedElemRestriction`

  @param[in]  rstr      `CeedElemRestriction`
  @param[out] rstr_type Variable to store restriction type

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionGetType(CeedElemRestriction rstr, CeedRestrictionType *rstr_type) {
  *rstr_type = rstr->rstr_type;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the strided status of a `CeedElemRestriction`

  @param[in]  rstr       `CeedElemRestriction`
  @param[out] is_strided Variable to store strided status

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionIsStrided(CeedElemRestriction rstr, bool *is_strided) {
  *is_strided = (rstr->rstr_type == CEED_RESTRICTION_STRIDED);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the points status of a `CeedElemRestriction`

  @param[in]  rstr      `CeedElemRestriction`
  @param[out] is_points Variable to store points status

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionIsPoints(CeedElemRestriction rstr, bool *is_points) {
  *is_points = (rstr->rstr_type == CEED_RESTRICTION_POINTS);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Check if two `CeedElemRestriction` created with @ref CeedElemRestrictionCreateAtPoints() and use the same points per element

  @param[in]  rstr_a         First `CeedElemRestriction`
  @param[in]  rstr_b         Second `CeedElemRestriction`
  @param[out] are_compatible Variable to store compatibility status

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionAtPointsAreCompatible(CeedElemRestriction rstr_a, CeedElemRestriction rstr_b, bool *are_compatible) {
  CeedInt num_elem_a, num_elem_b, num_points_a, num_points_b;
  Ceed    ceed;

  CeedCall(CeedElemRestrictionGetCeed(rstr_a, &ceed));

  // Cannot compare non-points restrictions
  CeedCheck(rstr_a->rstr_type == CEED_RESTRICTION_POINTS, ceed, CEED_ERROR_UNSUPPORTED, "First CeedElemRestriction must be AtPoints");
  CeedCheck(rstr_b->rstr_type == CEED_RESTRICTION_POINTS, ceed, CEED_ERROR_UNSUPPORTED, "Second CeedElemRestriction must be AtPoints");

  CeedCall(CeedElemRestrictionGetNumElements(rstr_a, &num_elem_a));
  CeedCall(CeedElemRestrictionGetNumElements(rstr_b, &num_elem_b));
  CeedCall(CeedElemRestrictionGetNumPoints(rstr_a, &num_points_a));
  CeedCall(CeedElemRestrictionGetNumPoints(rstr_b, &num_points_b));

  // Check size and contents of offsets arrays
  *are_compatible = true;
  if (num_elem_a != num_elem_b) *are_compatible = false;
  if (num_points_a != num_points_b) *are_compatible = false;
  if (*are_compatible) {
    const CeedInt *offsets_a, *offsets_b;

    CeedCall(CeedElemRestrictionGetOffsets(rstr_a, CEED_MEM_HOST, &offsets_a));
    CeedCall(CeedElemRestrictionGetOffsets(rstr_b, CEED_MEM_HOST, &offsets_b));
    for (CeedInt i = 0; i < num_elem_a + 1 + num_points_a; i++) *are_compatible &= offsets_a[i] == offsets_b[i];
    CeedCall(CeedElemRestrictionRestoreOffsets(rstr_a, &offsets_a));
    CeedCall(CeedElemRestrictionRestoreOffsets(rstr_b, &offsets_b));
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the strides of a strided `CeedElemRestriction`

  @param[in]  rstr    `CeedElemRestriction`
  @param[out] strides Variable to store strides array

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionGetStrides(CeedElemRestriction rstr, CeedInt strides[3]) {
  CeedCheck(rstr->strides, CeedElemRestrictionReturnCeed(rstr), CEED_ERROR_MINOR, "CeedElemRestriction has no stride data");
  for (CeedInt i = 0; i < 3; i++) strides[i] = rstr->strides[i];
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the backend stride status of a `CeedElemRestriction`

  @param[in]  rstr                 `CeedElemRestriction`
  @param[out] has_backend_strides  Variable to store stride status

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionHasBackendStrides(CeedElemRestriction rstr, bool *has_backend_strides) {
  CeedCheck(rstr->strides, CeedElemRestrictionReturnCeed(rstr), CEED_ERROR_MINOR, "CeedElemRestriction has no stride data");
  *has_backend_strides = ((rstr->strides[0] == CEED_STRIDES_BACKEND[0]) && (rstr->strides[1] == CEED_STRIDES_BACKEND[1]) &&
                          (rstr->strides[2] == CEED_STRIDES_BACKEND[2]));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get read-only access to a `CeedElemRestriction` offsets array by @ref CeedMemType

  @param[in]  rstr     `CeedElemRestriction` to retrieve offsets
  @param[in]  mem_type Memory type on which to access the array.
                         If the backend uses a different memory type, this will perform a copy (possibly cached).
  @param[out] offsets  Array on memory type `mem_type`

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionGetOffsets(CeedElemRestriction rstr, CeedMemType mem_type, const CeedInt **offsets) {
  if (rstr->rstr_base) {
    CeedCall(CeedElemRestrictionGetOffsets(rstr->rstr_base, mem_type, offsets));
  } else {
    CeedCheck(rstr->GetOffsets, CeedElemRestrictionReturnCeed(rstr), CEED_ERROR_UNSUPPORTED,
              "Backend does not implement CeedElemRestrictionGetOffsets");
    CeedCall(rstr->GetOffsets(rstr, mem_type, offsets));
    rstr->num_readers++;
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restore an offsets array obtained using @ref CeedElemRestrictionGetOffsets()

  @param[in] rstr    `CeedElemRestriction` to restore
  @param[in] offsets Array of offset data

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionRestoreOffsets(CeedElemRestriction rstr, const CeedInt **offsets) {
  if (rstr->rstr_base) {
    CeedCall(CeedElemRestrictionRestoreOffsets(rstr->rstr_base, offsets));
  } else {
    *offsets = NULL;
    rstr->num_readers--;
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get read-only access to a `CeedElemRestriction` orientations array by @ref CeedMemType

  @param[in]  rstr     `CeedElemRestriction` to retrieve orientations
  @param[in]  mem_type Memory type on which to access the array.
                         If the backend uses a different memory type, this will perform a copy (possibly cached).
  @param[out] orients  Array on memory type `mem_type`

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionGetOrientations(CeedElemRestriction rstr, CeedMemType mem_type, const bool **orients) {
  CeedCheck(rstr->GetOrientations, CeedElemRestrictionReturnCeed(rstr), CEED_ERROR_UNSUPPORTED,
            "Backend does not implement CeedElemRestrictionGetOrientations");
  CeedCall(rstr->GetOrientations(rstr, mem_type, orients));
  rstr->num_readers++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restore an orientations array obtained using @ref CeedElemRestrictionGetOrientations()

  @param[in] rstr    `CeedElemRestriction` to restore
  @param[in] orients Array of orientation data

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionRestoreOrientations(CeedElemRestriction rstr, const bool **orients) {
  *orients = NULL;
  rstr->num_readers--;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get read-only access to a `CeedElemRestriction` curl-conforming orientations array by @ref CeedMemType

  @param[in]  rstr         `CeedElemRestriction` to retrieve curl-conforming orientations
  @param[in]  mem_type     Memory type on which to access the array.
                             If the backend uses a different memory type, this will perform a copy (possibly cached).
  @param[out] curl_orients Array on memory type `mem_type`

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionGetCurlOrientations(CeedElemRestriction rstr, CeedMemType mem_type, const CeedInt8 **curl_orients) {
  CeedCheck(rstr->GetCurlOrientations, CeedElemRestrictionReturnCeed(rstr), CEED_ERROR_UNSUPPORTED,
            "Backend does not implement CeedElemRestrictionGetCurlOrientations");
  CeedCall(rstr->GetCurlOrientations(rstr, mem_type, curl_orients));
  rstr->num_readers++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restore an orientations array obtained using @ref CeedElemRestrictionGetCurlOrientations()

  @param[in] rstr         `CeedElemRestriction` to restore
  @param[in] curl_orients Array of orientation data

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionRestoreCurlOrientations(CeedElemRestriction rstr, const CeedInt8 **curl_orients) {
  *curl_orients = NULL;
  rstr->num_readers--;
  return CEED_ERROR_SUCCESS;
}

/**

  @brief Get the L-vector layout of a strided `CeedElemRestriction`

  @param[in]  rstr    `CeedElemRestriction`
  @param[out] layout  Variable to store layout array, stored as `[nodes, components, elements]`.
                        The data for node `i`, component `j`, element `k` in the E-vector is given by `i*layout[0] + j*layout[1] + k*layout[2]`.

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionGetLLayout(CeedElemRestriction rstr, CeedInt layout[3]) {
  bool                has_backend_strides;
  CeedRestrictionType rstr_type;
  Ceed                ceed;

  CeedCall(CeedElemRestrictionGetCeed(rstr, &ceed));
  CeedCall(CeedElemRestrictionGetType(rstr, &rstr_type));
  CeedCheck(rstr_type == CEED_RESTRICTION_STRIDED, ceed, CEED_ERROR_MINOR, "Only strided CeedElemRestriction have strided L-vector layout");
  CeedCall(CeedElemRestrictionHasBackendStrides(rstr, &has_backend_strides));
  if (has_backend_strides) {
    CeedCheck(rstr->l_layout[0], ceed, CEED_ERROR_MINOR, "CeedElemRestriction has no L-vector layout data");
    for (CeedInt i = 0; i < 3; i++) layout[i] = rstr->l_layout[i];
  } else {
    CeedCall(CeedElemRestrictionGetStrides(rstr, layout));
  }
  return CEED_ERROR_SUCCESS;
}

/**

  @brief Set the L-vector layout of a strided `CeedElemRestriction`

  @param[in] rstr   `CeedElemRestriction`
  @param[in] layout Variable to containing layout array, stored as `[nodes, components, elements]`.
                      The data for node `i`, component `j`, element `k` in the E-vector is given by `i*layout[0] + j*layout[1] + k*layout[2]`.

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionSetLLayout(CeedElemRestriction rstr, CeedInt layout[3]) {
  CeedRestrictionType rstr_type;

  CeedCall(CeedElemRestrictionGetType(rstr, &rstr_type));
  CeedCheck(rstr_type == CEED_RESTRICTION_STRIDED, CeedElemRestrictionReturnCeed(rstr), CEED_ERROR_MINOR,
            "Only strided CeedElemRestriction have strided L-vector layout");
  for (CeedInt i = 0; i < 3; i++) rstr->l_layout[i] = layout[i];
  return CEED_ERROR_SUCCESS;
}

/**

  @brief Get the E-vector layout of a `CeedElemRestriction`

  @param[in]  rstr    `CeedElemRestriction`
  @param[out] layout  Variable to store layout array, stored as `[nodes, components, elements]`.
                        The data for node `i`, component `j`, element `k` in the E-vector is given by `i*layout[0] + j*layout[1] + k*layout[2]`.

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionGetELayout(CeedElemRestriction rstr, CeedInt layout[3]) {
  CeedCheck(rstr->e_layout[0], CeedElemRestrictionReturnCeed(rstr), CEED_ERROR_MINOR, "CeedElemRestriction has no E-vector layout data");
  for (CeedInt i = 0; i < 3; i++) layout[i] = rstr->e_layout[i];
  return CEED_ERROR_SUCCESS;
}

/**

  @brief Set the E-vector layout of a `CeedElemRestriction`

  @param[in] rstr   `CeedElemRestriction`
  @param[in] layout Variable to containing layout array, stored as `[nodes, components, elements]`.
                      The data for node `i`, component `j`, element `k` in the E-vector is given by `i*layout[0] + j*layout[1] + k*layout[2]`.

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionSetELayout(CeedElemRestriction rstr, CeedInt layout[3]) {
  for (CeedInt i = 0; i < 3; i++) rstr->e_layout[i] = layout[i];
  return CEED_ERROR_SUCCESS;
}

/**

  @brief Get the E-vector element offset of a `CeedElemRestriction` at points

  @param[in]  rstr        `CeedElemRestriction`
  @param[in]  elem        Element number index into E-vector for
  @param[out] elem_offset Offset for element `elem` in the E-vector.
                            The data for point `i`, component `j`, element `elem` in the E-vector is given by `i*e_layout[0] + j*e_layout[1] + elem_offset`.

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionGetAtPointsElementOffset(CeedElemRestriction rstr, CeedInt elem, CeedSize *elem_offset) {
  CeedInt             num_comp;
  CeedRestrictionType rstr_type;

  CeedCall(CeedElemRestrictionGetType(rstr, &rstr_type));
  CeedCheck(rstr_type == CEED_RESTRICTION_POINTS, CeedElemRestrictionReturnCeed(rstr), CEED_ERROR_INCOMPATIBLE,
            "Can only compute offset for a points CeedElemRestriction");

  // Backend method
  if (rstr->GetAtPointsElementOffset) {
    CeedCall(rstr->GetAtPointsElementOffset(rstr, elem, elem_offset));
    return CEED_ERROR_SUCCESS;
  }

  // Default layout (CPU)
  *elem_offset = 0;
  CeedCall(CeedElemRestrictionGetNumComponents(rstr, &num_comp));
  for (CeedInt i = 0; i < elem; i++) {
    CeedInt num_points;

    CeedCall(CeedElemRestrictionGetNumPointsInElement(rstr, i, &num_points));
    *elem_offset += num_points * num_comp;
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the backend data of a `CeedElemRestriction`

  @param[in]  rstr `CeedElemRestriction`
  @param[out] data Variable to store data

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionGetData(CeedElemRestriction rstr, void *data) {
  *(void **)data = rstr->data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set the backend data of a `CeedElemRestriction`

  @param[in,out] rstr `CeedElemRestriction`
  @param[in]     data Data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionSetData(CeedElemRestriction rstr, void *data) {
  rstr->data = data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Increment the reference counter for a `CeedElemRestriction`

  @param[in,out] rstr `CeedElemRestriction` to increment the reference counter

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionReference(CeedElemRestriction rstr) {
  rstr->ref_count++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Estimate number of FLOPs required to apply `CeedElemRestriction` in `t_mode`

  @param[in]  rstr   `CeedElemRestriction` to estimate FLOPs for
  @param[in]  t_mode Apply restriction or transpose
  @param[out] flops  Address of variable to hold FLOPs estimate

  @ref Backend
**/
int CeedElemRestrictionGetFlopsEstimate(CeedElemRestriction rstr, CeedTransposeMode t_mode, CeedSize *flops) {
  CeedSize            e_size, scale = 0;
  CeedRestrictionType rstr_type;

  CeedCall(CeedElemRestrictionGetEVectorSize(rstr, &e_size));
  CeedCall(CeedElemRestrictionGetType(rstr, &rstr_type));
  if (t_mode == CEED_TRANSPOSE) {
    switch (rstr_type) {
      case CEED_RESTRICTION_POINTS:
        scale = 0;
        break;
      case CEED_RESTRICTION_STRIDED:
      case CEED_RESTRICTION_STANDARD:
        scale = 1;
        break;
      case CEED_RESTRICTION_ORIENTED:
        scale = 2;
        break;
      case CEED_RESTRICTION_CURL_ORIENTED:
        scale = 6;
        break;
    }
  } else {
    switch (rstr_type) {
      case CEED_RESTRICTION_STRIDED:
      case CEED_RESTRICTION_STANDARD:
      case CEED_RESTRICTION_POINTS:
        scale = 0;
        break;
      case CEED_RESTRICTION_ORIENTED:
        scale = 1;
        break;
      case CEED_RESTRICTION_CURL_ORIENTED:
        scale = 5;
        break;
    }
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

/// Argument for @ref CeedOperatorSetField() indicating that the field does not require a `CeedElemRestriction`
const CeedElemRestriction CEED_ELEMRESTRICTION_NONE = &ceed_elemrestriction_none;

/**
  @brief Create a `CeedElemRestriction`

  @param[in]  ceed        `Ceed` context used to create the `CeedElemRestriction`
  @param[in]  num_elem    Number of elements described in the `offsets` array
  @param[in]  elem_size   Size (number of "nodes") per element
  @param[in]  num_comp    Number of field components per interpolation node (1 for scalar fields)
  @param[in]  comp_stride Stride between components for the same L-vector "node".
                            Data for node `i`, component `j`, element `k` can be found in the L-vector at index `offsets[i + k*elem_size] + j*comp_stride`.
  @param[in]  l_size      The size of the L-vector.
                            This vector may be larger than the elements and fields given by this restriction.
  @param[in]  mem_type    Memory type of the `offsets` array, see @ref CeedMemType
  @param[in]  copy_mode   Copy mode for the `offsets` array, see @ref CeedCopyMode
  @param[in]  offsets     Array of shape `[num_elem, elem_size]`.
                            Row `i` holds the ordered list of the offsets (into the input `CeedVector`) for the unknowns corresponding to element `i`, where 0 <= i < @a num_elem.
                            All offsets must be in the range `[0, l_size - 1]`.
  @param[out] rstr        Address of the variable where the newly created `CeedElemRestriction` will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionCreate(Ceed ceed, CeedInt num_elem, CeedInt elem_size, CeedInt num_comp, CeedInt comp_stride, CeedSize l_size,
                              CeedMemType mem_type, CeedCopyMode copy_mode, const CeedInt *offsets, CeedElemRestriction *rstr) {
  if (!ceed->ElemRestrictionCreate) {
    Ceed delegate;

    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "ElemRestriction"));
    CeedCheck(delegate, ceed, CEED_ERROR_UNSUPPORTED, "Backend does not implement CeedElemRestrictionCreate");
    CeedCall(CeedElemRestrictionCreate(delegate, num_elem, elem_size, num_comp, comp_stride, l_size, mem_type, copy_mode, offsets, rstr));
    return CEED_ERROR_SUCCESS;
  }

  CeedCheck(num_elem >= 0, ceed, CEED_ERROR_DIMENSION, "Number of elements must be non-negative");
  CeedCheck(elem_size > 0, ceed, CEED_ERROR_DIMENSION, "Element size must be at least 1");
  CeedCheck(num_comp > 0, ceed, CEED_ERROR_DIMENSION, "CeedElemRestriction must have at least 1 component");
  CeedCheck(num_comp == 1 || comp_stride > 0, ceed, CEED_ERROR_DIMENSION, "CeedElemRestriction component stride must be at least 1");

  CeedCall(CeedCalloc(1, rstr));
  CeedCall(CeedReferenceCopy(ceed, &(*rstr)->ceed));
  (*rstr)->ref_count   = 1;
  (*rstr)->num_elem    = num_elem;
  (*rstr)->elem_size   = elem_size;
  (*rstr)->num_comp    = num_comp;
  (*rstr)->comp_stride = comp_stride;
  (*rstr)->l_size      = l_size;
  (*rstr)->e_size      = (CeedSize)num_elem * (CeedSize)elem_size * (CeedSize)num_comp;
  (*rstr)->num_block   = num_elem;
  (*rstr)->block_size  = 1;
  (*rstr)->rstr_type   = CEED_RESTRICTION_STANDARD;
  CeedCall(ceed->ElemRestrictionCreate(mem_type, copy_mode, offsets, NULL, NULL, *rstr));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a `CeedElemRestriction` with orientation signs

  @param[in]  ceed        `Ceed` context used to create the `CeedElemRestriction`
  @param[in]  num_elem    Number of elements described in the `offsets` array
  @param[in]  elem_size   Size (number of "nodes") per element
  @param[in]  num_comp    Number of field components per interpolation node (1 for scalar fields)
  @param[in]  comp_stride Stride between components for the same L-vector "node".
                            Data for node `i`, component `j`, element `k` can be found in the L-vector at index `offsets[i + k*elem_size] + j*comp_stride`.
  @param[in]  l_size      The size of the L-vector.
                            This vector may be larger than the elements and fields given by this restriction.
  @param[in]  mem_type    Memory type of the `offsets` array, see @ref CeedMemType
  @param[in]  copy_mode   Copy mode for the `offsets` array, see @ref CeedCopyMode
  @param[in]  offsets     Array of shape `[num_elem, elem_size]`.
                            Row i holds the ordered list of the offsets (into the input `CeedVector`) for the unknowns corresponding to element `i`, where `0 <= i < num_elem`.
                            All offsets must be in the range `[0, l_size - 1]`.
  @param[in]  orients     Boolean array of shape `[num_elem, elem_size]` with `false` for positively oriented and `true` to flip the orientation
  @param[out] rstr        Address of the variable where the newly created `CeedElemRestriction` will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionCreateOriented(Ceed ceed, CeedInt num_elem, CeedInt elem_size, CeedInt num_comp, CeedInt comp_stride, CeedSize l_size,
                                      CeedMemType mem_type, CeedCopyMode copy_mode, const CeedInt *offsets, const bool *orients,
                                      CeedElemRestriction *rstr) {
  if (!ceed->ElemRestrictionCreate) {
    Ceed delegate;

    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "ElemRestriction"));
    CeedCheck(delegate, ceed, CEED_ERROR_UNSUPPORTED, "Backend does not implement CeedElemRestrictionCreateOriented");
    CeedCall(
        CeedElemRestrictionCreateOriented(delegate, num_elem, elem_size, num_comp, comp_stride, l_size, mem_type, copy_mode, offsets, orients, rstr));
    return CEED_ERROR_SUCCESS;
  }

  CeedCheck(num_elem >= 0, ceed, CEED_ERROR_DIMENSION, "Number of elements must be non-negative");
  CeedCheck(elem_size > 0, ceed, CEED_ERROR_DIMENSION, "Element size must be at least 1");
  CeedCheck(num_comp > 0, ceed, CEED_ERROR_DIMENSION, "CeedElemRestriction must have at least 1 component");
  CeedCheck(num_comp == 1 || comp_stride > 0, ceed, CEED_ERROR_DIMENSION, "CeedElemRestriction component stride must be at least 1");

  CeedCall(CeedCalloc(1, rstr));
  CeedCall(CeedReferenceCopy(ceed, &(*rstr)->ceed));
  (*rstr)->ref_count   = 1;
  (*rstr)->num_elem    = num_elem;
  (*rstr)->elem_size   = elem_size;
  (*rstr)->num_comp    = num_comp;
  (*rstr)->comp_stride = comp_stride;
  (*rstr)->l_size      = l_size;
  (*rstr)->e_size      = (CeedSize)num_elem * (CeedSize)elem_size * (CeedSize)num_comp;
  (*rstr)->num_block   = num_elem;
  (*rstr)->block_size  = 1;
  (*rstr)->rstr_type   = CEED_RESTRICTION_ORIENTED;
  CeedCall(ceed->ElemRestrictionCreate(mem_type, copy_mode, offsets, orients, NULL, *rstr));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a `CeedElemRestriction` with a general tridiagonal transformation matrix for curl-conforming elements

  @param[in]  ceed         `Ceed` context used to create the `CeedElemRestriction`
  @param[in]  num_elem     Number of elements described in the `offsets` array
  @param[in]  elem_size    Size (number of "nodes") per element
  @param[in]  num_comp     Number of field components per interpolation node (1 for scalar fields)
  @param[in]  comp_stride  Stride between components for the same L-vector "node".
                             Data for node `i`, component `j`, element `k` can be found in the L-vector at index `offsets[i + k*elem_size] + j*comp_stride`.
  @param[in]  l_size       The size of the L-vector.
                             This vector may be larger than the elements and fields given by this restriction.
  @param[in]  mem_type     Memory type of the `offsets` array, see @ref CeedMemType
  @param[in]  copy_mode    Copy mode for the `offsets` array, see @ref CeedCopyMode
  @param[in]  offsets      Array of shape `[num_elem, elem_size]`.
                             Row `i` holds the ordered list of the offsets (into the input `CeedVector`) for the unknowns corresponding to element `i`, where `0 <= i < num_elem`.
                             All offsets must be in the range `[0, l_size - 1]`.
  @param[in]  curl_orients Array of shape `[num_elem, 3 * elem_size]` representing a row-major tridiagonal matrix (`curl_orients[i * 3 * elem_size] = curl_orients[(i + 1) * 3 * elem_size - 1] = 0`, where `0 <= i < num_elem`) which is applied to the element unknowns upon restriction.
                             This orientation matrix allows for pairs of face degrees of freedom on elements for \f$H(\mathrm{curl})\f$ spaces to be coupled in the element restriction operation, which is a way to resolve face orientation issues for 3D meshes (https://dl.acm.org/doi/pdf/10.1145/3524456).
  @param[out] rstr         Address of the variable where the newly created `CeedElemRestriction` will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionCreateCurlOriented(Ceed ceed, CeedInt num_elem, CeedInt elem_size, CeedInt num_comp, CeedInt comp_stride, CeedSize l_size,
                                          CeedMemType mem_type, CeedCopyMode copy_mode, const CeedInt *offsets, const CeedInt8 *curl_orients,
                                          CeedElemRestriction *rstr) {
  if (!ceed->ElemRestrictionCreate) {
    Ceed delegate;

    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "ElemRestriction"));
    CeedCheck(delegate, ceed, CEED_ERROR_UNSUPPORTED, "Backend does not implement CeedElemRestrictionCreateCurlOriented");
    CeedCall(CeedElemRestrictionCreateCurlOriented(delegate, num_elem, elem_size, num_comp, comp_stride, l_size, mem_type, copy_mode, offsets,
                                                   curl_orients, rstr));
    return CEED_ERROR_SUCCESS;
  }

  CeedCheck(num_elem >= 0, ceed, CEED_ERROR_DIMENSION, "Number of elements must be non-negative");
  CeedCheck(elem_size > 0, ceed, CEED_ERROR_DIMENSION, "Element size must be at least 1");
  CeedCheck(num_comp > 0, ceed, CEED_ERROR_DIMENSION, "CeedElemRestriction must have at least 1 component");
  CeedCheck(num_comp == 1 || comp_stride > 0, ceed, CEED_ERROR_DIMENSION, "CeedElemRestriction component stride must be at least 1");

  CeedCall(CeedCalloc(1, rstr));
  CeedCall(CeedReferenceCopy(ceed, &(*rstr)->ceed));
  (*rstr)->ref_count   = 1;
  (*rstr)->num_elem    = num_elem;
  (*rstr)->elem_size   = elem_size;
  (*rstr)->num_comp    = num_comp;
  (*rstr)->comp_stride = comp_stride;
  (*rstr)->l_size      = l_size;
  (*rstr)->e_size      = (CeedSize)num_elem * (CeedSize)elem_size * (CeedSize)num_comp;
  (*rstr)->num_block   = num_elem;
  (*rstr)->block_size  = 1;
  (*rstr)->rstr_type   = CEED_RESTRICTION_CURL_ORIENTED;
  CeedCall(ceed->ElemRestrictionCreate(mem_type, copy_mode, offsets, NULL, curl_orients, *rstr));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a strided `CeedElemRestriction`

  @param[in]  ceed      `Ceed` context used to create the `CeedElemRestriction`
  @param[in]  num_elem  Number of elements described by the restriction
  @param[in]  elem_size Size (number of "nodes") per element
  @param[in]  num_comp  Number of field components per interpolation "node" (1 for scalar fields)
  @param[in]  l_size    The size of the L-vector.
                          This vector may be larger than the elements and fields given by this restriction.
  @param[in]  strides   Array for strides between `[nodes, components, elements]`.
                          Data for node `i`, component `j`, element `k` can be found in the L-vector at index `i*strides[0] + j*strides[1] + k*strides[2]`.
                          @ref CEED_STRIDES_BACKEND may be used for `CeedVector` ordered by the same `Ceed` backend.
                          `CEED_STRIDES_BACKEND` should only be used pass data between `CeedOperator` created with the same `Ceed` backend.
                          The L-vector layout will, in general, be different between `Ceed` backends.
  @param[out] rstr      Address of the variable where the newly created `CeedElemRestriction` will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionCreateStrided(Ceed ceed, CeedInt num_elem, CeedInt elem_size, CeedInt num_comp, CeedSize l_size, const CeedInt strides[3],
                                     CeedElemRestriction *rstr) {
  if (!ceed->ElemRestrictionCreate) {
    Ceed delegate;

    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "ElemRestriction"));
    CeedCheck(delegate, ceed, CEED_ERROR_UNSUPPORTED, "Backend does not implement CeedElemRestrictionCreateStrided");
    CeedCall(CeedElemRestrictionCreateStrided(delegate, num_elem, elem_size, num_comp, l_size, strides, rstr));
    return CEED_ERROR_SUCCESS;
  }

  CeedCheck(num_elem >= 0, ceed, CEED_ERROR_DIMENSION, "Number of elements must be non-negative");
  CeedCheck(elem_size > 0, ceed, CEED_ERROR_DIMENSION, "Element size must be at least 1");
  CeedCheck(num_comp > 0, ceed, CEED_ERROR_DIMENSION, "CeedElemRestriction must have at least 1 component");
  CeedCheck(l_size >= (CeedSize)num_elem * elem_size * num_comp, ceed, CEED_ERROR_DIMENSION,
            "L-vector size must be at least num_elem * elem_size * num_comp");

  CeedCall(CeedCalloc(1, rstr));
  CeedCall(CeedReferenceCopy(ceed, &(*rstr)->ceed));
  (*rstr)->ref_count  = 1;
  (*rstr)->num_elem   = num_elem;
  (*rstr)->elem_size  = elem_size;
  (*rstr)->num_comp   = num_comp;
  (*rstr)->l_size     = l_size;
  (*rstr)->e_size     = (CeedSize)num_elem * (CeedSize)elem_size * (CeedSize)num_comp;
  (*rstr)->num_block  = num_elem;
  (*rstr)->block_size = 1;
  (*rstr)->rstr_type  = CEED_RESTRICTION_STRIDED;
  CeedCall(CeedMalloc(3, &(*rstr)->strides));
  for (CeedInt i = 0; i < 3; i++) (*rstr)->strides[i] = strides[i];
  CeedCall(ceed->ElemRestrictionCreate(CEED_MEM_HOST, CEED_OWN_POINTER, NULL, NULL, NULL, *rstr));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a points `CeedElemRestriction`, for restricting for restricting from a all local points to the current element in which they are located.

  The offsets array is arranged as

  element_0_start_index
  element_1_start_index
  ...
  element_n_start_index
  element_n_stop_index
  element_0_point_0
  element_0_point_1
  ...

  @param[in]  ceed       `Ceed` context used to create the `CeedElemRestriction`
  @param[in]  num_elem   Number of elements described in the `offsets` array
  @param[in]  num_points Number of points described in the `offsets` array
  @param[in]  num_comp   Number of field components per interpolation node (1 for scalar fields).
                           Components are assumed to be contiguous by point.
  @param[in]  l_size     The size of the L-vector.
                           This vector may be larger than the elements and fields given by this restriction.
  @param[in]  mem_type   Memory type of the `offsets` array, see @ref CeedMemType
  @param[in]  copy_mode  Copy mode for the `offsets` array, see @ref CeedCopyMode
  @param[in]  offsets    Array of size `num_elem + 1 + num_points`.
                           The first portion of the offsets array holds the ranges of indices corresponding to each element.
                           The second portion holds the indices for each element.
  @param[out] rstr       Address of the variable where the newly created `CeedElemRestriction` will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
 **/
int CeedElemRestrictionCreateAtPoints(Ceed ceed, CeedInt num_elem, CeedInt num_points, CeedInt num_comp, CeedSize l_size, CeedMemType mem_type,
                                      CeedCopyMode copy_mode, const CeedInt *offsets, CeedElemRestriction *rstr) {
  if (!ceed->ElemRestrictionCreateAtPoints) {
    Ceed delegate;

    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "ElemRestriction"));
    CeedCheck(delegate, ceed, CEED_ERROR_UNSUPPORTED, "Backend does not implement CeedElemRestrictionCreateAtPoints");
    CeedCall(CeedElemRestrictionCreateAtPoints(delegate, num_elem, num_points, num_comp, l_size, mem_type, copy_mode, offsets, rstr));
    return CEED_ERROR_SUCCESS;
  }

  CeedCheck(num_elem >= 0, ceed, CEED_ERROR_DIMENSION, "Number of elements must be non-negative");
  CeedCheck(num_points >= 0, ceed, CEED_ERROR_DIMENSION, "Number of points must be non-negative");
  CeedCheck(num_comp > 0, ceed, CEED_ERROR_DIMENSION, "CeedElemRestriction must have at least 1 component");
  CeedCheck(l_size >= (CeedSize)num_points * num_comp, ceed, CEED_ERROR_DIMENSION, "L-vector must be at least num_points * num_comp");

  CeedCall(CeedCalloc(1, rstr));
  CeedCall(CeedReferenceCopy(ceed, &(*rstr)->ceed));
  (*rstr)->ref_count   = 1;
  (*rstr)->num_elem    = num_elem;
  (*rstr)->num_points  = num_points;
  (*rstr)->num_comp    = num_comp;
  (*rstr)->comp_stride = 1;
  (*rstr)->l_size      = l_size;
  (*rstr)->e_size      = (CeedSize)num_points * (CeedSize)num_comp;
  (*rstr)->num_block   = num_elem;
  (*rstr)->block_size  = 1;
  (*rstr)->rstr_type   = CEED_RESTRICTION_POINTS;
  CeedCall(ceed->ElemRestrictionCreateAtPoints(mem_type, copy_mode, offsets, NULL, NULL, *rstr));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a blocked `CeedElemRestriction`, typically only used by backends

  @param[in]  ceed        `Ceed` context used to create the `CeedElemRestriction`
  @param[in]  num_elem    Number of elements described in the `offsets` array
  @param[in]  elem_size   Size (number of unknowns) per element
  @param[in]  block_size  Number of elements in a block
  @param[in]  num_comp    Number of field components per interpolation node (1 for scalar fields)
  @param[in]  comp_stride Stride between components for the same L-vector "node".
                            Data for node `i`, component `j`, element `k` can be found in the L-vector at index `offsets[i + k*elem_size] + j*comp_stride`.
  @param[in]  l_size      The size of the L-vector.
                            This vector may be larger than the elements and fields given by this restriction.
  @param[in]  mem_type    Memory type of the `offsets` array, see @ref CeedMemType
  @param[in]  copy_mode   Copy mode for the `offsets` array, see @ref CeedCopyMode
  @param[in]  offsets     Array of shape `[num_elem, elem_size]`.
                            Row `i` holds the ordered list of the offsets (into the input `CeedVector`) for the unknowns corresponding to element `i`, where `0 <= i < num_elem`.
                            All offsets must be in the range `[0, l_size - 1]`.
                            The backend will permute and pad this array to the desired ordering for the blocksize, which is typically given by the backend.
                            The default reordering is to interlace elements.
  @param[out] rstr        Address of the variable where the newly created `CeedElemRestriction` will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
 **/
int CeedElemRestrictionCreateBlocked(Ceed ceed, CeedInt num_elem, CeedInt elem_size, CeedInt block_size, CeedInt num_comp, CeedInt comp_stride,
                                     CeedSize l_size, CeedMemType mem_type, CeedCopyMode copy_mode, const CeedInt *offsets,
                                     CeedElemRestriction *rstr) {
  CeedInt *block_offsets, num_block = (num_elem / block_size) + !!(num_elem % block_size);

  if (!ceed->ElemRestrictionCreateBlocked) {
    Ceed delegate;

    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "ElemRestriction"));
    CeedCheck(delegate, ceed, CEED_ERROR_UNSUPPORTED, "Backend does not implement CeedElemRestrictionCreateBlocked");
    CeedCall(CeedElemRestrictionCreateBlocked(delegate, num_elem, elem_size, block_size, num_comp, comp_stride, l_size, mem_type, copy_mode, offsets,
                                              rstr));
    return CEED_ERROR_SUCCESS;
  }

  CeedCheck(num_elem >= 0, ceed, CEED_ERROR_DIMENSION, "Number of elements must be non-negative");
  CeedCheck(elem_size > 0, ceed, CEED_ERROR_DIMENSION, "Element size must be at least 1");
  CeedCheck(block_size > 0, ceed, CEED_ERROR_DIMENSION, "Block size must be at least 1");
  CeedCheck(num_comp > 0, ceed, CEED_ERROR_DIMENSION, "CeedElemRestriction must have at least 1 component");
  CeedCheck(num_comp == 1 || comp_stride > 0, ceed, CEED_ERROR_DIMENSION, "CeedElemRestriction component stride must be at least 1");

  CeedCall(CeedCalloc(num_block * block_size * elem_size, &block_offsets));
  CeedCall(CeedPermutePadOffsets(offsets, block_offsets, num_block, num_elem, block_size, elem_size));

  CeedCall(CeedCalloc(1, rstr));
  CeedCall(CeedReferenceCopy(ceed, &(*rstr)->ceed));
  (*rstr)->ref_count   = 1;
  (*rstr)->num_elem    = num_elem;
  (*rstr)->elem_size   = elem_size;
  (*rstr)->num_comp    = num_comp;
  (*rstr)->comp_stride = comp_stride;
  (*rstr)->l_size      = l_size;
  (*rstr)->e_size      = (CeedSize)num_block * (CeedSize)block_size * (CeedSize)elem_size * (CeedSize)num_comp;
  (*rstr)->num_block   = num_block;
  (*rstr)->block_size  = block_size;
  (*rstr)->rstr_type   = CEED_RESTRICTION_STANDARD;
  CeedCall(ceed->ElemRestrictionCreateBlocked(CEED_MEM_HOST, CEED_OWN_POINTER, (const CeedInt *)block_offsets, NULL, NULL, *rstr));
  if (copy_mode == CEED_OWN_POINTER) CeedCall(CeedFree(&offsets));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a blocked oriented `CeedElemRestriction`, typically only used by backends

  @param[in]  ceed        `Ceed` context used to create the `CeedElemRestriction`
  @param[in]  num_elem    Number of elements described in the `offsets` array.
  @param[in]  elem_size   Size (number of unknowns) per element
  @param[in]  block_size  Number of elements in a block
  @param[in]  num_comp    Number of field components per interpolation node (1 for scalar fields)
  @param[in]  comp_stride Stride between components for the same L-vector "node".
                            Data for node `i`, component `j`, element `k` can be found in the L-vector at index `offsets[i + k*elem_size] + j*comp_stride`.
  @param[in]  l_size      The size of the L-vector.
                            This vector may be larger than the elements and fields given by this restriction.
  @param[in]  mem_type    Memory type of the `offsets` array, see @ref CeedMemType
  @param[in]  copy_mode   Copy mode for the `offsets` array, see @ref CeedCopyMode
  @param[in]  offsets     Array of shape `[num_elem, elem_size]`.
                            Row `i` holds the ordered list of the offsets (into the input `CeedVector`) for the unknowns corresponding to element `i`, where `0 <= i < num_elem`.
                            All offsets must be in the range `[0, l_size - 1]`.
                            The backend will permute and pad this array to the desired ordering for the blocksize, which is typically given by the backend.
                            The default reordering is to interlace elements.
  @param[in]  orients     Boolean array of shape `[num_elem, elem_size]` with `false` for positively oriented and `true` to flip the orientation.
                            Will also be permuted and padded similarly to `offsets`.
  @param[out] rstr        Address of the variable where the newly created `CeedElemRestriction` will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
 **/
int CeedElemRestrictionCreateBlockedOriented(Ceed ceed, CeedInt num_elem, CeedInt elem_size, CeedInt block_size, CeedInt num_comp,
                                             CeedInt comp_stride, CeedSize l_size, CeedMemType mem_type, CeedCopyMode copy_mode,
                                             const CeedInt *offsets, const bool *orients, CeedElemRestriction *rstr) {
  bool    *block_orients;
  CeedInt *block_offsets, num_block = (num_elem / block_size) + !!(num_elem % block_size);

  if (!ceed->ElemRestrictionCreateBlocked) {
    Ceed delegate;

    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "ElemRestriction"));
    CeedCheck(delegate, ceed, CEED_ERROR_UNSUPPORTED, "Backend does not implement CeedElemRestrictionCreateBlockedOriented");
    CeedCall(CeedElemRestrictionCreateBlockedOriented(delegate, num_elem, elem_size, block_size, num_comp, comp_stride, l_size, mem_type, copy_mode,
                                                      offsets, orients, rstr));
    return CEED_ERROR_SUCCESS;
  }

  CeedCheck(elem_size > 0, ceed, CEED_ERROR_DIMENSION, "Element size must be at least 1");
  CeedCheck(block_size > 0, ceed, CEED_ERROR_DIMENSION, "Block size must be at least 1");
  CeedCheck(num_comp > 0, ceed, CEED_ERROR_DIMENSION, "CeedElemRestriction must have at least 1 component");
  CeedCheck(num_comp == 1 || comp_stride > 0, ceed, CEED_ERROR_DIMENSION, "CeedElemRestriction component stride must be at least 1");

  CeedCall(CeedCalloc(num_block * block_size * elem_size, &block_offsets));
  CeedCall(CeedCalloc(num_block * block_size * elem_size, &block_orients));
  CeedCall(CeedPermutePadOffsets(offsets, block_offsets, num_block, num_elem, block_size, elem_size));
  CeedCall(CeedPermutePadOrients(orients, block_orients, num_block, num_elem, block_size, elem_size));

  CeedCall(CeedCalloc(1, rstr));
  CeedCall(CeedReferenceCopy(ceed, &(*rstr)->ceed));
  (*rstr)->ref_count   = 1;
  (*rstr)->num_elem    = num_elem;
  (*rstr)->elem_size   = elem_size;
  (*rstr)->num_comp    = num_comp;
  (*rstr)->comp_stride = comp_stride;
  (*rstr)->l_size      = l_size;
  (*rstr)->e_size      = (CeedSize)num_block * (CeedSize)block_size * (CeedSize)elem_size * (CeedSize)num_comp;
  (*rstr)->num_block   = num_block;
  (*rstr)->block_size  = block_size;
  (*rstr)->rstr_type   = CEED_RESTRICTION_ORIENTED;
  CeedCall(
      ceed->ElemRestrictionCreateBlocked(CEED_MEM_HOST, CEED_OWN_POINTER, (const CeedInt *)block_offsets, (const bool *)block_orients, NULL, *rstr));
  if (copy_mode == CEED_OWN_POINTER) CeedCall(CeedFree(&offsets));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a blocked curl-oriented `CeedElemRestriction`, typically only used by backends

  @param[in]  ceed         `Ceed` context used to create the `CeedElemRestriction`
  @param[in]  num_elem     Number of elements described in the `offsets` array.
  @param[in]  elem_size    Size (number of unknowns) per element
  @param[in]  block_size   Number of elements in a block
  @param[in]  num_comp     Number of field components per interpolation node (1 for scalar fields)
  @param[in]  comp_stride  Stride between components for the same L-vector "node".
                             Data for node `i`, component `j`, element `k` can be found in the L-vector at index `offsets[i + k*elem_size] + j*comp_stride`.
  @param[in]  l_size       The size of the L-vector.
                             This vector may be larger than the elements and fields given by this restriction.
  @param[in]  mem_type     Memory type of the `offsets` array, see @ref CeedMemType
  @param[in]  copy_mode    Copy mode for the `offsets` array, see @ref CeedCopyMode
  @param[in]  offsets      Array of shape `[num_elem, elem_size]`.
                             Row `i` holds the ordered list of the offsets (into the input `CeedVector`) for the unknowns corresponding to element `i`, where `0 <= i < num_elem`.
                             All offsets must be in the range `[0, l_size - 1]`.
                             The backend will permute and pad this array to the desired  ordering for the blocksize, which is typically given by the backend.
                             The default reordering is to interlace elements.
  @param[in]  curl_orients Array of shape `[num_elem, 3 * elem_size]` representing a row-major tridiagonal matrix (`curl_orients[i * 3 * elem_size] = curl_orients[(i + 1) * 3 * elem_size - 1] = 0`, where `0 <= i < num_elem`) which is applied to the element unknowns upon restriction.
                             This orientation matrix allows for pairs of face degrees of freedom on elements for \f$H(\mathrm{curl})\f$ spaces to be coupled in the element restriction  operation, which is a way to resolve face orientation issues for 3D meshes (https://dl.acm.org/doi/pdf/10.1145/3524456).
                             Will also be permuted and padded similarly to offsets.
  @param[out] rstr         Address of the variable where the newly created `CeedElemRestriction` will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
 **/
int CeedElemRestrictionCreateBlockedCurlOriented(Ceed ceed, CeedInt num_elem, CeedInt elem_size, CeedInt block_size, CeedInt num_comp,
                                                 CeedInt comp_stride, CeedSize l_size, CeedMemType mem_type, CeedCopyMode copy_mode,
                                                 const CeedInt *offsets, const CeedInt8 *curl_orients, CeedElemRestriction *rstr) {
  CeedInt8 *block_curl_orients;
  CeedInt  *block_offsets, num_block = (num_elem / block_size) + !!(num_elem % block_size);

  if (!ceed->ElemRestrictionCreateBlocked) {
    Ceed delegate;

    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "ElemRestriction"));
    CeedCheck(delegate, ceed, CEED_ERROR_UNSUPPORTED, "Backend does not implement CeedElemRestrictionCreateBlockedCurlOriented");
    CeedCall(CeedElemRestrictionCreateBlockedCurlOriented(delegate, num_elem, elem_size, block_size, num_comp, comp_stride, l_size, mem_type,
                                                          copy_mode, offsets, curl_orients, rstr));
    return CEED_ERROR_SUCCESS;
  }

  CeedCheck(num_elem >= 0, ceed, CEED_ERROR_DIMENSION, "Number of elements must be non-negative");
  CeedCheck(elem_size > 0, ceed, CEED_ERROR_DIMENSION, "Element size must be at least 1");
  CeedCheck(block_size > 0, ceed, CEED_ERROR_DIMENSION, "Block size must be at least 1");
  CeedCheck(num_comp > 0, ceed, CEED_ERROR_DIMENSION, "CeedElemRestriction must have at least 1 component");
  CeedCheck(num_comp == 1 || comp_stride > 0, ceed, CEED_ERROR_DIMENSION, "CeedElemRestriction component stride must be at least 1");

  CeedCall(CeedCalloc(num_block * block_size * elem_size, &block_offsets));
  CeedCall(CeedCalloc(num_block * block_size * 3 * elem_size, &block_curl_orients));
  CeedCall(CeedPermutePadOffsets(offsets, block_offsets, num_block, num_elem, block_size, elem_size));
  CeedCall(CeedPermutePadCurlOrients(curl_orients, block_curl_orients, num_block, num_elem, block_size, 3 * elem_size));

  CeedCall(CeedCalloc(1, rstr));
  CeedCall(CeedReferenceCopy(ceed, &(*rstr)->ceed));
  (*rstr)->ref_count   = 1;
  (*rstr)->num_elem    = num_elem;
  (*rstr)->elem_size   = elem_size;
  (*rstr)->num_comp    = num_comp;
  (*rstr)->comp_stride = comp_stride;
  (*rstr)->l_size      = l_size;
  (*rstr)->e_size      = (CeedSize)num_block * (CeedSize)block_size * (CeedSize)elem_size * (CeedSize)num_comp;
  (*rstr)->num_block   = num_block;
  (*rstr)->block_size  = block_size;
  (*rstr)->rstr_type   = CEED_RESTRICTION_CURL_ORIENTED;
  CeedCall(ceed->ElemRestrictionCreateBlocked(CEED_MEM_HOST, CEED_OWN_POINTER, (const CeedInt *)block_offsets, NULL,
                                              (const CeedInt8 *)block_curl_orients, *rstr));
  if (copy_mode == CEED_OWN_POINTER) CeedCall(CeedFree(&offsets));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a blocked strided `CeedElemRestriction`, typically only used by backends

  @param[in]  ceed        `Ceed` context used to create the `CeedElemRestriction`
  @param[in]  num_elem    Number of elements described by the restriction
  @param[in]  elem_size   Size (number of "nodes") per element
  @param[in]  block_size  Number of elements in a block
  @param[in]  num_comp    Number of field components per interpolation node (1 for scalar fields)
  @param[in]  l_size      The size of the L-vector.
                            This vector may be larger than the elements and fields given by this restriction.
  @param[in]  strides     Array for strides between `[nodes, components, elements]`.
                            Data for node `i`, component `j`, element `k` can be found in the L-vector at index `i*strides[0] + j*strides[1] +k*strides[2]`.
                            @ref CEED_STRIDES_BACKEND may be used for `CeedVector` ordered by the same `Ceed` backend.
  @param[out] rstr        Address of the variable where the newly created `CeedElemRestriction` will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionCreateBlockedStrided(Ceed ceed, CeedInt num_elem, CeedInt elem_size, CeedInt block_size, CeedInt num_comp, CeedSize l_size,
                                            const CeedInt strides[3], CeedElemRestriction *rstr) {
  CeedInt num_block = (num_elem / block_size) + !!(num_elem % block_size);

  if (!ceed->ElemRestrictionCreateBlocked) {
    Ceed delegate;

    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "ElemRestriction"));
    CeedCheck(delegate, ceed, CEED_ERROR_UNSUPPORTED, "Backend does not implement CeedElemRestrictionCreateBlockedStrided");
    CeedCall(CeedElemRestrictionCreateBlockedStrided(delegate, num_elem, elem_size, block_size, num_comp, l_size, strides, rstr));
    return CEED_ERROR_SUCCESS;
  }

  CeedCheck(num_elem >= 0, ceed, CEED_ERROR_DIMENSION, "Number of elements must be non-negative");
  CeedCheck(elem_size > 0, ceed, CEED_ERROR_DIMENSION, "Element size must be at least 1");
  CeedCheck(block_size > 0, ceed, CEED_ERROR_DIMENSION, "Block size must be at least 1");
  CeedCheck(num_comp > 0, ceed, CEED_ERROR_DIMENSION, "CeedElemRestriction must have at least 1 component");
  CeedCheck(l_size >= (CeedSize)num_elem * elem_size * num_comp, ceed, CEED_ERROR_DIMENSION,
            "L-vector size must be at least num_elem * elem_size * num_comp");

  CeedCall(CeedCalloc(1, rstr));
  CeedCall(CeedReferenceCopy(ceed, &(*rstr)->ceed));
  (*rstr)->ref_count  = 1;
  (*rstr)->num_elem   = num_elem;
  (*rstr)->elem_size  = elem_size;
  (*rstr)->num_comp   = num_comp;
  (*rstr)->l_size     = l_size;
  (*rstr)->e_size     = (CeedSize)num_block * (CeedSize)block_size * (CeedSize)elem_size * (CeedSize)num_comp;
  (*rstr)->num_block  = num_block;
  (*rstr)->block_size = block_size;
  (*rstr)->rstr_type  = CEED_RESTRICTION_STRIDED;
  CeedCall(CeedMalloc(3, &(*rstr)->strides));
  for (CeedInt i = 0; i < 3; i++) (*rstr)->strides[i] = strides[i];
  CeedCall(ceed->ElemRestrictionCreateBlocked(CEED_MEM_HOST, CEED_OWN_POINTER, NULL, NULL, NULL, *rstr));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Copy the pointer to a `CeedElemRestriction` and set @ref CeedElemRestrictionApply() implementation to use the unsigned version.

  Both pointers should be destroyed with @ref CeedElemRestrictionDestroy().

  @param[in]     rstr          `CeedElemRestriction` to create unsigned reference to
  @param[in,out] rstr_unsigned Variable to store unsigned `CeedElemRestriction`

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionCreateUnsignedCopy(CeedElemRestriction rstr, CeedElemRestriction *rstr_unsigned) {
  CeedCall(CeedCalloc(1, rstr_unsigned));

  // Copy old rstr
  memcpy(*rstr_unsigned, rstr, sizeof(struct CeedElemRestriction_private));
  (*rstr_unsigned)->ceed = NULL;
  CeedCall(CeedReferenceCopy(rstr->ceed, &(*rstr_unsigned)->ceed));
  (*rstr_unsigned)->ref_count = 1;
  (*rstr_unsigned)->strides   = NULL;
  if (rstr->strides) {
    CeedCall(CeedMalloc(3, &(*rstr_unsigned)->strides));
    for (CeedInt i = 0; i < 3; i++) (*rstr_unsigned)->strides[i] = rstr->strides[i];
  }
  CeedCall(CeedElemRestrictionReferenceCopy(rstr, &(*rstr_unsigned)->rstr_base));

  // Override Apply
  (*rstr_unsigned)->Apply = rstr->ApplyUnsigned;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Copy the pointer to a `CeedElemRestriction` and set @ref CeedElemRestrictionApply() implementation to use the unoriented version.

  Both pointers should be destroyed with @ref CeedElemRestrictionDestroy().

  @param[in]     rstr            `CeedElemRestriction` to create unoriented reference to
  @param[in,out] rstr_unoriented Variable to store unoriented `CeedElemRestriction`

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionCreateUnorientedCopy(CeedElemRestriction rstr, CeedElemRestriction *rstr_unoriented) {
  CeedCall(CeedCalloc(1, rstr_unoriented));

  // Copy old rstr
  memcpy(*rstr_unoriented, rstr, sizeof(struct CeedElemRestriction_private));
  (*rstr_unoriented)->ceed = NULL;
  CeedCall(CeedReferenceCopy(rstr->ceed, &(*rstr_unoriented)->ceed));
  (*rstr_unoriented)->ref_count = 1;
  (*rstr_unoriented)->strides   = NULL;
  if (rstr->strides) {
    CeedCall(CeedMalloc(3, &(*rstr_unoriented)->strides));
    for (CeedInt i = 0; i < 3; i++) (*rstr_unoriented)->strides[i] = rstr->strides[i];
  }
  CeedCall(CeedElemRestrictionReferenceCopy(rstr, &(*rstr_unoriented)->rstr_base));

  // Override Apply
  (*rstr_unoriented)->Apply = rstr->ApplyUnoriented;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Copy the pointer to a `CeedElemRestriction`.

  Both pointers should be destroyed with @ref CeedElemRestrictionDestroy().

  Note: If the value of `*rstr_copy` passed into this function is non-`NULL`, then it is assumed that `*rstr_copy` is a pointer to a `CeedElemRestriction`.
        This `CeedElemRestriction` will be destroyed if `*rstr_copy` is the only reference to this `CeedElemRestriction`.

  @param[in]     rstr      `CeedElemRestriction` to copy reference to
  @param[in,out] rstr_copy Variable to store copied reference

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionReferenceCopy(CeedElemRestriction rstr, CeedElemRestriction *rstr_copy) {
  if (rstr != CEED_ELEMRESTRICTION_NONE) CeedCall(CeedElemRestrictionReference(rstr));
  CeedCall(CeedElemRestrictionDestroy(rstr_copy));
  *rstr_copy = rstr;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create `CeedVector` associated with a `CeedElemRestriction`

  @param[in]  rstr  `CeedElemRestriction`
  @param[out] l_vec The address of the L-vector to be created, or `NULL`
  @param[out] e_vec The address of the E-vector to be created, or `NULL`

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionCreateVector(CeedElemRestriction rstr, CeedVector *l_vec, CeedVector *e_vec) {
  CeedSize e_size, l_size;
  Ceed     ceed;

  CeedCall(CeedElemRestrictionGetCeed(rstr, &ceed));
  CeedCall(CeedElemRestrictionGetLVectorSize(rstr, &l_size));
  CeedCall(CeedElemRestrictionGetEVectorSize(rstr, &e_size));
  if (l_vec) CeedCall(CeedVectorCreate(ceed, l_size, l_vec));
  if (e_vec) CeedCall(CeedVectorCreate(ceed, e_size, e_vec));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restrict an L-vector to an E-vector or apply its transpose

  @param[in]  rstr    `CeedElemRestriction`
  @param[in]  t_mode  Apply restriction or transpose
  @param[in]  u       Input vector (of size `l_size` when `t_mode` = @ref CEED_NOTRANSPOSE)
  @param[out] ru      Output vector (of shape `[num_elem * elem_size]` when `t_mode` = @ref CEED_NOTRANSPOSE).
                        Ordering of the e-vector is decided by the backend.
  @param[in]  request Request or @ref CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionApply(CeedElemRestriction rstr, CeedTransposeMode t_mode, CeedVector u, CeedVector ru, CeedRequest *request) {
  CeedSize min_u_len, min_ru_len, len;
  CeedInt  num_elem;
  Ceed     ceed;

  CeedCall(CeedElemRestrictionGetCeed(rstr, &ceed));
  if (t_mode == CEED_NOTRANSPOSE) {
    CeedCall(CeedElemRestrictionGetEVectorSize(rstr, &min_ru_len));
    CeedCall(CeedElemRestrictionGetLVectorSize(rstr, &min_u_len));
  } else {
    CeedCall(CeedElemRestrictionGetEVectorSize(rstr, &min_u_len));
    CeedCall(CeedElemRestrictionGetLVectorSize(rstr, &min_ru_len));
  }
  CeedCall(CeedVectorGetLength(u, &len));
  CeedCheck(min_u_len <= len, ceed, CEED_ERROR_DIMENSION,
            "Input vector size %" CeedInt_FMT " not compatible with element restriction (%" CeedInt_FMT ", %" CeedInt_FMT ")", len, min_ru_len,
            min_u_len);
  CeedCall(CeedVectorGetLength(ru, &len));
  CeedCheck(min_ru_len <= len, ceed, CEED_ERROR_DIMENSION,
            "Output vector size %" CeedInt_FMT " not compatible with element restriction (%" CeedInt_FMT ", %" CeedInt_FMT ")", len, min_u_len,
            min_ru_len);
  CeedCall(CeedElemRestrictionGetNumElements(rstr, &num_elem));
  if (num_elem > 0) CeedCall(rstr->Apply(rstr, t_mode, u, ru, request));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restrict an L-vector of points to a single element or apply its transpose

  @param[in]  rstr    `CeedElemRestriction`
  @param[in]  elem    Element number in range `[0, num_elem)`
  @param[in]  t_mode  Apply restriction or transpose
  @param[in]  u       Input vector (of size `l_size` when `t_mode` = @ref CEED_NOTRANSPOSE)
  @param[out] ru      Output vector (of shape [`num_points * num_comp]` when `t_mode` = @ref CEED_NOTRANSPOSE).
                        Ordering of the e-vector is decided by the backend.
  @param[in]  request Request or @ref CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionApplyAtPointsInElement(CeedElemRestriction rstr, CeedInt elem, CeedTransposeMode t_mode, CeedVector u, CeedVector ru,
                                              CeedRequest *request) {
  CeedSize min_u_len, min_ru_len, len;
  CeedInt  num_elem;
  Ceed     ceed;

  CeedCall(CeedElemRestrictionGetCeed(rstr, &ceed));
  CeedCheck(rstr->ApplyAtPointsInElement, ceed, CEED_ERROR_UNSUPPORTED, "Backend does not implement CeedElemRestrictionApplyAtPointsInElement");

  if (t_mode == CEED_NOTRANSPOSE) {
    CeedInt num_points, num_comp;

    CeedCall(CeedElemRestrictionGetLVectorSize(rstr, &min_u_len));
    CeedCall(CeedElemRestrictionGetNumPointsInElement(rstr, elem, &num_points));
    CeedCall(CeedElemRestrictionGetNumComponents(rstr, &num_comp));
    min_ru_len = (CeedSize)num_points * (CeedSize)num_comp;
  } else {
    CeedInt num_points, num_comp;

    CeedCall(CeedElemRestrictionGetNumPointsInElement(rstr, elem, &num_points));
    CeedCall(CeedElemRestrictionGetNumComponents(rstr, &num_comp));
    min_u_len = (CeedSize)num_points * (CeedSize)num_comp;
    CeedCall(CeedElemRestrictionGetLVectorSize(rstr, &min_ru_len));
  }
  CeedCall(CeedVectorGetLength(u, &len));
  CeedCheck(min_u_len <= len, ceed, CEED_ERROR_DIMENSION,
            "Input vector size %" CeedInt_FMT " not compatible with element restriction (%" CeedInt_FMT ", %" CeedInt_FMT
            ") for element %" CeedInt_FMT,
            len, min_ru_len, min_u_len, elem);
  CeedCall(CeedVectorGetLength(ru, &len));
  CeedCheck(min_ru_len <= len, ceed, CEED_ERROR_DIMENSION,
            "Output vector size %" CeedInt_FMT " not compatible with element restriction (%" CeedInt_FMT ", %" CeedInt_FMT
            ") for element %" CeedInt_FMT,
            len, min_ru_len, min_u_len, elem);
  CeedCall(CeedElemRestrictionGetNumElements(rstr, &num_elem));
  CeedCheck(elem < num_elem, ceed, CEED_ERROR_DIMENSION,
            "Cannot retrieve element %" CeedInt_FMT ", element %" CeedInt_FMT " > total elements %" CeedInt_FMT "", elem, elem, num_elem);
  if (num_elem > 0) CeedCall(rstr->ApplyAtPointsInElement(rstr, elem, t_mode, u, ru, request));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restrict an L-vector to a block of an E-vector or apply its transpose

  @param[in]  rstr    `CeedElemRestriction`
  @param[in]  block   Block number to restrict to/from, i.e. `block = 0` will handle elements `[0 : block_size]` and `block = 3` will handle elements `[3*block_size : 4*block_size]`
  @param[in]  t_mode  Apply restriction or transpose
  @param[in]  u       Input vector (of size `l_size` when `t_mode` = @ref CEED_NOTRANSPOSE)
  @param[out] ru      Output vector (of shape `[block_size * elem_size]` when `t_mode` = @ref CEED_NOTRANSPOSE).
                        Ordering of the e-vector is decided by the backend.
  @param[in]  request Request or @ref CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionApplyBlock(CeedElemRestriction rstr, CeedInt block, CeedTransposeMode t_mode, CeedVector u, CeedVector ru,
                                  CeedRequest *request) {
  CeedSize min_u_len, min_ru_len, len;
  CeedInt  block_size, num_elem;
  Ceed     ceed;

  CeedCall(CeedElemRestrictionGetCeed(rstr, &ceed));
  CeedCheck(rstr->ApplyBlock, ceed, CEED_ERROR_UNSUPPORTED, "Backend does not implement CeedElemRestrictionApplyBlock");

  CeedCall(CeedElemRestrictionGetBlockSize(rstr, &block_size));
  if (t_mode == CEED_NOTRANSPOSE) {
    CeedInt elem_size, num_comp;

    CeedCall(CeedElemRestrictionGetElementSize(rstr, &elem_size));
    CeedCall(CeedElemRestrictionGetNumComponents(rstr, &num_comp));
    CeedCall(CeedElemRestrictionGetLVectorSize(rstr, &min_u_len));
    min_ru_len = (CeedSize)block_size * (CeedSize)elem_size * (CeedSize)num_comp;
  } else {
    CeedInt elem_size, num_comp;

    CeedCall(CeedElemRestrictionGetElementSize(rstr, &elem_size));
    CeedCall(CeedElemRestrictionGetNumComponents(rstr, &num_comp));
    CeedCall(CeedElemRestrictionGetLVectorSize(rstr, &min_ru_len));
    min_u_len = (CeedSize)block_size * (CeedSize)elem_size * (CeedSize)num_comp;
  }
  CeedCall(CeedVectorGetLength(u, &len));
  CeedCheck(min_u_len == len, ceed, CEED_ERROR_DIMENSION,
            "Input vector size %" CeedInt_FMT " not compatible with element restriction (%" CeedInt_FMT ", %" CeedInt_FMT ")", len, min_u_len,
            min_ru_len);
  CeedCall(CeedVectorGetLength(ru, &len));
  CeedCheck(min_ru_len == len, ceed, CEED_ERROR_DIMENSION,
            "Output vector size %" CeedInt_FMT " not compatible with element restriction (%" CeedInt_FMT ", %" CeedInt_FMT ")", len, min_ru_len,
            min_u_len);
  CeedCall(CeedElemRestrictionGetNumElements(rstr, &num_elem));
  CeedCheck(block_size * block <= num_elem, ceed, CEED_ERROR_DIMENSION,
            "Cannot retrieve block %" CeedInt_FMT ", element %" CeedInt_FMT " > total elements %" CeedInt_FMT "", block, block_size * block,
            num_elem);
  CeedCall(rstr->ApplyBlock(rstr, block, t_mode, u, ru, request));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the `Ceed` associated with a `CeedElemRestriction`

  @param[in]  rstr `CeedElemRestriction`
  @param[out] ceed Variable to store `Ceed`

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedElemRestrictionGetCeed(CeedElemRestriction rstr, Ceed *ceed) {
  *ceed = CeedElemRestrictionReturnCeed(rstr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Return the `Ceed` associated with a `CeedElemRestriction`

  @param[in]  rstr `CeedElemRestriction`

  @return `Ceed` associated with the `rstr`

  @ref Advanced
**/
Ceed CeedElemRestrictionReturnCeed(CeedElemRestriction rstr) { return rstr->ceed; }

/**
  @brief Get the L-vector component stride

  @param[in]  rstr        `CeedElemRestriction`
  @param[out] comp_stride Variable to store component stride

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedElemRestrictionGetCompStride(CeedElemRestriction rstr, CeedInt *comp_stride) {
  *comp_stride = rstr->comp_stride;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the total number of elements in the range of a `CeedElemRestriction`

  @param[in] rstr      `CeedElemRestriction`
  @param[out] num_elem Variable to store number of elements

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedElemRestrictionGetNumElements(CeedElemRestriction rstr, CeedInt *num_elem) {
  *num_elem = rstr->num_elem;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the size of elements in the `CeedElemRestriction`

  @param[in]  rstr      `CeedElemRestriction`
  @param[out] elem_size Variable to store size of elements

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedElemRestrictionGetElementSize(CeedElemRestriction rstr, CeedInt *elem_size) {
  *elem_size = rstr->elem_size;
  return CEED_ERROR_SUCCESS;
}

/**

  @brief Get the number of points in the l-vector for a points `CeedElemRestriction`

  @param[in]  rstr       `CeedElemRestriction`
  @param[out] num_points The number of points in the l-vector

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionGetNumPoints(CeedElemRestriction rstr, CeedInt *num_points) {
  CeedRestrictionType rstr_type;

  CeedCall(CeedElemRestrictionGetType(rstr, &rstr_type));
  CeedCheck(rstr_type == CEED_RESTRICTION_POINTS, CeedElemRestrictionReturnCeed(rstr), CEED_ERROR_INCOMPATIBLE,
            "Can only retrieve the number of points for a points CeedElemRestriction");

  *num_points = rstr->num_points;
  return CEED_ERROR_SUCCESS;
}

/**

  @brief Get the number of points in an element of a `CeedElemRestriction` at points

  @param[in]  rstr       `CeedElemRestriction`
  @param[in]  elem       Index number of element to retrieve the number of points for
  @param[out] num_points The number of points in the element at index elem

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionGetNumPointsInElement(CeedElemRestriction rstr, CeedInt elem, CeedInt *num_points) {
  CeedRestrictionType rstr_type;
  const CeedInt      *offsets;

  CeedCall(CeedElemRestrictionGetType(rstr, &rstr_type));
  CeedCheck(rstr_type == CEED_RESTRICTION_POINTS, CeedElemRestrictionReturnCeed(rstr), CEED_ERROR_INCOMPATIBLE,
            "Can only retrieve the number of points for a points CeedElemRestriction");

  CeedCall(CeedElemRestrictionGetOffsets(rstr, CEED_MEM_HOST, &offsets));
  *num_points = offsets[elem + 1] - offsets[elem];
  CeedCall(CeedElemRestrictionRestoreOffsets(rstr, &offsets));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the maximum number of points in an element for a `CeedElemRestriction` at points

  @param[in]  rstr       `CeedElemRestriction`
  @param[out] max_points Variable to store size of elements

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedElemRestrictionGetMaxPointsInElement(CeedElemRestriction rstr, CeedInt *max_points) {
  CeedInt             num_elem;
  CeedRestrictionType rstr_type;

  CeedCall(CeedElemRestrictionGetType(rstr, &rstr_type));
  CeedCheck(rstr_type == CEED_RESTRICTION_POINTS, CeedElemRestrictionReturnCeed(rstr), CEED_ERROR_INCOMPATIBLE,
            "Cannot compute max points for a CeedElemRestriction that does not use points");

  CeedCall(CeedElemRestrictionGetNumElements(rstr, &num_elem));
  *max_points = 0;
  for (CeedInt e = 0; e < num_elem; e++) {
    CeedInt num_points;

    CeedCall(CeedElemRestrictionGetNumPointsInElement(rstr, e, &num_points));
    *max_points = CeedIntMax(num_points, *max_points);
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the size of the l-vector for a `CeedElemRestriction`

  @param[in]  rstr   `CeedElemRestriction`
  @param[out] l_size Variable to store l-vector size

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedElemRestrictionGetLVectorSize(CeedElemRestriction rstr, CeedSize *l_size) {
  *l_size = rstr->l_size;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the size of the e-vector for a `CeedElemRestriction`

  @param[in]  rstr   `CeedElemRestriction`
  @param[out] e_size Variable to store e-vector size

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedElemRestrictionGetEVectorSize(CeedElemRestriction rstr, CeedSize *e_size) {
  *e_size = rstr->e_size;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the number of components in the elements of a `CeedElemRestriction`

  @param[in]  rstr     `CeedElemRestriction`
  @param[out] num_comp Variable to store number of components

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedElemRestrictionGetNumComponents(CeedElemRestriction rstr, CeedInt *num_comp) {
  *num_comp = rstr->num_comp;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the number of blocks in a `CeedElemRestriction`

  @param[in]  rstr      `CeedElemRestriction`
  @param[out] num_block Variable to store number of blocks

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedElemRestrictionGetNumBlocks(CeedElemRestriction rstr, CeedInt *num_block) {
  *num_block = rstr->num_block;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the size of blocks in the `CeedElemRestriction`

  @param[in]  rstr       `CeedElemRestriction`
  @param[out] block_size Variable to store size of blocks

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedElemRestrictionGetBlockSize(CeedElemRestriction rstr, CeedInt *block_size) {
  *block_size = rstr->block_size;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the multiplicity of nodes in a `CeedElemRestriction`

  @param[in]  rstr `CeedElemRestriction`
  @param[out] mult Vector to store multiplicity (of size `l_size`)

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
  @brief View a `CeedElemRestriction`

  @param[in] rstr   `CeedElemRestriction` to view
  @param[in] stream Stream to write; typically `stdout` or a file

  @return Error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionView(CeedElemRestriction rstr, FILE *stream) {
  CeedRestrictionType rstr_type;

  CeedCall(CeedElemRestrictionGetType(rstr, &rstr_type));
  if (rstr_type == CEED_RESTRICTION_POINTS) {
    CeedInt max_points;

    CeedCall(CeedElemRestrictionGetMaxPointsInElement(rstr, &max_points));
    fprintf(stream,
            "CeedElemRestriction at points from (%" CeedSize_FMT ", %" CeedInt_FMT ") to %" CeedInt_FMT " elements with a maximum of %" CeedInt_FMT
            " points on an element\n",
            rstr->l_size, rstr->num_comp, rstr->num_elem, max_points);
  } else {
    char strides_str[500];

    if (rstr->strides) {
      sprintf(strides_str, "[%" CeedInt_FMT ", %" CeedInt_FMT ", %" CeedInt_FMT "]", rstr->strides[0], rstr->strides[1], rstr->strides[2]);
    } else {
      sprintf(strides_str, "%" CeedInt_FMT, rstr->comp_stride);
    }
    fprintf(stream,
            "%sCeedElemRestriction from (%" CeedSize_FMT ", %" CeedInt_FMT ") to %" CeedInt_FMT " elements with %" CeedInt_FMT
            " nodes each and %s %s\n",
            rstr->block_size > 1 ? "Blocked " : "", rstr->l_size, rstr->num_comp, rstr->num_elem, rstr->elem_size,
            rstr->strides ? "strides" : "component stride", strides_str);
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Destroy a `CeedElemRestriction`

  @param[in,out] rstr `CeedElemRestriction` to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionDestroy(CeedElemRestriction *rstr) {
  if (!*rstr || *rstr == CEED_ELEMRESTRICTION_NONE || --(*rstr)->ref_count > 0) {
    *rstr = NULL;
    return CEED_ERROR_SUCCESS;
  }
  CeedCheck((*rstr)->num_readers == 0, (*rstr)->ceed, CEED_ERROR_ACCESS,
            "Cannot destroy CeedElemRestriction, a process has read access to the offset data");

  // Only destroy backend data once between rstr and unsigned copy
  if ((*rstr)->rstr_base) CeedCall(CeedElemRestrictionDestroy(&(*rstr)->rstr_base));
  else if ((*rstr)->Destroy) CeedCall((*rstr)->Destroy(*rstr));

  CeedCall(CeedFree(&(*rstr)->strides));
  CeedCall(CeedDestroy(&(*rstr)->ceed));
  CeedCall(CeedFree(rstr));
  return CEED_ERROR_SUCCESS;
}

/// @}
