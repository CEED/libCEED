// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed-impl.h>
#include <ceed.h>
#include <ceed/backend.h>

/// @file
/// Implementation of CeedObject functionality

/// ----------------------------------------------------------------------------
/// CeedObject Backend API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedBackend
/// @{

/**
  @brief Create a `CeedObject`

  @param[in]  ceed          `Ceed` object to reference
  @param[in]  view_function `Ceed*` function for viewing the `obj`
  @param[out] obj           Address of the variable where is `CeedObject` exists

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedObjectCreate(Ceed ceed, int (*view_function)(CeedObject, FILE *), CeedObject obj) {
  obj->ceed = NULL;
  if (ceed) CeedCall(CeedReferenceCopy(ceed, &obj->ceed));
  obj->ViewFunction = view_function;
  obj->ref_count    = 1;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Increment the reference counter for a `CeedObject`

  @param[in,out] obj `CeedObject` to increment the reference counter

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedObjectReference(CeedObject obj) {
  obj->ref_count++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Decrement the reference counter for a `CeedObject`

  @param[in,out] obj `CeedObject` to decrement the reference counter

  @return The new reference count

  @ref Backend
**/
int CeedObjectDereference(CeedObject obj) {
  return --obj->ref_count;  // prefix notation, to get new number of references
}

/**
  @brief Destroy a @ref CeedObject

  @param[in,out] obj `CeedObject` to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedObjectDestroy(CeedObject obj) {
  CeedCheck(obj->ref_count == 0, CeedObjectReturnCeed(obj), CEED_ERROR_ACCESS, "Cannot destroy CeedObject, it is still referenced by another object");
  if (obj->ceed) CeedCall(CeedDestroy(&obj->ceed));
  return CEED_ERROR_SUCCESS;
}

/// @}

/// ----------------------------------------------------------------------------
/// CeedObject Public API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedUser
/// @{

/**
  @brief View a `CeedObject`

  @param[in] obj    `CeedObject` to view
  @param[in] stream Stream to view to, e.g., `stdout`

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedObjectView(CeedObject obj, FILE *stream) {
  if (obj->ViewFunction) CeedCall(obj->ViewFunction(obj, stream));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set the number of tabs to indent for @ref CeedObjectView() output

  @param[in] obj      `CeedObject` to set the number of view tabs
  @param[in] num_tabs Number of view tabs to set

  @return Error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedObjectSetNumViewTabs(CeedObject obj, CeedInt num_tabs) {
  CeedCheck(num_tabs >= 0, CeedObjectReturnCeed(obj), CEED_ERROR_MINOR, "Number of view tabs must be non-negative");
  obj->num_view_tabs = num_tabs;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the number of tabs to indent for @ref CeedObjectView() output

  @param[in]  obj      `CeedObject` to get the number of view tabs
  @param[out] num_tabs Number of view tabs

  @return Error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedObjectGetNumViewTabs(CeedObject obj, CeedInt *num_tabs) {
  *num_tabs = obj->num_view_tabs;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the `Ceed` associated with a `CeedObject`

  @param[in]  obj   `CeedObject`
  @param[out] ceed  Variable to store `Ceed`

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedObjectGetCeed(CeedObject obj, Ceed *ceed) {
  *ceed = NULL;
  CeedCall(CeedReferenceCopy(CeedObjectReturnCeed(obj), ceed));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Return the `Ceed` associated with a `CeedObject`

  @param[in] obj `CeedObject`

  @return `Ceed` associated with the `basis`

  @ref Advanced
**/
Ceed CeedObjectReturnCeed(CeedObject obj) { return (obj->ceed) ? obj->ceed : (Ceed)obj; }

/// @}
