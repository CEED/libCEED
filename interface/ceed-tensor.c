// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed-impl.h>
#include <ceed/backend.h>
#include <ceed/ceed.h>

/// @file
/// Implementation of CeedTensorContract interfaces

/// ----------------------------------------------------------------------------
/// CeedTensorContract Backend API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedBasisBackend
/// @{

/**
  @brief Create a CeedTensorContract object for a CeedBasis

  @param ceed           A Ceed object where the CeedTensorContract will be created
  @param basis          CeedBasis for which the tensor contraction will be used
  @param[out] contract  Address of the variable where the newly created
                          CeedTensorContract will be stored.

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedTensorContractCreate(Ceed ceed, CeedBasis basis, CeedTensorContract *contract) {
  if (!ceed->TensorContractCreate) {
    Ceed delegate;
    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "TensorContract"));

    if (!delegate) {
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support TensorContractCreate");
      // LCOV_EXCL_STOP
    }

    CeedCall(CeedTensorContractCreate(delegate, basis, contract));
    return CEED_ERROR_SUCCESS;
  }

  CeedCall(CeedCalloc(1, contract));

  (*contract)->ceed = ceed;
  CeedCall(CeedReference(ceed));
  CeedCall(ceed->TensorContractCreate(basis, *contract));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Apply tensor contraction

    Contracts on the middle index
    NOTRANSPOSE: v_ajc = t_jb u_abc
    TRANSPOSE:   v_ajc = t_bj u_abc
    If add != 0, "=" is replaced by "+="

  @param contract  CeedTensorContract to use
  @param A         First index of u, v
  @param B         Middle index of u, one index of t
  @param C         Last index of u, v
  @param J         Middle index of v, one index of t
  @param[in] t     Tensor array to contract against
  @param t_mode    Transpose mode for t, \ref CEED_NOTRANSPOSE for t_jb
                     \ref CEED_TRANSPOSE for t_bj
  @param add       Add mode
  @param[in] u     Input array
  @param[out] v    Output array

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedTensorContractApply(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const CeedScalar *restrict t,
                            CeedTransposeMode t_mode, const CeedInt add, const CeedScalar *restrict u, CeedScalar *restrict v) {
  CeedCall(contract->Apply(contract, A, B, C, J, t, t_mode, add, u, v));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get Ceed associated with a CeedTensorContract

  @param contract   CeedTensorContract
  @param[out] ceed  Variable to store Ceed

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedTensorContractGetCeed(CeedTensorContract contract, Ceed *ceed) {
  *ceed = contract->ceed;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get backend data of a CeedTensorContract

  @param contract   CeedTensorContract
  @param[out] data  Variable to store data

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedTensorContractGetData(CeedTensorContract contract, void *data) {
  *(void **)data = contract->data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set backend data of a CeedTensorContract

  @param[out] contract  CeedTensorContract
  @param data           Data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedTensorContractSetData(CeedTensorContract contract, void *data) {
  contract->data = data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Increment the reference counter for a CeedTensorContract

  @param contract  CeedTensorContract to increment the reference counter

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedTensorContractReference(CeedTensorContract contract) {
  contract->ref_count++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Destroy a CeedTensorContract

  @param contract  CeedTensorContract to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedTensorContractDestroy(CeedTensorContract *contract) {
  if (!*contract || --(*contract)->ref_count > 0) return CEED_ERROR_SUCCESS;
  if ((*contract)->Destroy) {
    CeedCall((*contract)->Destroy(*contract));
  }
  CeedCall(CeedDestroy(&(*contract)->ceed));
  CeedCall(CeedFree(contract));
  return CEED_ERROR_SUCCESS;
}

/// @}
