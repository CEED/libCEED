// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <ceed-impl.h>

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
int CeedTensorContractCreate(Ceed ceed, CeedBasis basis,
                             CeedTensorContract *contract) {
  int ierr;

  if (!ceed->TensorContractCreate) {
    Ceed delegate;
    ierr = CeedGetObjectDelegate(ceed, &delegate, "TensorContract");
    CeedChk(ierr);

    if (!delegate)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
                       "Backend does not support TensorContractCreate");
    // LCOV_EXCL_STOP

    ierr = CeedTensorContractCreate(delegate, basis, contract);
    CeedChk(ierr);
    return CEED_ERROR_SUCCESS;
  }

  ierr = CeedCalloc(1, contract); CeedChk(ierr);

  (*contract)->ceed = ceed;
  ierr = CeedIncrementRefCounter(ceed); CeedChk(ierr);
  ierr = ceed->TensorContractCreate(basis, *contract);
  CeedChk(ierr);
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
int CeedTensorContractApply(CeedTensorContract contract, CeedInt A, CeedInt B,
                            CeedInt C, CeedInt J, const CeedScalar *restrict t,
                            CeedTransposeMode t_mode, const CeedInt add,
                            const CeedScalar *restrict u,
                            CeedScalar *restrict v) {
  int ierr;

  ierr = contract->Apply(contract, A, B, C, J, t, t_mode, add,  u, v);
  CeedChk(ierr);
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
int CeedTensorContractIncrementRefCounter(CeedTensorContract contract) {
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
  int ierr;

  if (!*contract || --(*contract)->ref_count > 0) return CEED_ERROR_SUCCESS;
  if ((*contract)->Destroy) {
    ierr = (*contract)->Destroy(*contract); CeedChk(ierr);
  }
  ierr = CeedDestroy(&(*contract)->ceed); CeedChk(ierr);
  ierr = CeedFree(contract); CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/// @}
