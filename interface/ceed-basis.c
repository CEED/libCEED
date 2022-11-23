// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed-impl.h>
#include <ceed/backend.h>
#include <ceed/ceed.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

/// @file
/// Implementation of CeedBasis interfaces

/// @cond DOXYGEN_SKIP
static struct CeedBasis_private ceed_basis_collocated;
/// @endcond

/// @addtogroup CeedBasisUser
/// @{

/// Indicate that the quadrature points are collocated with the nodes
const CeedBasis CEED_BASIS_COLLOCATED = &ceed_basis_collocated;

/// @}

/// ----------------------------------------------------------------------------
/// CeedBasis Library Internal Functions
/// ----------------------------------------------------------------------------
/// @addtogroup CeedBasisDeveloper
/// @{

/**
  @brief Compute Householder reflection

    Computes A = (I - b v v^T) A
    where A is an mxn matrix indexed as A[i*row + j*col]

  @param[in,out] A  Matrix to apply Householder reflection to, in place
  @param v          Householder vector
  @param b          Scaling factor
  @param m          Number of rows in A
  @param n          Number of columns in A
  @param row        Row stride
  @param col        Col stride

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedHouseholderReflect(CeedScalar *A, const CeedScalar *v, CeedScalar b, CeedInt m, CeedInt n, CeedInt row, CeedInt col) {
  for (CeedInt j = 0; j < n; j++) {
    CeedScalar w = A[0 * row + j * col];
    for (CeedInt i = 1; i < m; i++) w += v[i] * A[i * row + j * col];
    A[0 * row + j * col] -= b * w;
    for (CeedInt i = 1; i < m; i++) A[i * row + j * col] -= b * w * v[i];
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Apply Householder Q matrix

    Compute A = Q A where Q is mxm and A is mxn.

  @param[in,out] A  Matrix to apply Householder Q to, in place
  @param Q          Householder Q matrix
  @param tau        Householder scaling factors
  @param t_mode     Transpose mode for application
  @param m          Number of rows in A
  @param n          Number of columns in A
  @param k          Number of elementary reflectors in Q, k<m
  @param row        Row stride in A
  @param col        Col stride in A

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
int CeedHouseholderApplyQ(CeedScalar *A, const CeedScalar *Q, const CeedScalar *tau, CeedTransposeMode t_mode, CeedInt m, CeedInt n, CeedInt k,
                          CeedInt row, CeedInt col) {
  CeedScalar *v;
  CeedCall(CeedMalloc(m, &v));
  for (CeedInt ii = 0; ii < k; ii++) {
    CeedInt i = t_mode == CEED_TRANSPOSE ? ii : k - 1 - ii;
    for (CeedInt j = i + 1; j < m; j++) v[j] = Q[j * k + i];
    // Apply Householder reflector (I - tau v v^T) collo_grad_1d^T
    CeedCall(CeedHouseholderReflect(&A[i * row], &v[i], tau[i], m - i, n, row, col));
  }
  CeedCall(CeedFree(&v));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Compute Givens rotation

    Computes A = G A (or G^T A in transpose mode)
    where A is an mxn matrix indexed as A[i*n + j*m]

  @param[in,out] A  Row major matrix to apply Givens rotation to, in place
  @param c          Cosine factor
  @param s          Sine factor
  @param t_mode     @ref CEED_NOTRANSPOSE to rotate the basis counter-clockwise,
                    which has the effect of rotating columns of A clockwise;
                    @ref CEED_TRANSPOSE for the opposite rotation
  @param i          First row/column to apply rotation
  @param k          Second row/column to apply rotation
  @param m          Number of rows in A
  @param n          Number of columns in A

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedGivensRotation(CeedScalar *A, CeedScalar c, CeedScalar s, CeedTransposeMode t_mode, CeedInt i, CeedInt k, CeedInt m, CeedInt n) {
  CeedInt stride_j = 1, stride_ik = m, num_its = n;
  if (t_mode == CEED_NOTRANSPOSE) {
    stride_j  = n;
    stride_ik = 1;
    num_its   = m;
  }

  // Apply rotation
  for (CeedInt j = 0; j < num_its; j++) {
    CeedScalar tau1 = A[i * stride_ik + j * stride_j], tau2 = A[k * stride_ik + j * stride_j];
    A[i * stride_ik + j * stride_j] = c * tau1 - s * tau2;
    A[k * stride_ik + j * stride_j] = s * tau1 + c * tau2;
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief View an array stored in a CeedBasis

  @param[in] name      Name of array
  @param[in] fp_fmt    Printing format
  @param[in] m         Number of rows in array
  @param[in] n         Number of columns in array
  @param[in] a         Array to be viewed
  @param[in] stream    Stream to view to, e.g., stdout

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedScalarView(const char *name, const char *fp_fmt, CeedInt m, CeedInt n, const CeedScalar *a, FILE *stream) {
  for (CeedInt i = 0; i < m; i++) {
    if (m > 1) fprintf(stream, "%12s[%" CeedInt_FMT "]:", name, i);
    else fprintf(stream, "%12s:", name);
    for (CeedInt j = 0; j < n; j++) fprintf(stream, fp_fmt, fabs(a[i * n + j]) > 1E-14 ? a[i * n + j] : 0);
    fputs("\n", stream);
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create the interpolation and gradient matrices for projection from
           the nodes of `basis_from` to the nodes of `basis_to`.
           The interpolation is given by `interp_project = interp_to^+ * interp_from`,
           where the pesudoinverse `interp_to^+` is given by QR factorization.
           The gradient is given by `grad_project = interp_to^+ * grad_from`.
         Note: `basis_from` and `basis_to` must have compatible quadrature
           spaces.

  @param[in] basis_from       CeedBasis to project from
  @param[in] basis_to         CeedBasis to project to
  @param[out] interp_project  Address of the variable where the newly created
                                interpolation matrix will be stored.
  @param[out] grad_project    Address of the variable where the newly created
                                gradient matrix will be stored.

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedBasisCreateProjectionMatrices(CeedBasis basis_from, CeedBasis basis_to, CeedScalar **interp_project, CeedScalar **grad_project) {
  Ceed ceed;
  CeedCall(CeedBasisGetCeed(basis_to, &ceed));

  // Check for compatible quadrature spaces
  CeedInt Q_to, Q_from;
  CeedCall(CeedBasisGetNumQuadraturePoints(basis_to, &Q_to));
  CeedCall(CeedBasisGetNumQuadraturePoints(basis_from, &Q_from));
  if (Q_to != Q_from) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "Bases must have compatible quadrature spaces");
    // LCOV_EXCL_STOP
  }

  // Check for matching tensor or non-tensor
  CeedInt P_to, P_from, Q = Q_to;
  bool    is_tensor_to, is_tensor_from;
  CeedCall(CeedBasisIsTensor(basis_to, &is_tensor_to));
  CeedCall(CeedBasisIsTensor(basis_from, &is_tensor_from));
  if (is_tensor_to && is_tensor_from) {
    CeedCall(CeedBasisGetNumNodes1D(basis_to, &P_to));
    CeedCall(CeedBasisGetNumNodes1D(basis_from, &P_from));
    CeedCall(CeedBasisGetNumQuadraturePoints1D(basis_from, &Q));
  } else if (!is_tensor_to && !is_tensor_from) {
    CeedCall(CeedBasisGetNumNodes(basis_to, &P_to));
    CeedCall(CeedBasisGetNumNodes(basis_from, &P_from));
  } else {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_MINOR, "Bases must both be tensor or non-tensor");
    // LCOV_EXCL_STOP
  }

  // Get source matrices
  CeedInt     dim;
  CeedScalar *interp_to, *interp_from, *tau;
  CeedCall(CeedBasisGetDimension(basis_to, &dim));
  CeedCall(CeedMalloc(Q * P_from, &interp_from));
  CeedCall(CeedMalloc(Q * P_to, &interp_to));
  CeedCall(CeedCalloc(P_to * P_from, interp_project));
  CeedCall(CeedCalloc(P_to * P_from * (is_tensor_to ? 1 : dim), grad_project));
  CeedCall(CeedMalloc(Q, &tau));
  const CeedScalar *interp_to_source = NULL, *interp_from_source = NULL, *grad_from_source;
  if (is_tensor_to) {
    CeedCall(CeedBasisGetInterp1D(basis_to, &interp_to_source));
    CeedCall(CeedBasisGetInterp1D(basis_from, &interp_from_source));
    CeedCall(CeedBasisGetGrad1D(basis_from, &grad_from_source));
  } else {
    CeedCall(CeedBasisGetInterp(basis_to, &interp_to_source));
    CeedCall(CeedBasisGetInterp(basis_from, &interp_from_source));
    CeedCall(CeedBasisGetGrad(basis_from, &grad_from_source));
  }

  // Build matrices
  CeedInt     num_matrices = 1 + (is_tensor_to ? 1 : dim);
  CeedScalar *input_from[num_matrices], *output_project[num_matrices];
  input_from[0]     = (CeedScalar *)interp_from_source;
  output_project[0] = *interp_project;
  for (CeedInt m = 1; m < num_matrices; m++) {
    input_from[m]     = (CeedScalar *)&grad_from_source[(m - 1) * Q * P_from];
    output_project[m] = &((*grad_project)[(m - 1) * P_to * P_from]);
  }
  for (CeedInt m = 0; m < num_matrices; m++) {
    // -- QR Factorization, interp_to = Q R
    memcpy(interp_to, interp_to_source, Q * P_to * sizeof(interp_to_source[0]));
    CeedCall(CeedQRFactorization(ceed, interp_to, tau, Q, P_to));

    // -- Apply Qtranspose, interp_to = Qtranspose interp_from
    memcpy(interp_from, input_from[m], Q * P_from * sizeof(input_from[m][0]));
    CeedCall(CeedHouseholderApplyQ(interp_from, interp_to, tau, CEED_TRANSPOSE, Q, P_from, P_to, P_from, 1));

    // -- Apply Rinv, interp_project = Rinv interp_c
    for (CeedInt j = 0; j < P_from; j++) {  // Column j
      output_project[m][j + P_from * (P_to - 1)] = interp_from[j + P_from * (P_to - 1)] / interp_to[P_to * P_to - 1];
      for (CeedInt i = P_to - 2; i >= 0; i--) {  // Row i
        output_project[m][j + P_from * i] = interp_from[j + P_from * i];
        for (CeedInt k = i + 1; k < P_to; k++) {
          output_project[m][j + P_from * i] -= interp_to[k + P_to * i] * output_project[m][j + P_from * k];
        }
        output_project[m][j + P_from * i] /= interp_to[i + P_to * i];
      }
    }
  }

  // Cleanup
  CeedCall(CeedFree(&tau));
  CeedCall(CeedFree(&interp_to));
  CeedCall(CeedFree(&interp_from));

  return CEED_ERROR_SUCCESS;
}

/// @}

/// ----------------------------------------------------------------------------
/// Ceed Backend API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedBasisBackend
/// @{

/**
  @brief Return collocated grad matrix

  @param basis               CeedBasis
  @param[out] collo_grad_1d  Row-major (Q_1d * Q_1d) matrix expressing derivatives of
                               basis functions at quadrature points

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedBasisGetCollocatedGrad(CeedBasis basis, CeedScalar *collo_grad_1d) {
  int         i, j, k;
  Ceed        ceed;
  CeedInt     P_1d = (basis)->P_1d, Q_1d = (basis)->Q_1d;
  CeedScalar *interp_1d, *grad_1d, *tau;

  CeedCall(CeedMalloc(Q_1d * P_1d, &interp_1d));
  CeedCall(CeedMalloc(Q_1d * P_1d, &grad_1d));
  CeedCall(CeedMalloc(Q_1d, &tau));
  memcpy(interp_1d, (basis)->interp_1d, Q_1d * P_1d * sizeof(basis)->interp_1d[0]);
  memcpy(grad_1d, (basis)->grad_1d, Q_1d * P_1d * sizeof(basis)->interp_1d[0]);

  // QR Factorization, interp_1d = Q R
  CeedCall(CeedBasisGetCeed(basis, &ceed));
  CeedCall(CeedQRFactorization(ceed, interp_1d, tau, Q_1d, P_1d));
  // Note: This function is for backend use, so all errors are terminal
  //   and we do not need to clean up memory on failure.

  // Apply Rinv, collo_grad_1d = grad_1d Rinv
  for (i = 0; i < Q_1d; i++) {  // Row i
    collo_grad_1d[Q_1d * i] = grad_1d[P_1d * i] / interp_1d[0];
    for (j = 1; j < P_1d; j++) {  // Column j
      collo_grad_1d[j + Q_1d * i] = grad_1d[j + P_1d * i];
      for (k = 0; k < j; k++) collo_grad_1d[j + Q_1d * i] -= interp_1d[j + P_1d * k] * collo_grad_1d[k + Q_1d * i];
      collo_grad_1d[j + Q_1d * i] /= interp_1d[j + P_1d * j];
    }
    for (j = P_1d; j < Q_1d; j++) collo_grad_1d[j + Q_1d * i] = 0;
  }

  // Apply Qtranspose, collo_grad = collo_grad Q_transpose
  CeedCall(CeedHouseholderApplyQ(collo_grad_1d, interp_1d, tau, CEED_NOTRANSPOSE, Q_1d, Q_1d, P_1d, 1, Q_1d));

  CeedCall(CeedFree(&interp_1d));
  CeedCall(CeedFree(&grad_1d));
  CeedCall(CeedFree(&tau));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get tensor status for given CeedBasis

  @param basis           CeedBasis
  @param[out] is_tensor  Variable to store tensor status

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedBasisIsTensor(CeedBasis basis, bool *is_tensor) {
  *is_tensor = basis->tensor_basis;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get backend data of a CeedBasis

  @param basis      CeedBasis
  @param[out] data  Variable to store data

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedBasisGetData(CeedBasis basis, void *data) {
  *(void **)data = basis->data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set backend data of a CeedBasis

  @param[out] basis  CeedBasis
  @param data        Data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedBasisSetData(CeedBasis basis, void *data) {
  basis->data = data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Increment the reference counter for a CeedBasis

  @param basis  Basis to increment the reference counter

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedBasisReference(CeedBasis basis) {
  basis->ref_count++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Estimate number of FLOPs required to apply CeedBasis in t_mode and eval_mode

  @param basis     Basis to estimate FLOPs for
  @param t_mode    Apply basis or transpose
  @param eval_mode Basis evaluation mode
  @param flops     Address of variable to hold FLOPs estimate

  @ref Backend
**/
int CeedBasisGetFlopsEstimate(CeedBasis basis, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedSize *flops) {
  bool is_tensor;

  CeedCall(CeedBasisIsTensor(basis, &is_tensor));
  if (is_tensor) {
    CeedInt dim, num_comp, P_1d, Q_1d;
    CeedCall(CeedBasisGetDimension(basis, &dim));
    CeedCall(CeedBasisGetNumComponents(basis, &num_comp));
    CeedCall(CeedBasisGetNumNodes1D(basis, &P_1d));
    CeedCall(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
    if (t_mode == CEED_TRANSPOSE) {
      P_1d = Q_1d;
      Q_1d = P_1d;
    }
    CeedInt tensor_flops = 0, pre = num_comp * CeedIntPow(P_1d, dim - 1), post = 1;
    for (CeedInt d = 0; d < dim; d++) {
      tensor_flops += 2 * pre * P_1d * post * Q_1d;
      pre /= P_1d;
      post *= Q_1d;
    }
    switch (eval_mode) {
      case CEED_EVAL_NONE:
        *flops = 0;
        break;
      case CEED_EVAL_INTERP:
        *flops = tensor_flops;
        break;
      case CEED_EVAL_GRAD:
        *flops = tensor_flops * 2;
        break;
      case CEED_EVAL_DIV:
        // LCOV_EXCL_START
        return CeedError(basis->ceed, CEED_ERROR_INCOMPATIBLE, "Tensor CEED_EVAL_DIV not supported");
        break;
      case CEED_EVAL_CURL:
        return CeedError(basis->ceed, CEED_ERROR_INCOMPATIBLE, "Tensor CEED_EVAL_CURL not supported");
        break;
      // LCOV_EXCL_STOP
      case CEED_EVAL_WEIGHT:
        *flops = dim * CeedIntPow(Q_1d, dim);
        break;
    }
  } else {
    CeedInt dim, num_comp, num_nodes, num_qpts, Q_comp;
    CeedCall(CeedBasisGetDimension(basis, &dim));
    CeedCall(CeedBasisGetNumComponents(basis, &num_comp));
    CeedCall(CeedBasisGetNumNodes(basis, &num_nodes));
    CeedCall(CeedBasisGetNumQuadraturePoints(basis, &num_qpts));
    CeedCall(CeedBasisGetNumQuadratureComponents(basis, &Q_comp));
    switch (eval_mode) {
      case CEED_EVAL_NONE:
        *flops = 0;
        break;
      case CEED_EVAL_INTERP:
        *flops = num_nodes * num_qpts * num_comp;
        break;
      case CEED_EVAL_GRAD:
        *flops = num_nodes * num_qpts * num_comp * dim;
        break;
      case CEED_EVAL_DIV:
        *flops = num_nodes * num_qpts;
        break;
      case CEED_EVAL_CURL:
        *flops = num_nodes * num_qpts * dim;
        break;
      case CEED_EVAL_WEIGHT:
        *flops = 0;
        break;
    }
  }

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get dimension for given CeedElemTopology

  @param topo      CeedElemTopology
  @param[out] dim  Variable to store dimension of topology

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedBasisGetTopologyDimension(CeedElemTopology topo, CeedInt *dim) {
  *dim = (CeedInt)topo >> 16;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get CeedTensorContract of a CeedBasis

  @param basis          CeedBasis
  @param[out] contract  Variable to store CeedTensorContract

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedBasisGetTensorContract(CeedBasis basis, CeedTensorContract *contract) {
  *contract = basis->contract;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set CeedTensorContract of a CeedBasis

  @param[out] basis  CeedBasis
  @param contract    CeedTensorContract to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedBasisSetTensorContract(CeedBasis basis, CeedTensorContract contract) {
  basis->contract = contract;
  CeedCall(CeedTensorContractReference(contract));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Return a reference implementation of matrix multiplication C = A B.
           Note, this is a reference implementation for CPU CeedScalar pointers
           that is not intended for high performance.

  @param ceed        A Ceed context for error handling
  @param[in] mat_A   Row-major matrix A
  @param[in] mat_B   Row-major matrix B
  @param[out] mat_C  Row-major output matrix C
  @param m           Number of rows of C
  @param n           Number of columns of C
  @param kk          Number of columns of A/rows of B

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedMatrixMatrixMultiply(Ceed ceed, const CeedScalar *mat_A, const CeedScalar *mat_B, CeedScalar *mat_C, CeedInt m, CeedInt n, CeedInt kk) {
  for (CeedInt i = 0; i < m; i++) {
    for (CeedInt j = 0; j < n; j++) {
      CeedScalar sum = 0;
      for (CeedInt k = 0; k < kk; k++) sum += mat_A[k + i * kk] * mat_B[j + k * n];
      mat_C[j + i * n] = sum;
    }
  }
  return CEED_ERROR_SUCCESS;
}

/// @}

/// ----------------------------------------------------------------------------
/// CeedBasis Public API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedBasisUser
/// @{

/**
  @brief Create a tensor-product basis for H^1 discretizations

  @param ceed        A Ceed object where the CeedBasis will be created
  @param dim         Topological dimension
  @param num_comp    Number of field components (1 for scalar fields)
  @param P_1d        Number of nodes in one dimension
  @param Q_1d        Number of quadrature points in one dimension
  @param interp_1d   Row-major (Q_1d * P_1d) matrix expressing the values of nodal
                       basis functions at quadrature points
  @param grad_1d     Row-major (Q_1d * P_1d) matrix expressing derivatives of nodal
                       basis functions at quadrature points
  @param q_ref_1d    Array of length Q_1d holding the locations of quadrature points
                       on the 1D reference element [-1, 1]
  @param q_weight_1d Array of length Q_1d holding the quadrature weights on the
                       reference element
  @param[out] basis  Address of the variable where the newly created
                       CeedBasis will be stored.

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedBasisCreateTensorH1(Ceed ceed, CeedInt dim, CeedInt num_comp, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d,
                            const CeedScalar *grad_1d, const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis *basis) {
  if (!ceed->BasisCreateTensorH1) {
    Ceed delegate;
    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "Basis"));

    if (!delegate) {
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support BasisCreateTensorH1");
      // LCOV_EXCL_STOP
    }

    CeedCall(CeedBasisCreateTensorH1(delegate, dim, num_comp, P_1d, Q_1d, interp_1d, grad_1d, q_ref_1d, q_weight_1d, basis));
    return CEED_ERROR_SUCCESS;
  }

  if (dim < 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "Basis dimension must be a positive value");
    // LCOV_EXCL_STOP
  }

  if (num_comp < 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "Basis must have at least 1 component");
    // LCOV_EXCL_STOP
  }

  if (P_1d < 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "Basis must have at least 1 node");
    // LCOV_EXCL_STOP
  }

  if (Q_1d < 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "Basis must have at least 1 quadrature point");
    // LCOV_EXCL_STOP
  }

  CeedElemTopology topo = dim == 1 ? CEED_TOPOLOGY_LINE : dim == 2 ? CEED_TOPOLOGY_QUAD : CEED_TOPOLOGY_HEX;

  CeedCall(CeedCalloc(1, basis));
  (*basis)->ceed = ceed;
  CeedCall(CeedReference(ceed));
  (*basis)->ref_count    = 1;
  (*basis)->tensor_basis = 1;
  (*basis)->dim          = dim;
  (*basis)->topo         = topo;
  (*basis)->num_comp     = num_comp;
  (*basis)->P_1d         = P_1d;
  (*basis)->Q_1d         = Q_1d;
  (*basis)->P            = CeedIntPow(P_1d, dim);
  (*basis)->Q            = CeedIntPow(Q_1d, dim);
  (*basis)->Q_comp       = 1;
  (*basis)->basis_space  = 1;  // 1 for H^1 space
  CeedCall(CeedCalloc(Q_1d, &(*basis)->q_ref_1d));
  CeedCall(CeedCalloc(Q_1d, &(*basis)->q_weight_1d));
  if (q_ref_1d) memcpy((*basis)->q_ref_1d, q_ref_1d, Q_1d * sizeof(q_ref_1d[0]));
  if (q_weight_1d) memcpy((*basis)->q_weight_1d, q_weight_1d, Q_1d * sizeof(q_weight_1d[0]));
  CeedCall(CeedCalloc(Q_1d * P_1d, &(*basis)->interp_1d));
  CeedCall(CeedCalloc(Q_1d * P_1d, &(*basis)->grad_1d));
  if (interp_1d) memcpy((*basis)->interp_1d, interp_1d, Q_1d * P_1d * sizeof(interp_1d[0]));
  if (grad_1d) memcpy((*basis)->grad_1d, grad_1d, Q_1d * P_1d * sizeof(grad_1d[0]));
  CeedCall(ceed->BasisCreateTensorH1(dim, P_1d, Q_1d, interp_1d, grad_1d, q_ref_1d, q_weight_1d, *basis));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a tensor-product Lagrange basis

  @param ceed        A Ceed object where the CeedBasis will be created
  @param dim         Topological dimension of element
  @param num_comp      Number of field components (1 for scalar fields)
  @param P           Number of Gauss-Lobatto nodes in one dimension.  The
                       polynomial degree of the resulting Q_k element is k=P-1.
  @param Q           Number of quadrature points in one dimension.
  @param quad_mode   Distribution of the Q quadrature points (affects order of
                       accuracy for the quadrature)
  @param[out] basis  Address of the variable where the newly created
                       CeedBasis will be stored.

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedBasisCreateTensorH1Lagrange(Ceed ceed, CeedInt dim, CeedInt num_comp, CeedInt P, CeedInt Q, CeedQuadMode quad_mode, CeedBasis *basis) {
  // Allocate
  int        ierr = CEED_ERROR_SUCCESS, i, j, k;
  CeedScalar c1, c2, c3, c4, dx, *nodes, *interp_1d, *grad_1d, *q_ref_1d, *q_weight_1d;

  if (dim < 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "Basis dimension must be a positive value");
    // LCOV_EXCL_STOP
  }

  if (num_comp < 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "Basis must have at least 1 component");
    // LCOV_EXCL_STOP
  }

  if (P < 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "Basis must have at least 1 node");
    // LCOV_EXCL_STOP
  }

  if (Q < 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "Basis must have at least 1 quadrature point");
    // LCOV_EXCL_STOP
  }

  // Get Nodes and Weights
  CeedCall(CeedCalloc(P * Q, &interp_1d));
  CeedCall(CeedCalloc(P * Q, &grad_1d));
  CeedCall(CeedCalloc(P, &nodes));
  CeedCall(CeedCalloc(Q, &q_ref_1d));
  CeedCall(CeedCalloc(Q, &q_weight_1d));
  if (CeedLobattoQuadrature(P, nodes, NULL) != CEED_ERROR_SUCCESS) goto cleanup;
  switch (quad_mode) {
    case CEED_GAUSS:
      ierr = CeedGaussQuadrature(Q, q_ref_1d, q_weight_1d);
      break;
    case CEED_GAUSS_LOBATTO:
      ierr = CeedLobattoQuadrature(Q, q_ref_1d, q_weight_1d);
      break;
  }
  if (ierr != CEED_ERROR_SUCCESS) goto cleanup;

  // Build B, D matrix
  // Fornberg, 1998
  for (i = 0; i < Q; i++) {
    c1                   = 1.0;
    c3                   = nodes[0] - q_ref_1d[i];
    interp_1d[i * P + 0] = 1.0;
    for (j = 1; j < P; j++) {
      c2 = 1.0;
      c4 = c3;
      c3 = nodes[j] - q_ref_1d[i];
      for (k = 0; k < j; k++) {
        dx = nodes[j] - nodes[k];
        c2 *= dx;
        if (k == j - 1) {
          grad_1d[i * P + j]   = c1 * (interp_1d[i * P + k] - c4 * grad_1d[i * P + k]) / c2;
          interp_1d[i * P + j] = -c1 * c4 * interp_1d[i * P + k] / c2;
        }
        grad_1d[i * P + k]   = (c3 * grad_1d[i * P + k] - interp_1d[i * P + k]) / dx;
        interp_1d[i * P + k] = c3 * interp_1d[i * P + k] / dx;
      }
      c1 = c2;
    }
  }
  // Pass to CeedBasisCreateTensorH1
  CeedCall(CeedBasisCreateTensorH1(ceed, dim, num_comp, P, Q, interp_1d, grad_1d, q_ref_1d, q_weight_1d, basis));
cleanup:
  CeedCall(CeedFree(&interp_1d));
  CeedCall(CeedFree(&grad_1d));
  CeedCall(CeedFree(&nodes));
  CeedCall(CeedFree(&q_ref_1d));
  CeedCall(CeedFree(&q_weight_1d));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a non tensor-product basis for H^1 discretizations

  @param ceed        A Ceed object where the CeedBasis will be created
  @param topo        Topology of element, e.g. hypercube, simplex, ect
  @param num_comp    Number of field components (1 for scalar fields)
  @param num_nodes   Total number of nodes
  @param num_qpts    Total number of quadrature points
  @param interp      Row-major (num_qpts * num_nodes) matrix expressing the values of
                       nodal basis functions at quadrature points
  @param grad        Row-major (num_qpts * dim * num_nodes) matrix expressing
                       derivatives of nodal basis functions at quadrature points
  @param q_ref       Array of length num_qpts holding the locations of quadrature
                       points on the reference element
  @param q_weight    Array of length num_qpts holding the quadrature weights on the
                       reference element
  @param[out] basis  Address of the variable where the newly created
                       CeedBasis will be stored.

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedBasisCreateH1(Ceed ceed, CeedElemTopology topo, CeedInt num_comp, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                      const CeedScalar *grad, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis *basis) {
  CeedInt P = num_nodes, Q = num_qpts, dim = 0;

  if (!ceed->BasisCreateH1) {
    Ceed delegate;
    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "Basis"));

    if (!delegate) {
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support BasisCreateH1");
      // LCOV_EXCL_STOP
    }

    CeedCall(CeedBasisCreateH1(delegate, topo, num_comp, num_nodes, num_qpts, interp, grad, q_ref, q_weight, basis));
    return CEED_ERROR_SUCCESS;
  }

  if (num_comp < 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "Basis must have at least 1 component");
    // LCOV_EXCL_STOP
  }

  if (num_nodes < 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "Basis must have at least 1 node");
    // LCOV_EXCL_STOP
  }

  if (num_qpts < 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "Basis must have at least 1 quadrature point");
    // LCOV_EXCL_STOP
  }

  CeedCall(CeedCalloc(1, basis));

  CeedCall(CeedBasisGetTopologyDimension(topo, &dim));

  (*basis)->ceed = ceed;
  CeedCall(CeedReference(ceed));
  (*basis)->ref_count    = 1;
  (*basis)->tensor_basis = 0;
  (*basis)->dim          = dim;
  (*basis)->topo         = topo;
  (*basis)->num_comp     = num_comp;
  (*basis)->P            = P;
  (*basis)->Q            = Q;
  (*basis)->Q_comp       = 1;
  (*basis)->basis_space  = 1;  // 1 for H^1 space
  CeedCall(CeedCalloc(Q * dim, &(*basis)->q_ref_1d));
  CeedCall(CeedCalloc(Q, &(*basis)->q_weight_1d));
  if (q_ref) memcpy((*basis)->q_ref_1d, q_ref, Q * dim * sizeof(q_ref[0]));
  if (q_weight) memcpy((*basis)->q_weight_1d, q_weight, Q * sizeof(q_weight[0]));
  CeedCall(CeedCalloc(Q * P, &(*basis)->interp));
  CeedCall(CeedCalloc(dim * Q * P, &(*basis)->grad));
  if (interp) memcpy((*basis)->interp, interp, Q * P * sizeof(interp[0]));
  if (grad) memcpy((*basis)->grad, grad, dim * Q * P * sizeof(grad[0]));
  CeedCall(ceed->BasisCreateH1(topo, dim, P, Q, interp, grad, q_ref, q_weight, *basis));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a non tensor-product basis for H(div) discretizations

  @param ceed        A Ceed object where the CeedBasis will be created
  @param topo        Topology of element (`CEED_TOPOLOGY_QUAD`, `CEED_TOPOLOGY_PRISM`, etc.),
                     dimension of which is used in some array sizes below
  @param num_comp    Number of components (usually 1 for vectors in H(div) bases)
  @param num_nodes   Total number of nodes (dofs per element)
  @param num_qpts    Total number of quadrature points
  @param interp      Row-major (dim*num_qpts * num_nodes) matrix expressing the values of
                       nodal basis functions at quadrature points
  @param div        Row-major (num_qpts * num_nodes) matrix expressing
                       divergence of nodal basis functions at quadrature points
  @param q_ref       Array of length num_qpts holding the locations of quadrature
                       points on the reference element
  @param q_weight    Array of length num_qpts holding the quadrature weights on the
                       reference element
  @param[out] basis  Address of the variable where the newly created
                       CeedBasis will be stored.

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedBasisCreateHdiv(Ceed ceed, CeedElemTopology topo, CeedInt num_comp, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                        const CeedScalar *div, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis *basis) {
  CeedInt Q = num_qpts, P = num_nodes, dim = 0;
  CeedCall(CeedBasisGetTopologyDimension(topo, &dim));
  if (!ceed->BasisCreateHdiv) {
    Ceed delegate;
    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "Basis"));

    if (!delegate) {
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Backend does not implement BasisCreateHdiv");
      // LCOV_EXCL_STOP
    }

    CeedCall(CeedBasisCreateHdiv(delegate, topo, num_comp, num_nodes, num_qpts, interp, div, q_ref, q_weight, basis));
    return CEED_ERROR_SUCCESS;
  }

  if (num_comp < 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "Basis must have at least 1 component");
    // LCOV_EXCL_STOP
  }

  if (num_nodes < 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "Basis must have at least 1 node");
    // LCOV_EXCL_STOP
  }

  if (num_qpts < 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_DIMENSION, "Basis must have at least 1 quadrature point");
    // LCOV_EXCL_STOP
  }

  CeedCall(CeedCalloc(1, basis));

  (*basis)->ceed = ceed;
  CeedCall(CeedReference(ceed));
  (*basis)->ref_count    = 1;
  (*basis)->tensor_basis = 0;
  (*basis)->dim          = dim;
  (*basis)->topo         = topo;
  (*basis)->num_comp     = num_comp;
  (*basis)->P            = P;
  (*basis)->Q            = Q;
  (*basis)->Q_comp       = dim;
  (*basis)->basis_space  = 2;  // 2 for H(div) space
  CeedCall(CeedMalloc(Q * dim, &(*basis)->q_ref_1d));
  CeedCall(CeedMalloc(Q, &(*basis)->q_weight_1d));
  if (q_ref) memcpy((*basis)->q_ref_1d, q_ref, Q * dim * sizeof(q_ref[0]));
  if (q_weight) memcpy((*basis)->q_weight_1d, q_weight, Q * sizeof(q_weight[0]));
  CeedCall(CeedMalloc(dim * Q * P, &(*basis)->interp));
  CeedCall(CeedMalloc(Q * P, &(*basis)->div));
  if (interp) memcpy((*basis)->interp, interp, dim * Q * P * sizeof(interp[0]));
  if (div) memcpy((*basis)->div, div, Q * P * sizeof(div[0]));
  CeedCall(ceed->BasisCreateHdiv(topo, dim, P, Q, interp, div, q_ref, q_weight, *basis));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a CeedBasis for projection from the nodes of `basis_from`
           to the nodes of `basis_to`. Only `CEED_EVAL_INTERP` and
           `CEED_EVAL_GRAD` will be valid for the new basis, `basis_project`.
           The interpolation is given by `interp_project = interp_to^+ * interp_from`,
           where the pesudoinverse `interp_to^+` is given by QR factorization.
           The gradient is given by `grad_project = interp_to^+ * grad_from`.
         Note: `basis_from` and `basis_to` must have compatible quadrature
           spaces.
         Note: `basis_project` will have the same number of components as
           `basis_from`, regardless of the number of components that
           `basis_to` has. If `basis_from` has 3 components and `basis_to`
           has 5 components, then `basis_project` will have 3 components.

  @param[in] basis_from      CeedBasis to prolong from
  @param[in] basis_to        CeedBasis to prolong to
  @param[out] basis_project  Address of the variable where the newly created
                               CeedBasis will be stored.

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedBasisCreateProjection(CeedBasis basis_from, CeedBasis basis_to, CeedBasis *basis_project) {
  Ceed ceed;
  CeedCall(CeedBasisGetCeed(basis_to, &ceed));

  // Create projectior matrix
  CeedScalar *interp_project, *grad_project;
  CeedCall(CeedBasisCreateProjectionMatrices(basis_from, basis_to, &interp_project, &grad_project));

  // Build basis
  bool        is_tensor;
  CeedInt     dim, num_comp;
  CeedScalar *q_ref, *q_weight;
  CeedCall(CeedBasisIsTensor(basis_to, &is_tensor));
  CeedCall(CeedBasisGetDimension(basis_to, &dim));
  CeedCall(CeedBasisGetNumComponents(basis_from, &num_comp));
  if (is_tensor) {
    CeedInt P_1d_to, P_1d_from;
    CeedCall(CeedBasisGetNumNodes1D(basis_from, &P_1d_from));
    CeedCall(CeedBasisGetNumNodes1D(basis_to, &P_1d_to));
    CeedCall(CeedCalloc(P_1d_to, &q_ref));
    CeedCall(CeedCalloc(P_1d_to, &q_weight));
    CeedCall(CeedBasisCreateTensorH1(ceed, dim, num_comp, P_1d_from, P_1d_to, interp_project, grad_project, q_ref, q_weight, basis_project));
  } else {
    CeedElemTopology topo;
    CeedCall(CeedBasisGetTopology(basis_to, &topo));
    CeedInt num_nodes_to, num_nodes_from;
    CeedCall(CeedBasisGetNumNodes(basis_from, &num_nodes_from));
    CeedCall(CeedBasisGetNumNodes(basis_to, &num_nodes_to));
    CeedCall(CeedCalloc(num_nodes_to * dim, &q_ref));
    CeedCall(CeedCalloc(num_nodes_to, &q_weight));
    CeedCall(CeedBasisCreateH1(ceed, topo, num_comp, num_nodes_from, num_nodes_to, interp_project, grad_project, q_ref, q_weight, basis_project));
  }

  // Cleanup
  CeedCall(CeedFree(&interp_project));
  CeedCall(CeedFree(&grad_project));
  CeedCall(CeedFree(&q_ref));
  CeedCall(CeedFree(&q_weight));

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Copy the pointer to a CeedBasis. Both pointers should
           be destroyed with `CeedBasisDestroy()`;
           Note: If `*basis_copy` is non-NULL, then it is assumed that
           `*basis_copy` is a pointer to a CeedBasis. This CeedBasis
           will be destroyed if `*basis_copy` is the only
           reference to this CeedBasis.

  @param basis            CeedBasis to copy reference to
  @param[out] basis_copy  Variable to store copied reference

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedBasisReferenceCopy(CeedBasis basis, CeedBasis *basis_copy) {
  CeedCall(CeedBasisReference(basis));
  CeedCall(CeedBasisDestroy(basis_copy));
  *basis_copy = basis;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief View a CeedBasis

  @param basis   CeedBasis to view
  @param stream  Stream to view to, e.g., stdout

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedBasisView(CeedBasis basis, FILE *stream) {
  CeedFESpace      FE_space = basis->basis_space;
  CeedElemTopology topo     = basis->topo;

  // Print FE space and element topology of the basis
  if (basis->tensor_basis) {
    fprintf(stream, "CeedBasis (%s on a %s element): dim=%" CeedInt_FMT " P=%" CeedInt_FMT " Q=%" CeedInt_FMT "\n", CeedFESpaces[FE_space],
            CeedElemTopologies[topo], basis->dim, basis->P_1d, basis->Q_1d);
  } else {
    fprintf(stream, "CeedBasis (%s on a %s element): dim=%" CeedInt_FMT " P=%" CeedInt_FMT " Q=%" CeedInt_FMT "\n", CeedFESpaces[FE_space],
            CeedElemTopologies[topo], basis->dim, basis->P, basis->Q);
  }
  // Print quadrature data, interpolation/gradient/divergene/curl of the basis
  if (basis->tensor_basis) {  // tensor basis
    CeedCall(CeedScalarView("qref1d", "\t% 12.8f", 1, basis->Q_1d, basis->q_ref_1d, stream));
    CeedCall(CeedScalarView("qweight1d", "\t% 12.8f", 1, basis->Q_1d, basis->q_weight_1d, stream));
    CeedCall(CeedScalarView("interp1d", "\t% 12.8f", basis->Q_1d, basis->P_1d, basis->interp_1d, stream));
    CeedCall(CeedScalarView("grad1d", "\t% 12.8f", basis->Q_1d, basis->P_1d, basis->grad_1d, stream));
  } else {  // non-tensor basis
    CeedCall(CeedScalarView("qref", "\t% 12.8f", 1, basis->Q * basis->dim, basis->q_ref_1d, stream));
    CeedCall(CeedScalarView("qweight", "\t% 12.8f", 1, basis->Q, basis->q_weight_1d, stream));
    CeedCall(CeedScalarView("interp", "\t% 12.8f", basis->Q_comp * basis->Q, basis->P, basis->interp, stream));
    if (basis->grad) {
      CeedCall(CeedScalarView("grad", "\t% 12.8f", basis->dim * basis->Q, basis->P, basis->grad, stream));
    }
    if (basis->div) {
      CeedCall(CeedScalarView("div", "\t% 12.8f", basis->Q, basis->P, basis->div, stream));
    }
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Apply basis evaluation from nodes to quadrature points or vice versa

  @param basis     CeedBasis to evaluate
  @param num_elem  The number of elements to apply the basis evaluation to;
                     the backend will specify the ordering in
                     CeedElemRestrictionCreateBlocked()
  @param t_mode    \ref CEED_NOTRANSPOSE to evaluate from nodes to quadrature
                     points, \ref CEED_TRANSPOSE to apply the transpose, mapping
                     from quadrature points to nodes
  @param eval_mode \ref CEED_EVAL_NONE to use values directly,
                     \ref CEED_EVAL_INTERP to use interpolated values,
                     \ref CEED_EVAL_GRAD to use gradients,
                     \ref CEED_EVAL_WEIGHT to use quadrature weights.
  @param[in] u     Input CeedVector
  @param[out] v    Output CeedVector

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedBasisApply(CeedBasis basis, CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u, CeedVector v) {
  CeedSize u_length = 0, v_length;
  CeedInt  dim, num_comp, num_nodes, num_qpts;
  CeedCall(CeedBasisGetDimension(basis, &dim));
  CeedCall(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCall(CeedBasisGetNumNodes(basis, &num_nodes));
  CeedCall(CeedBasisGetNumQuadraturePoints(basis, &num_qpts));
  CeedCall(CeedVectorGetLength(v, &v_length));
  if (u) {
    CeedCall(CeedVectorGetLength(u, &u_length));
  }

  if (!basis->Apply) {
    // LCOV_EXCL_START
    return CeedError(basis->ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support BasisApply");
    // LCOV_EXCL_STOP
  }

  // Check compatibility of topological and geometrical dimensions
  if ((t_mode == CEED_TRANSPOSE && (v_length % num_nodes != 0 || u_length % num_qpts != 0)) ||
      (t_mode == CEED_NOTRANSPOSE && (u_length % num_nodes != 0 || v_length % num_qpts != 0))) {
    // LCOV_EXCL_START
    return CeedError(basis->ceed, CEED_ERROR_DIMENSION, "Length of input/output vectors incompatible with basis dimensions");
    // LCOV_EXCL_STOP
  }

  // Check vector lengths to prevent out of bounds issues
  bool bad_dims = false;
  switch (eval_mode) {
    case CEED_EVAL_NONE:
    case CEED_EVAL_INTERP:
      bad_dims = ((t_mode == CEED_TRANSPOSE && (u_length < num_elem * num_comp * num_qpts || v_length < num_elem * num_comp * num_nodes)) ||
                  (t_mode == CEED_NOTRANSPOSE && (v_length < num_elem * num_qpts * num_comp || u_length < num_elem * num_comp * num_nodes)));
      break;
    case CEED_EVAL_GRAD:
      bad_dims = ((t_mode == CEED_TRANSPOSE && (u_length < num_elem * num_comp * num_qpts * dim || v_length < num_elem * num_comp * num_nodes)) ||
                  (t_mode == CEED_NOTRANSPOSE && (v_length < num_elem * num_qpts * num_comp * dim || u_length < num_elem * num_comp * num_nodes)));
      break;
    case CEED_EVAL_WEIGHT:
      bad_dims = v_length < num_elem * num_qpts;
      break;
    // LCOV_EXCL_START
    case CEED_EVAL_DIV:
      bad_dims = ((t_mode == CEED_TRANSPOSE && (u_length < num_elem * num_comp * num_qpts || v_length < num_elem * num_comp * num_nodes)) ||
                  (t_mode == CEED_NOTRANSPOSE && (v_length < num_elem * num_qpts * num_comp || u_length < num_elem * num_comp * num_nodes)));
      break;
    case CEED_EVAL_CURL:
      bad_dims = ((t_mode == CEED_TRANSPOSE && (u_length < num_elem * num_comp * num_qpts || v_length < num_elem * num_comp * num_nodes)) ||
                  (t_mode == CEED_NOTRANSPOSE && (v_length < num_elem * num_qpts * num_comp || u_length < num_elem * num_comp * num_nodes)));
      break;
      // LCOV_EXCL_STOP
  }
  if (bad_dims) {
    // LCOV_EXCL_START
    return CeedError(basis->ceed, CEED_ERROR_DIMENSION, "Input/output vectors too short for basis and evaluation mode");
    // LCOV_EXCL_STOP
  }

  CeedCall(basis->Apply(basis, num_elem, t_mode, eval_mode, u, v));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get Ceed associated with a CeedBasis

  @param basis      CeedBasis
  @param[out] ceed  Variable to store Ceed

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetCeed(CeedBasis basis, Ceed *ceed) {
  *ceed = basis->ceed;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get dimension for given CeedBasis

  @param basis     CeedBasis
  @param[out] dim  Variable to store dimension of basis

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetDimension(CeedBasis basis, CeedInt *dim) {
  *dim = basis->dim;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get topology for given CeedBasis

  @param basis      CeedBasis
  @param[out] topo  Variable to store topology of basis

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetTopology(CeedBasis basis, CeedElemTopology *topo) {
  *topo = basis->topo;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get number of Q-vector components for given CeedBasis

  @param basis          CeedBasis
  @param[out] Q_comp  Variable to store number of Q-vector components of basis

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetNumQuadratureComponents(CeedBasis basis, CeedInt *Q_comp) {
  *Q_comp = basis->Q_comp;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get number of components for given CeedBasis

  @param basis          CeedBasis
  @param[out] num_comp  Variable to store number of components of basis

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetNumComponents(CeedBasis basis, CeedInt *num_comp) {
  *num_comp = basis->num_comp;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get total number of nodes (in dim dimensions) of a CeedBasis

  @param basis   CeedBasis
  @param[out] P  Variable to store number of nodes

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedBasisGetNumNodes(CeedBasis basis, CeedInt *P) {
  *P = basis->P;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get total number of nodes (in 1 dimension) of a CeedBasis

  @param basis     CeedBasis
  @param[out] P_1d  Variable to store number of nodes

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetNumNodes1D(CeedBasis basis, CeedInt *P_1d) {
  if (!basis->tensor_basis) {
    // LCOV_EXCL_START
    return CeedError(basis->ceed, CEED_ERROR_MINOR, "Cannot supply P_1d for non-tensor basis");
    // LCOV_EXCL_STOP
  }

  *P_1d = basis->P_1d;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get total number of quadrature points (in dim dimensions) of a CeedBasis

  @param basis   CeedBasis
  @param[out] Q  Variable to store number of quadrature points

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedBasisGetNumQuadraturePoints(CeedBasis basis, CeedInt *Q) {
  *Q = basis->Q;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get total number of quadrature points (in 1 dimension) of a CeedBasis

  @param basis      CeedBasis
  @param[out] Q_1d  Variable to store number of quadrature points

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetNumQuadraturePoints1D(CeedBasis basis, CeedInt *Q_1d) {
  if (!basis->tensor_basis) {
    // LCOV_EXCL_START
    return CeedError(basis->ceed, CEED_ERROR_MINOR, "Cannot supply Q_1d for non-tensor basis");
    // LCOV_EXCL_STOP
  }

  *Q_1d = basis->Q_1d;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get reference coordinates of quadrature points (in dim dimensions)
         of a CeedBasis

  @param basis       CeedBasis
  @param[out] q_ref  Variable to store reference coordinates of quadrature points

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetQRef(CeedBasis basis, const CeedScalar **q_ref) {
  *q_ref = basis->q_ref_1d;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get quadrature weights of quadrature points (in dim dimensions)
         of a CeedBasis

  @param basis          CeedBasis
  @param[out] q_weight  Variable to store quadrature weights

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetQWeights(CeedBasis basis, const CeedScalar **q_weight) {
  *q_weight = basis->q_weight_1d;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get interpolation matrix of a CeedBasis

  @param basis        CeedBasis
  @param[out] interp  Variable to store interpolation matrix

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetInterp(CeedBasis basis, const CeedScalar **interp) {
  if (!basis->interp && basis->tensor_basis) {
    // Allocate
    CeedCall(CeedMalloc(basis->Q * basis->P, &basis->interp));

    // Initialize
    for (CeedInt i = 0; i < basis->Q * basis->P; i++) basis->interp[i] = 1.0;

    // Calculate
    for (CeedInt d = 0; d < basis->dim; d++) {
      for (CeedInt qpt = 0; qpt < basis->Q; qpt++) {
        for (CeedInt node = 0; node < basis->P; node++) {
          CeedInt p = (node / CeedIntPow(basis->P_1d, d)) % basis->P_1d;
          CeedInt q = (qpt / CeedIntPow(basis->Q_1d, d)) % basis->Q_1d;
          basis->interp[qpt * (basis->P) + node] *= basis->interp_1d[q * basis->P_1d + p];
        }
      }
    }
  }
  *interp = basis->interp;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get 1D interpolation matrix of a tensor product CeedBasis

  @param basis           CeedBasis
  @param[out] interp_1d  Variable to store interpolation matrix

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedBasisGetInterp1D(CeedBasis basis, const CeedScalar **interp_1d) {
  if (!basis->tensor_basis) {
    // LCOV_EXCL_START
    return CeedError(basis->ceed, CEED_ERROR_MINOR, "CeedBasis is not a tensor product basis.");
    // LCOV_EXCL_STOP
  }

  *interp_1d = basis->interp_1d;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get gradient matrix of a CeedBasis

  @param basis      CeedBasis
  @param[out] grad  Variable to store gradient matrix

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetGrad(CeedBasis basis, const CeedScalar **grad) {
  if (!basis->grad && basis->tensor_basis) {
    // Allocate
    CeedCall(CeedMalloc(basis->dim * basis->Q * basis->P, &basis->grad));

    // Initialize
    for (CeedInt i = 0; i < basis->dim * basis->Q * basis->P; i++) basis->grad[i] = 1.0;

    // Calculate
    for (CeedInt d = 0; d < basis->dim; d++) {
      for (CeedInt i = 0; i < basis->dim; i++) {
        for (CeedInt qpt = 0; qpt < basis->Q; qpt++) {
          for (CeedInt node = 0; node < basis->P; node++) {
            CeedInt p = (node / CeedIntPow(basis->P_1d, d)) % basis->P_1d;
            CeedInt q = (qpt / CeedIntPow(basis->Q_1d, d)) % basis->Q_1d;
            if (i == d) basis->grad[(i * basis->Q + qpt) * (basis->P) + node] *= basis->grad_1d[q * basis->P_1d + p];
            else basis->grad[(i * basis->Q + qpt) * (basis->P) + node] *= basis->interp_1d[q * basis->P_1d + p];
          }
        }
      }
    }
  }
  *grad = basis->grad;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get 1D gradient matrix of a tensor product CeedBasis

  @param basis         CeedBasis
  @param[out] grad_1d  Variable to store gradient matrix

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetGrad1D(CeedBasis basis, const CeedScalar **grad_1d) {
  if (!basis->tensor_basis) {
    // LCOV_EXCL_START
    return CeedError(basis->ceed, CEED_ERROR_MINOR, "CeedBasis is not a tensor product basis.");
    // LCOV_EXCL_STOP
  }

  *grad_1d = basis->grad_1d;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get divergence matrix of a CeedBasis

  @param basis     CeedBasis
  @param[out] div  Variable to store divergence matrix

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetDiv(CeedBasis basis, const CeedScalar **div) {
  if (!basis->div) {
    // LCOV_EXCL_START
    return CeedError(basis->ceed, CEED_ERROR_MINOR, "CeedBasis does not have divergence matrix.");
    // LCOV_EXCL_STOP
  }

  *div = basis->div;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Destroy a CeedBasis

  @param basis CeedBasis to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedBasisDestroy(CeedBasis *basis) {
  if (!*basis || --(*basis)->ref_count > 0) return CEED_ERROR_SUCCESS;
  if ((*basis)->Destroy) CeedCall((*basis)->Destroy(*basis));
  if ((*basis)->contract) CeedCall(CeedTensorContractDestroy(&(*basis)->contract));
  CeedCall(CeedFree(&(*basis)->interp));
  CeedCall(CeedFree(&(*basis)->interp_1d));
  CeedCall(CeedFree(&(*basis)->grad));
  CeedCall(CeedFree(&(*basis)->div));
  CeedCall(CeedFree(&(*basis)->grad_1d));
  CeedCall(CeedFree(&(*basis)->q_ref_1d));
  CeedCall(CeedFree(&(*basis)->q_weight_1d));
  CeedCall(CeedDestroy(&(*basis)->ceed));
  CeedCall(CeedFree(basis));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Construct a Gauss-Legendre quadrature

  @param Q                 Number of quadrature points (integrates polynomials of
                             degree 2*Q-1 exactly)
  @param[out] q_ref_1d     Array of length Q to hold the abscissa on [-1, 1]
  @param[out] q_weight_1d  Array of length Q to hold the weights

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedGaussQuadrature(CeedInt Q, CeedScalar *q_ref_1d, CeedScalar *q_weight_1d) {
  // Allocate
  CeedScalar P0, P1, P2, dP2, xi, wi, PI = 4.0 * atan(1.0);
  // Build q_ref_1d, q_weight_1d
  for (CeedInt i = 0; i <= Q / 2; i++) {
    // Guess
    xi = cos(PI * (CeedScalar)(2 * i + 1) / ((CeedScalar)(2 * Q)));
    // Pn(xi)
    P0 = 1.0;
    P1 = xi;
    P2 = 0.0;
    for (CeedInt j = 2; j <= Q; j++) {
      P2 = (((CeedScalar)(2 * j - 1)) * xi * P1 - ((CeedScalar)(j - 1)) * P0) / ((CeedScalar)(j));
      P0 = P1;
      P1 = P2;
    }
    // First Newton Step
    dP2 = (xi * P2 - P0) * (CeedScalar)Q / (xi * xi - 1.0);
    xi  = xi - P2 / dP2;
    // Newton to convergence
    for (CeedInt k = 0; k < 100 && fabs(P2) > 10 * CEED_EPSILON; k++) {
      P0 = 1.0;
      P1 = xi;
      for (CeedInt j = 2; j <= Q; j++) {
        P2 = (((CeedScalar)(2 * j - 1)) * xi * P1 - ((CeedScalar)(j - 1)) * P0) / ((CeedScalar)(j));
        P0 = P1;
        P1 = P2;
      }
      dP2 = (xi * P2 - P0) * (CeedScalar)Q / (xi * xi - 1.0);
      xi  = xi - P2 / dP2;
    }
    // Save xi, wi
    wi                     = 2.0 / ((1.0 - xi * xi) * dP2 * dP2);
    q_weight_1d[i]         = wi;
    q_weight_1d[Q - 1 - i] = wi;
    q_ref_1d[i]            = -xi;
    q_ref_1d[Q - 1 - i]    = xi;
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Construct a Gauss-Legendre-Lobatto quadrature

  @param Q                 Number of quadrature points (integrates polynomials of
                             degree 2*Q-3 exactly)
  @param[out] q_ref_1d     Array of length Q to hold the abscissa on [-1, 1]
  @param[out] q_weight_1d  Array of length Q to hold the weights

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedLobattoQuadrature(CeedInt Q, CeedScalar *q_ref_1d, CeedScalar *q_weight_1d) {
  // Allocate
  CeedScalar P0, P1, P2, dP2, d2P2, xi, wi, PI = 4.0 * atan(1.0);
  // Build q_ref_1d, q_weight_1d
  // Set endpoints
  if (Q < 2) {
    // LCOV_EXCL_START
    return CeedError(NULL, CEED_ERROR_DIMENSION, "Cannot create Lobatto quadrature with Q=%" CeedInt_FMT " < 2 points", Q);
    // LCOV_EXCL_STOP
  }
  wi = 2.0 / ((CeedScalar)(Q * (Q - 1)));
  if (q_weight_1d) {
    q_weight_1d[0]     = wi;
    q_weight_1d[Q - 1] = wi;
  }
  q_ref_1d[0]     = -1.0;
  q_ref_1d[Q - 1] = 1.0;
  // Interior
  for (CeedInt i = 1; i <= (Q - 1) / 2; i++) {
    // Guess
    xi = cos(PI * (CeedScalar)(i) / (CeedScalar)(Q - 1));
    // Pn(xi)
    P0 = 1.0;
    P1 = xi;
    P2 = 0.0;
    for (CeedInt j = 2; j < Q; j++) {
      P2 = (((CeedScalar)(2 * j - 1)) * xi * P1 - ((CeedScalar)(j - 1)) * P0) / ((CeedScalar)(j));
      P0 = P1;
      P1 = P2;
    }
    // First Newton step
    dP2  = (xi * P2 - P0) * (CeedScalar)Q / (xi * xi - 1.0);
    d2P2 = (2 * xi * dP2 - (CeedScalar)(Q * (Q - 1)) * P2) / (1.0 - xi * xi);
    xi   = xi - dP2 / d2P2;
    // Newton to convergence
    for (CeedInt k = 0; k < 100 && fabs(dP2) > 10 * CEED_EPSILON; k++) {
      P0 = 1.0;
      P1 = xi;
      for (CeedInt j = 2; j < Q; j++) {
        P2 = (((CeedScalar)(2 * j - 1)) * xi * P1 - ((CeedScalar)(j - 1)) * P0) / ((CeedScalar)(j));
        P0 = P1;
        P1 = P2;
      }
      dP2  = (xi * P2 - P0) * (CeedScalar)Q / (xi * xi - 1.0);
      d2P2 = (2 * xi * dP2 - (CeedScalar)(Q * (Q - 1)) * P2) / (1.0 - xi * xi);
      xi   = xi - dP2 / d2P2;
    }
    // Save xi, wi
    wi = 2.0 / (((CeedScalar)(Q * (Q - 1))) * P2 * P2);
    if (q_weight_1d) {
      q_weight_1d[i]         = wi;
      q_weight_1d[Q - 1 - i] = wi;
    }
    q_ref_1d[i]         = -xi;
    q_ref_1d[Q - 1 - i] = xi;
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Return QR Factorization of a matrix

  @param ceed         A Ceed context for error handling
  @param[in,out] mat  Row-major matrix to be factorized in place
  @param[in,out] tau  Vector of length m of scaling factors
  @param m            Number of rows
  @param n            Number of columns

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedQRFactorization(Ceed ceed, CeedScalar *mat, CeedScalar *tau, CeedInt m, CeedInt n) {
  CeedScalar v[m];

  // Check matrix shape
  if (n > m) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Cannot compute QR factorization with n > m");
    // LCOV_EXCL_STOP
  }

  for (CeedInt i = 0; i < n; i++) {
    if (i >= m - 1) {  // last row of matrix, no reflection needed
      tau[i] = 0.;
      break;
    }
    // Calculate Householder vector, magnitude
    CeedScalar sigma = 0.0;
    v[i]             = mat[i + n * i];
    for (CeedInt j = i + 1; j < m; j++) {
      v[j] = mat[i + n * j];
      sigma += v[j] * v[j];
    }
    CeedScalar norm = sqrt(v[i] * v[i] + sigma);  // norm of v[i:m]
    CeedScalar R_ii = -copysign(norm, v[i]);
    v[i] -= R_ii;
    // norm of v[i:m] after modification above and scaling below
    //   norm = sqrt(v[i]*v[i] + sigma) / v[i];
    //   tau = 2 / (norm*norm)
    tau[i] = 2 * v[i] * v[i] / (v[i] * v[i] + sigma);
    for (CeedInt j = i + 1; j < m; j++) v[j] /= v[i];

    // Apply Householder reflector to lower right panel
    CeedHouseholderReflect(&mat[i * n + i + 1], &v[i], tau[i], m - i, n - i - 1, n, 1);
    // Save v
    mat[i + n * i] = R_ii;
    for (CeedInt j = i + 1; j < m; j++) mat[i + n * j] = v[j];
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Return symmetric Schur decomposition of the symmetric matrix mat via
           symmetric QR factorization

  @param ceed         A Ceed context for error handling
  @param[in,out] mat  Row-major matrix to be factorized in place
  @param[out] lambda  Vector of length n of eigenvalues
  @param n            Number of rows/columns

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
CeedPragmaOptimizeOff int CeedSymmetricSchurDecomposition(Ceed ceed, CeedScalar *mat, CeedScalar *lambda, CeedInt n) {
  // Check bounds for clang-tidy
  if (n < 2) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Cannot compute symmetric Schur decomposition of scalars");
    // LCOV_EXCL_STOP
  }

  CeedScalar v[n - 1], tau[n - 1], mat_T[n * n];

  // Copy mat to mat_T and set mat to I
  memcpy(mat_T, mat, n * n * sizeof(mat[0]));
  for (CeedInt i = 0; i < n; i++) {
    for (CeedInt j = 0; j < n; j++) mat[j + n * i] = (i == j) ? 1 : 0;
  }

  // Reduce to tridiagonal
  for (CeedInt i = 0; i < n - 1; i++) {
    // Calculate Householder vector, magnitude
    CeedScalar sigma = 0.0;
    v[i]             = mat_T[i + n * (i + 1)];
    for (CeedInt j = i + 1; j < n - 1; j++) {
      v[j] = mat_T[i + n * (j + 1)];
      sigma += v[j] * v[j];
    }
    CeedScalar norm = sqrt(v[i] * v[i] + sigma);  // norm of v[i:n-1]
    CeedScalar R_ii = -copysign(norm, v[i]);
    v[i] -= R_ii;
    // norm of v[i:m] after modification above and scaling below
    //   norm = sqrt(v[i]*v[i] + sigma) / v[i];
    //   tau = 2 / (norm*norm)
    tau[i] = i == n - 2 ? 2 : 2 * v[i] * v[i] / (v[i] * v[i] + sigma);
    for (CeedInt j = i + 1; j < n - 1; j++) v[j] /= v[i];

    // Update sub and super diagonal
    for (CeedInt j = i + 2; j < n; j++) {
      mat_T[i + n * j] = 0;
      mat_T[j + n * i] = 0;
    }
    // Apply symmetric Householder reflector to lower right panel
    CeedHouseholderReflect(&mat_T[(i + 1) + n * (i + 1)], &v[i], tau[i], n - (i + 1), n - (i + 1), n, 1);
    CeedHouseholderReflect(&mat_T[(i + 1) + n * (i + 1)], &v[i], tau[i], n - (i + 1), n - (i + 1), 1, n);

    // Save v
    mat_T[i + n * (i + 1)] = R_ii;
    mat_T[(i + 1) + n * i] = R_ii;
    for (CeedInt j = i + 1; j < n - 1; j++) {
      mat_T[i + n * (j + 1)] = v[j];
    }
  }
  // Backwards accumulation of Q
  for (CeedInt i = n - 2; i >= 0; i--) {
    if (tau[i] > 0.0) {
      v[i] = 1;
      for (CeedInt j = i + 1; j < n - 1; j++) {
        v[j]                   = mat_T[i + n * (j + 1)];
        mat_T[i + n * (j + 1)] = 0;
      }
      CeedHouseholderReflect(&mat[(i + 1) + n * (i + 1)], &v[i], tau[i], n - (i + 1), n - (i + 1), n, 1);
    }
  }

  // Reduce sub and super diagonal
  CeedInt    p = 0, q = 0, itr = 0, max_itr = n * n * n * n;
  CeedScalar tol = CEED_EPSILON;

  while (itr < max_itr) {
    // Update p, q, size of reduced portions of diagonal
    p = 0;
    q = 0;
    for (CeedInt i = n - 2; i >= 0; i--) {
      if (fabs(mat_T[i + n * (i + 1)]) < tol) q += 1;
      else break;
    }
    for (CeedInt i = 0; i < n - q - 1; i++) {
      if (fabs(mat_T[i + n * (i + 1)]) < tol) p += 1;
      else break;
    }
    if (q == n - 1) break;  // Finished reducing

    // Reduce tridiagonal portion
    CeedScalar t_nn = mat_T[(n - 1 - q) + n * (n - 1 - q)], t_nnm1 = mat_T[(n - 2 - q) + n * (n - 1 - q)];
    CeedScalar d  = (mat_T[(n - 2 - q) + n * (n - 2 - q)] - t_nn) / 2;
    CeedScalar mu = t_nn - t_nnm1 * t_nnm1 / (d + copysign(sqrt(d * d + t_nnm1 * t_nnm1), d));
    CeedScalar x  = mat_T[p + n * p] - mu;
    CeedScalar z  = mat_T[p + n * (p + 1)];
    for (CeedInt k = p; k < n - q - 1; k++) {
      // Compute Givens rotation
      CeedScalar c = 1, s = 0;
      if (fabs(z) > tol) {
        if (fabs(z) > fabs(x)) {
          CeedScalar tau = -x / z;
          s = 1 / sqrt(1 + tau * tau), c = s * tau;
        } else {
          CeedScalar tau = -z / x;
          c = 1 / sqrt(1 + tau * tau), s = c * tau;
        }
      }

      // Apply Givens rotation to T
      CeedGivensRotation(mat_T, c, s, CEED_NOTRANSPOSE, k, k + 1, n, n);
      CeedGivensRotation(mat_T, c, s, CEED_TRANSPOSE, k, k + 1, n, n);

      // Apply Givens rotation to Q
      CeedGivensRotation(mat, c, s, CEED_NOTRANSPOSE, k, k + 1, n, n);

      // Update x, z
      if (k < n - q - 2) {
        x = mat_T[k + n * (k + 1)];
        z = mat_T[k + n * (k + 2)];
      }
    }
    itr++;
  }

  // Save eigenvalues
  for (CeedInt i = 0; i < n; i++) lambda[i] = mat_T[i + n * i];

  // Check convergence
  if (itr == max_itr && q < n - 1) {
    // LCOV_EXCL_START
    return CeedError(ceed, CEED_ERROR_MINOR, "Symmetric QR failed to converge");
    // LCOV_EXCL_STOP
  }
  return CEED_ERROR_SUCCESS;
}
CeedPragmaOptimizeOn

    /**
      @brief Return Simultaneous Diagonalization of two matrices. This solves the
               generalized eigenvalue problem A x = lambda B x, where A and B
               are symmetric and B is positive definite. We generate the matrix X
               and vector Lambda such that X^T A X = Lambda and X^T B X = I. This
               is equivalent to the LAPACK routine 'sygv' with TYPE = 1.

      @param ceed         A Ceed context for error handling
      @param[in] mat_A    Row-major matrix to be factorized with eigenvalues
      @param[in] mat_B    Row-major matrix to be factorized to identity
      @param[out] mat_X   Row-major orthogonal matrix
      @param[out] lambda  Vector of length n of generalized eigenvalues
      @param n            Number of rows/columns

      @return An error code: 0 - success, otherwise - failure

      @ref Utility
    **/
    CeedPragmaOptimizeOff int
    CeedSimultaneousDiagonalization(Ceed ceed, CeedScalar *mat_A, CeedScalar *mat_B, CeedScalar *mat_X, CeedScalar *lambda, CeedInt n) {
  CeedScalar *mat_C, *mat_G, *vec_D;
  CeedCall(CeedCalloc(n * n, &mat_C));
  CeedCall(CeedCalloc(n * n, &mat_G));
  CeedCall(CeedCalloc(n, &vec_D));

  // Compute B = G D G^T
  memcpy(mat_G, mat_B, n * n * sizeof(mat_B[0]));
  CeedCall(CeedSymmetricSchurDecomposition(ceed, mat_G, vec_D, n));

  // Sort eigenvalues
  for (CeedInt i = n - 1; i >= 0; i--) {
    for (CeedInt j = 0; j < i; j++) {
      if (fabs(vec_D[j]) > fabs(vec_D[j + 1])) {
        CeedScalar temp;
        temp         = vec_D[j];
        vec_D[j]     = vec_D[j + 1];
        vec_D[j + 1] = temp;
        for (CeedInt k = 0; k < n; k++) {
          temp                 = mat_G[k * n + j];
          mat_G[k * n + j]     = mat_G[k * n + j + 1];
          mat_G[k * n + j + 1] = temp;
        }
      }
    }
  }

  // Compute C = (G D^1/2)^-1 A (G D^1/2)^-T
  //           = D^-1/2 G^T A G D^-1/2
  // -- D = D^-1/2
  for (CeedInt i = 0; i < n; i++) vec_D[i] = 1. / sqrt(vec_D[i]);
  // -- G = G D^-1/2
  // -- C = D^-1/2 G^T
  for (CeedInt i = 0; i < n; i++) {
    for (CeedInt j = 0; j < n; j++) {
      mat_G[i * n + j] *= vec_D[j];
      mat_C[j * n + i] = mat_G[i * n + j];
    }
  }
  // -- X = (D^-1/2 G^T) A
  CeedCall(CeedMatrixMatrixMultiply(ceed, (const CeedScalar *)mat_C, (const CeedScalar *)mat_A, mat_X, n, n, n));
  // -- C = (D^-1/2 G^T A) (G D^-1/2)
  CeedCall(CeedMatrixMatrixMultiply(ceed, (const CeedScalar *)mat_X, (const CeedScalar *)mat_G, mat_C, n, n, n));

  // Compute Q^T C Q = lambda
  CeedCall(CeedSymmetricSchurDecomposition(ceed, mat_C, lambda, n));

  // Sort eigenvalues
  for (CeedInt i = n - 1; i >= 0; i--) {
    for (CeedInt j = 0; j < i; j++) {
      if (fabs(lambda[j]) > fabs(lambda[j + 1])) {
        CeedScalar temp;
        temp          = lambda[j];
        lambda[j]     = lambda[j + 1];
        lambda[j + 1] = temp;
        for (CeedInt k = 0; k < n; k++) {
          temp                 = mat_C[k * n + j];
          mat_C[k * n + j]     = mat_C[k * n + j + 1];
          mat_C[k * n + j + 1] = temp;
        }
      }
    }
  }

  // Set X = (G D^1/2)^-T Q
  //       = G D^-1/2 Q
  CeedCall(CeedMatrixMatrixMultiply(ceed, (const CeedScalar *)mat_G, (const CeedScalar *)mat_C, mat_X, n, n, n));

  // Cleanup
  CeedCall(CeedFree(&mat_C));
  CeedCall(CeedFree(&mat_G));
  CeedCall(CeedFree(&vec_D));
  return CEED_ERROR_SUCCESS;
}
CeedPragmaOptimizeOn

    /// @}
