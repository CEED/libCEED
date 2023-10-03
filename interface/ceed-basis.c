// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed-impl.h>
#include <ceed.h>
#include <ceed/backend.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

/// @file
/// Implementation of CeedBasis interfaces

/// @cond DOXYGEN_SKIP
static struct CeedBasis_private ceed_basis_none;
/// @endcond

/// @addtogroup CeedBasisUser
/// @{

/// Argument for CeedOperatorSetField indicating that the field does not require a CeedBasis
const CeedBasis CEED_BASIS_NONE = &ceed_basis_none;

/// This feature will be removed. Use CEED_BASIS_NONE.
const CeedBasis CEED_BASIS_COLLOCATED = &ceed_basis_none;

/// @}

/// ----------------------------------------------------------------------------
/// CeedBasis Library Internal Functions
/// ----------------------------------------------------------------------------
/// @addtogroup CeedBasisDeveloper
/// @{

/**
  @brief Compute Chebyshev polynomial values at a point

  @param[in]  x           Coordinate to evaluate Chebyshev polynomials at
  @param[in]  n           Number of Chebyshev polynomials to evaluate, n >= 2
  @param[out] chebyshev_x Array of Chebyshev polynomial values

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedChebyshevPolynomialsAtPoint(CeedScalar x, CeedInt n, CeedScalar *chebyshev_x) {
  chebyshev_x[0] = 1.0;
  chebyshev_x[1] = 2 * x;
  for (CeedInt i = 2; i < n; i++) chebyshev_x[i] = 2 * x * chebyshev_x[i - 1] - chebyshev_x[i - 2];
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Compute values of the derivative of Chebyshev polynomials at a point

  @param[in]  x           Coordinate to evaluate derivative of Chebyshev polynomials at
  @param[in]  n           Number of Chebyshev polynomials to evaluate, n >= 2
  @param[out] chebyshev_x Array of Chebyshev polynomial derivative values

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedChebyshevDerivativeAtPoint(CeedScalar x, CeedInt n, CeedScalar *chebyshev_dx) {
  CeedScalar chebyshev_x[3];

  chebyshev_x[1]  = 1.0;
  chebyshev_x[2]  = 2 * x;
  chebyshev_dx[0] = 0.0;
  chebyshev_dx[1] = 2.0;
  for (CeedInt i = 2; i < n; i++) {
    chebyshev_x[0]  = chebyshev_x[1];
    chebyshev_x[1]  = chebyshev_x[2];
    chebyshev_x[2]  = 2 * x * chebyshev_x[1] - chebyshev_x[0];
    chebyshev_dx[i] = 2 * x * chebyshev_dx[i - 1] + 2 * chebyshev_x[1] - chebyshev_dx[i - 2];
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Compute Householder reflection

  Computes A = (I - b v v^T) A, where A is an mxn matrix indexed as A[i*row + j*col]

  @param[in,out] A   Matrix to apply Householder reflection to, in place
  @param[in]     v   Householder vector
  @param[in]     b   Scaling factor
  @param[in]     m   Number of rows in A
  @param[in]     n   Number of columns in A
  @param[in]     row Row stride
  @param[in]     col Col stride

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
  @brief Compute Givens rotation

  Computes A = G A (or G^T A in transpose mode), where A is an mxn matrix indexed as A[i*n + j*m]

  @param[in,out] A      Row major matrix to apply Givens rotation to, in place
  @param[in]     c      Cosine factor
  @param[in]     s      Sine factor
  @param[in]     t_mode @ref CEED_NOTRANSPOSE to rotate the basis counter-clockwise, which has the effect of rotating columns of A clockwise;
                          @ref CEED_TRANSPOSE for the opposite rotation
  @param[in]     i      First row/column to apply rotation
  @param[in]     k      Second row/column to apply rotation
  @param[in]     m      Number of rows in A
  @param[in]     n      Number of columns in A

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

  @param[in] name   Name of array
  @param[in] fp_fmt Printing format
  @param[in] m      Number of rows in array
  @param[in] n      Number of columns in array
  @param[in] a      Array to be viewed
  @param[in] stream Stream to view to, e.g., stdout

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedScalarView(const char *name, const char *fp_fmt, CeedInt m, CeedInt n, const CeedScalar *a, FILE *stream) {
  if (m > 1) {
    fprintf(stream, "  %s:\n", name);
  } else {
    char padded_name[12];

    snprintf(padded_name, 11, "%s:", name);
    fprintf(stream, "  %-10s", padded_name);
  }
  for (CeedInt i = 0; i < m; i++) {
    if (m > 1) fprintf(stream, "    [%" CeedInt_FMT "]", i);
    for (CeedInt j = 0; j < n; j++) fprintf(stream, fp_fmt, fabs(a[i * n + j]) > 1E-14 ? a[i * n + j] : 0);
    fputs("\n", stream);
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create the interpolation and gradient matrices for projection from the nodes of `basis_from` to the nodes of `basis_to`.

  The interpolation is given by `interp_project = interp_to^+ * interp_from`, where the pseudoinverse `interp_to^+` is given by QR factorization.
  The gradient is given by `grad_project = interp_to^+ * grad_from`, and is only computed for H^1 spaces otherwise it should not be used.

  Note: `basis_from` and `basis_to` must have compatible quadrature spaces.

  @param[in]  basis_from     CeedBasis to project from
  @param[in]  basis_to       CeedBasis to project to
  @param[out] interp_project Address of the variable where the newly created interpolation matrix will be stored.
  @param[out] grad_project   Address of the variable where the newly created gradient matrix will be stored.

  @return An error code: 0 - success, otherwise - failure

  @ref Developer
**/
static int CeedBasisCreateProjectionMatrices(CeedBasis basis_from, CeedBasis basis_to, CeedScalar **interp_project, CeedScalar **grad_project) {
  Ceed    ceed;
  bool    is_tensor_to, is_tensor_from;
  CeedInt Q, Q_to, Q_from, P_to, P_from;

  CeedCall(CeedBasisGetCeed(basis_to, &ceed));

  // Check for compatible quadrature spaces
  CeedCall(CeedBasisGetNumQuadraturePoints(basis_to, &Q_to));
  CeedCall(CeedBasisGetNumQuadraturePoints(basis_from, &Q_from));
  CeedCheck(Q_to == Q_from, ceed, CEED_ERROR_DIMENSION, "Bases must have compatible quadrature spaces");
  Q = Q_to;

  // Check for matching tensor or non-tensor
  CeedCall(CeedBasisIsTensor(basis_to, &is_tensor_to));
  CeedCall(CeedBasisIsTensor(basis_from, &is_tensor_from));
  CeedCheck(is_tensor_to == is_tensor_from, ceed, CEED_ERROR_MINOR, "Bases must both be tensor or non-tensor");
  if (is_tensor_to) {
    CeedCall(CeedBasisGetNumNodes1D(basis_to, &P_to));
    CeedCall(CeedBasisGetNumNodes1D(basis_from, &P_from));
    CeedCall(CeedBasisGetNumQuadraturePoints1D(basis_from, &Q));
  } else {
    CeedCall(CeedBasisGetNumNodes(basis_to, &P_to));
    CeedCall(CeedBasisGetNumNodes(basis_from, &P_from));
  }

  // Check for matching FE space
  CeedFESpace fe_space_to, fe_space_from;
  CeedCall(CeedBasisGetFESpace(basis_to, &fe_space_to));
  CeedCall(CeedBasisGetFESpace(basis_from, &fe_space_from));
  CeedCheck(fe_space_to == fe_space_from, ceed, CEED_ERROR_MINOR, "Bases must both be the same FE space type");

  // Get source matrices
  CeedInt           dim, q_comp = 1;
  CeedScalar       *interp_to, *interp_from, *tau;
  const CeedScalar *interp_to_source = NULL, *interp_from_source = NULL, *grad_from_source = NULL;

  CeedCall(CeedBasisGetDimension(basis_to, &dim));
  if (is_tensor_to) {
    CeedCall(CeedBasisGetInterp1D(basis_to, &interp_to_source));
    CeedCall(CeedBasisGetInterp1D(basis_from, &interp_from_source));
  } else {
    CeedCall(CeedBasisGetNumQuadratureComponents(basis_from, CEED_EVAL_INTERP, &q_comp));
    CeedCall(CeedBasisGetInterp(basis_to, &interp_to_source));
    CeedCall(CeedBasisGetInterp(basis_from, &interp_from_source));
  }
  CeedCall(CeedMalloc(Q * P_from * q_comp, &interp_from));
  CeedCall(CeedMalloc(Q * P_to * q_comp, &interp_to));
  CeedCall(CeedCalloc(P_to * P_from, interp_project));
  CeedCall(CeedMalloc(Q * q_comp, &tau));

  // `grad_project = interp_to^+ * grad_from` is computed for the H^1 space case so the
  // projection basis will have a gradient operation (allocated even if not H^1 for the
  // basis construction later on)
  if (fe_space_to == CEED_FE_SPACE_H1) {
    if (is_tensor_to) {
      CeedCall(CeedBasisGetGrad1D(basis_from, &grad_from_source));
    } else {
      CeedCall(CeedBasisGetGrad(basis_from, &grad_from_source));
    }
  }
  CeedCall(CeedCalloc(P_to * P_from * (is_tensor_to ? 1 : dim), grad_project));

  // QR Factorization, interp_to = Q R
  memcpy(interp_to, interp_to_source, Q * P_to * q_comp * sizeof(interp_to_source[0]));
  CeedCall(CeedQRFactorization(ceed, interp_to, tau, Q * q_comp, P_to));

  // Build matrices
  CeedInt     num_matrices = 1 + (fe_space_to == CEED_FE_SPACE_H1) * (is_tensor_to ? 1 : dim);
  CeedScalar *input_from[num_matrices], *output_project[num_matrices];

  input_from[0]     = (CeedScalar *)interp_from_source;
  output_project[0] = *interp_project;
  for (CeedInt m = 1; m < num_matrices; m++) {
    input_from[m]     = (CeedScalar *)&grad_from_source[(m - 1) * Q * P_from];
    output_project[m] = &((*grad_project)[(m - 1) * P_to * P_from]);
  }
  for (CeedInt m = 0; m < num_matrices; m++) {
    // Apply Q^T, interp_from = Q^T interp_from
    memcpy(interp_from, input_from[m], Q * P_from * q_comp * sizeof(input_from[m][0]));
    CeedCall(CeedHouseholderApplyQ(interp_from, interp_to, tau, CEED_TRANSPOSE, Q * q_comp, P_from, P_to, P_from, 1));

    // Apply Rinv, output_project = Rinv interp_from
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

  @param[in]  basis         CeedBasis
  @param[out] collo_grad_1d Row-major (Q_1d * Q_1d) matrix expressing derivatives of basis functions at quadrature points

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedBasisGetCollocatedGrad(CeedBasis basis, CeedScalar *collo_grad_1d) {
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
  // Note: This function is for backend use, so all errors are terminal and we do not need to clean up memory on failure.

  // Apply R_inv, collo_grad_1d = grad_1d R_inv
  for (CeedInt i = 0; i < Q_1d; i++) {  // Row i
    collo_grad_1d[Q_1d * i] = grad_1d[P_1d * i] / interp_1d[0];
    for (CeedInt j = 1; j < P_1d; j++) {  // Column j
      collo_grad_1d[j + Q_1d * i] = grad_1d[j + P_1d * i];
      for (CeedInt k = 0; k < j; k++) collo_grad_1d[j + Q_1d * i] -= interp_1d[j + P_1d * k] * collo_grad_1d[k + Q_1d * i];
      collo_grad_1d[j + Q_1d * i] /= interp_1d[j + P_1d * j];
    }
    for (CeedInt j = P_1d; j < Q_1d; j++) collo_grad_1d[j + Q_1d * i] = 0;
  }

  // Apply Q^T, collo_grad_1d = collo_grad_1d Q^T
  CeedCall(CeedHouseholderApplyQ(collo_grad_1d, interp_1d, tau, CEED_NOTRANSPOSE, Q_1d, Q_1d, P_1d, 1, Q_1d));

  CeedCall(CeedFree(&interp_1d));
  CeedCall(CeedFree(&grad_1d));
  CeedCall(CeedFree(&tau));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get tensor status for given CeedBasis

  @param[in]  basis     CeedBasis
  @param[out] is_tensor Variable to store tensor status

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedBasisIsTensor(CeedBasis basis, bool *is_tensor) {
  *is_tensor = basis->is_tensor_basis;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get backend data of a CeedBasis

  @param[in]  basis CeedBasis
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

  @param[in,out] basis  CeedBasis
  @param[in]     data   Data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedBasisSetData(CeedBasis basis, void *data) {
  basis->data = data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Increment the reference counter for a CeedBasis

  @param[in,out] basis Basis to increment the reference counter

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedBasisReference(CeedBasis basis) {
  basis->ref_count++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get number of Q-vector components for given CeedBasis

  @param[in]  basis  CeedBasis
  @param[in]  eval_mode \ref CEED_EVAL_INTERP to use interpolated values,
                          \ref CEED_EVAL_GRAD to use gradients,
                          \ref CEED_EVAL_DIV to use divergence,
                          \ref CEED_EVAL_CURL to use curl.
  @param[out] q_comp Variable to store number of Q-vector components of basis

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedBasisGetNumQuadratureComponents(CeedBasis basis, CeedEvalMode eval_mode, CeedInt *q_comp) {
  switch (eval_mode) {
    case CEED_EVAL_INTERP:
      *q_comp = (basis->fe_space == CEED_FE_SPACE_H1) ? 1 : basis->dim;
      break;
    case CEED_EVAL_GRAD:
      *q_comp = basis->dim;
      break;
    case CEED_EVAL_DIV:
      *q_comp = 1;
      break;
    case CEED_EVAL_CURL:
      *q_comp = (basis->dim < 3) ? 1 : basis->dim;
      break;
    case CEED_EVAL_NONE:
    case CEED_EVAL_WEIGHT:
      *q_comp = 1;
      break;
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Estimate number of FLOPs required to apply CeedBasis in t_mode and eval_mode

  @param[in]  basis     Basis to estimate FLOPs for
  @param[in]  t_mode    Apply basis or transpose
  @param[in]  eval_mode Basis evaluation mode
  @param[out] flops     Address of variable to hold FLOPs estimate

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
      case CEED_EVAL_CURL:
        // LCOV_EXCL_START
        return CeedError(basis->ceed, CEED_ERROR_INCOMPATIBLE, "Tensor basis evaluation for %s not supported", CeedEvalModes[eval_mode]);
        break;
      // LCOV_EXCL_STOP
      case CEED_EVAL_WEIGHT:
        *flops = dim * CeedIntPow(Q_1d, dim);
        break;
    }
  } else {
    CeedInt dim, num_comp, q_comp, num_nodes, num_qpts;

    CeedCall(CeedBasisGetDimension(basis, &dim));
    CeedCall(CeedBasisGetNumComponents(basis, &num_comp));
    CeedCall(CeedBasisGetNumQuadratureComponents(basis, eval_mode, &q_comp));
    CeedCall(CeedBasisGetNumNodes(basis, &num_nodes));
    CeedCall(CeedBasisGetNumQuadraturePoints(basis, &num_qpts));
    switch (eval_mode) {
      case CEED_EVAL_NONE:
        *flops = 0;
        break;
      case CEED_EVAL_INTERP:
      case CEED_EVAL_GRAD:
      case CEED_EVAL_DIV:
      case CEED_EVAL_CURL:
        *flops = num_nodes * num_qpts * num_comp * q_comp;
        break;
      case CEED_EVAL_WEIGHT:
        *flops = 0;
        break;
    }
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get CeedFESpace for a CeedBasis

  @param[in]  basis     CeedBasis
  @param[out] fe_space  Variable to store CeedFESpace

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedBasisGetFESpace(CeedBasis basis, CeedFESpace *fe_space) {
  *fe_space = basis->fe_space;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get dimension for given CeedElemTopology

  @param[in]  topo CeedElemTopology
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

  @param[in]  basis     CeedBasis
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

  @param[in,out] basis    CeedBasis
  @param[in]     contract CeedTensorContract to set

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

  Note: This is a reference implementation for CPU CeedScalar pointers that is not intended for high performance.

  @param[in]  ceed  Ceed context for error handling
  @param[in]  mat_A Row-major matrix A
  @param[in]  mat_B Row-major matrix B
  @param[out] mat_C Row-major output matrix C
  @param[in]  m     Number of rows of C
  @param[in]  n     Number of columns of C
  @param[in]  kk    Number of columns of A/rows of B

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

/**
  @brief Return QR Factorization of a matrix

  @param[in]     ceed Ceed context for error handling
  @param[in,out] mat  Row-major matrix to be factorized in place
  @param[in,out] tau  Vector of length m of scaling factors
  @param[in]     m    Number of rows
  @param[in]     n    Number of columns

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedQRFactorization(Ceed ceed, CeedScalar *mat, CeedScalar *tau, CeedInt m, CeedInt n) {
  CeedScalar v[m];

  // Check matrix shape
  CeedCheck(n <= m, ceed, CEED_ERROR_UNSUPPORTED, "Cannot compute QR factorization with n > m");

  for (CeedInt i = 0; i < n; i++) {
    CeedScalar sigma = 0.0;

    if (i >= m - 1) {  // last row of matrix, no reflection needed
      tau[i] = 0.;
      break;
    }
    // Calculate Householder vector, magnitude
    v[i] = mat[i + n * i];
    for (CeedInt j = i + 1; j < m; j++) {
      v[j] = mat[i + n * j];
      sigma += v[j] * v[j];
    }
    const CeedScalar norm = sqrt(v[i] * v[i] + sigma);  // norm of v[i:m]
    const CeedScalar R_ii = -copysign(norm, v[i]);

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
  @brief Apply Householder Q matrix

  Compute mat_A = mat_Q mat_A, where mat_Q is mxm and mat_A is mxn.

  @param[in,out] mat_A  Matrix to apply Householder Q to, in place
  @param[in]     mat_Q  Householder Q matrix
  @param[in]     tau    Householder scaling factors
  @param[in]     t_mode Transpose mode for application
  @param[in]     m      Number of rows in A
  @param[in]     n      Number of columns in A
  @param[in]     k      Number of elementary reflectors in Q, k<m
  @param[in]     row    Row stride in A
  @param[in]     col    Col stride in A

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedHouseholderApplyQ(CeedScalar *mat_A, const CeedScalar *mat_Q, const CeedScalar *tau, CeedTransposeMode t_mode, CeedInt m, CeedInt n,
                          CeedInt k, CeedInt row, CeedInt col) {
  CeedScalar *v;

  CeedCall(CeedMalloc(m, &v));
  for (CeedInt ii = 0; ii < k; ii++) {
    CeedInt i = t_mode == CEED_TRANSPOSE ? ii : k - 1 - ii;
    for (CeedInt j = i + 1; j < m; j++) v[j] = mat_Q[j * k + i];
    // Apply Householder reflector (I - tau v v^T) collo_grad_1d^T
    CeedCall(CeedHouseholderReflect(&mat_A[i * row], &v[i], tau[i], m - i, n, row, col));
  }
  CeedCall(CeedFree(&v));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Return symmetric Schur decomposition of the symmetric matrix mat via symmetric QR factorization

  @param[in]     ceed   Ceed context for error handling
  @param[in,out] mat    Row-major matrix to be factorized in place
  @param[out]    lambda Vector of length n of eigenvalues
  @param[in]     n      Number of rows/columns

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
CeedPragmaOptimizeOff
int CeedSymmetricSchurDecomposition(Ceed ceed, CeedScalar *mat, CeedScalar *lambda, CeedInt n) {
  // Check bounds for clang-tidy
  CeedCheck(n > 1, ceed, CEED_ERROR_UNSUPPORTED, "Cannot compute symmetric Schur decomposition of scalars");

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

    v[i] = mat_T[i + n * (i + 1)];
    for (CeedInt j = i + 1; j < n - 1; j++) {
      v[j] = mat_T[i + n * (j + 1)];
      sigma += v[j] * v[j];
    }
    const CeedScalar norm = sqrt(v[i] * v[i] + sigma);  // norm of v[i:n-1]
    const CeedScalar R_ii = -copysign(norm, v[i]);

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
          const CeedScalar tau = -x / z;

          s = 1 / sqrt(1 + tau * tau);
          c = s * tau;
        } else {
          const CeedScalar tau = -z / x;

          c = 1 / sqrt(1 + tau * tau);
          s = c * tau;
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
  CeedCheck(itr < max_itr || q > n, ceed, CEED_ERROR_MINOR, "Symmetric QR failed to converge");
  return CEED_ERROR_SUCCESS;
}
CeedPragmaOptimizeOn

/**
  @brief Return Simultaneous Diagonalization of two matrices.

  This solves the generalized eigenvalue problem A x = lambda B x, where A and B are symmetric and B is positive definite.
  We generate the matrix X and vector Lambda such that X^T A X = Lambda and X^T B X = I.
  This is equivalent to the LAPACK routine 'sygv' with TYPE = 1.

  @param[in]  ceed   Ceed context for error handling
  @param[in]  mat_A  Row-major matrix to be factorized with eigenvalues
  @param[in]  mat_B  Row-major matrix to be factorized to identity
  @param[out] mat_X  Row-major orthogonal matrix
  @param[out] lambda Vector of length n of generalized eigenvalues
  @param[in]  n      Number of rows/columns

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
CeedPragmaOptimizeOff
int CeedSimultaneousDiagonalization(Ceed ceed, CeedScalar *mat_A, CeedScalar *mat_B, CeedScalar *mat_X, CeedScalar *lambda, CeedInt n) {
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
        CeedScalarSwap(vec_D[j], vec_D[j + 1]);
        for (CeedInt k = 0; k < n; k++) CeedScalarSwap(mat_G[k * n + j], mat_G[k * n + j + 1]);
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
        CeedScalarSwap(lambda[j], lambda[j + 1]);
        for (CeedInt k = 0; k < n; k++) CeedScalarSwap(mat_C[k * n + j], mat_C[k * n + j + 1]);
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

/// ----------------------------------------------------------------------------
/// CeedBasis Public API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedBasisUser
/// @{

/**
  @brief Create a tensor-product basis for H^1 discretizations

  @param[in]  ceed        Ceed object where the CeedBasis will be created
  @param[in]  dim         Topological dimension
  @param[in]  num_comp    Number of field components (1 for scalar fields)
  @param[in]  P_1d        Number of nodes in one dimension
  @param[in]  Q_1d        Number of quadrature points in one dimension
  @param[in]  interp_1d   Row-major (Q_1d * P_1d) matrix expressing the values of nodal basis functions at quadrature points
  @param[in]  grad_1d     Row-major (Q_1d * P_1d) matrix expressing derivatives of nodal basis functions at quadrature points
  @param[in]  q_ref_1d    Array of length Q_1d holding the locations of quadrature points on the 1D reference element [-1, 1]
  @param[in]  q_weight_1d Array of length Q_1d holding the quadrature weights on the reference element
  @param[out] basis       Address of the variable where the newly created CeedBasis will be stored.

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedBasisCreateTensorH1(Ceed ceed, CeedInt dim, CeedInt num_comp, CeedInt P_1d, CeedInt Q_1d, const CeedScalar *interp_1d,
                            const CeedScalar *grad_1d, const CeedScalar *q_ref_1d, const CeedScalar *q_weight_1d, CeedBasis *basis) {
  if (!ceed->BasisCreateTensorH1) {
    Ceed delegate;

    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "Basis"));
    CeedCheck(delegate, ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support BasisCreateTensorH1");
    CeedCall(CeedBasisCreateTensorH1(delegate, dim, num_comp, P_1d, Q_1d, interp_1d, grad_1d, q_ref_1d, q_weight_1d, basis));
    return CEED_ERROR_SUCCESS;
  }

  CeedCheck(dim > 0, ceed, CEED_ERROR_DIMENSION, "Basis dimension must be a positive value");
  CeedCheck(num_comp > 0, ceed, CEED_ERROR_DIMENSION, "Basis must have at least 1 component");
  CeedCheck(P_1d > 0, ceed, CEED_ERROR_DIMENSION, "Basis must have at least 1 node");
  CeedCheck(Q_1d > 0, ceed, CEED_ERROR_DIMENSION, "Basis must have at least 1 quadrature point");

  CeedElemTopology topo = dim == 1 ? CEED_TOPOLOGY_LINE : dim == 2 ? CEED_TOPOLOGY_QUAD : CEED_TOPOLOGY_HEX;

  CeedCall(CeedCalloc(1, basis));
  CeedCall(CeedReferenceCopy(ceed, &(*basis)->ceed));
  (*basis)->ref_count       = 1;
  (*basis)->is_tensor_basis = true;
  (*basis)->dim             = dim;
  (*basis)->topo            = topo;
  (*basis)->num_comp        = num_comp;
  (*basis)->P_1d            = P_1d;
  (*basis)->Q_1d            = Q_1d;
  (*basis)->P               = CeedIntPow(P_1d, dim);
  (*basis)->Q               = CeedIntPow(Q_1d, dim);
  (*basis)->fe_space        = CEED_FE_SPACE_H1;
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

  @param[in]  ceed      Ceed object where the CeedBasis will be created
  @param[in]  dim       Topological dimension of element
  @param[in]  num_comp  Number of field components (1 for scalar fields)
  @param[in]  P         Number of Gauss-Lobatto nodes in one dimension.
                          The polynomial degree of the resulting Q_k element is k=P-1.
  @param[in]  Q         Number of quadrature points in one dimension.
  @param[in]  quad_mode Distribution of the Q quadrature points (affects order of accuracy for the quadrature)
  @param[out] basis     Address of the variable where the newly created CeedBasis will be stored.

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedBasisCreateTensorH1Lagrange(Ceed ceed, CeedInt dim, CeedInt num_comp, CeedInt P, CeedInt Q, CeedQuadMode quad_mode, CeedBasis *basis) {
  // Allocate
  int        ierr = CEED_ERROR_SUCCESS;
  CeedScalar c1, c2, c3, c4, dx, *nodes, *interp_1d, *grad_1d, *q_ref_1d, *q_weight_1d;

  CeedCheck(dim > 0, ceed, CEED_ERROR_DIMENSION, "Basis dimension must be a positive value");
  CeedCheck(num_comp > 0, ceed, CEED_ERROR_DIMENSION, "Basis must have at least 1 component");
  CeedCheck(P > 0, ceed, CEED_ERROR_DIMENSION, "Basis must have at least 1 node");
  CeedCheck(Q > 0, ceed, CEED_ERROR_DIMENSION, "Basis must have at least 1 quadrature point");

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
  for (CeedInt i = 0; i < Q; i++) {
    c1                   = 1.0;
    c3                   = nodes[0] - q_ref_1d[i];
    interp_1d[i * P + 0] = 1.0;
    for (CeedInt j = 1; j < P; j++) {
      c2 = 1.0;
      c4 = c3;
      c3 = nodes[j] - q_ref_1d[i];
      for (CeedInt k = 0; k < j; k++) {
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

  @param[in]  ceed      Ceed object where the CeedBasis will be created
  @param[in]  topo      Topology of element, e.g. hypercube, simplex, ect
  @param[in]  num_comp  Number of field components (1 for scalar fields)
  @param[in]  num_nodes Total number of nodes
  @param[in]  num_qpts  Total number of quadrature points
  @param[in]  interp    Row-major (num_qpts * num_nodes) matrix expressing the values of nodal basis functions at quadrature points
  @param[in]  grad      Row-major (dim * num_qpts * num_nodes) matrix expressing derivatives of nodal basis functions at quadrature points
  @param[in]  q_ref     Array of length num_qpts * dim holding the locations of quadrature points on the reference element
  @param[in]  q_weight  Array of length num_qpts holding the quadrature weights on the reference element
  @param[out] basis     Address of the variable where the newly created CeedBasis will be stored.

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedBasisCreateH1(Ceed ceed, CeedElemTopology topo, CeedInt num_comp, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                      const CeedScalar *grad, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis *basis) {
  CeedInt P = num_nodes, Q = num_qpts, dim = 0;

  if (!ceed->BasisCreateH1) {
    Ceed delegate;

    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "Basis"));
    CeedCheck(delegate, ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support BasisCreateH1");
    CeedCall(CeedBasisCreateH1(delegate, topo, num_comp, num_nodes, num_qpts, interp, grad, q_ref, q_weight, basis));
    return CEED_ERROR_SUCCESS;
  }

  CeedCheck(num_comp > 0, ceed, CEED_ERROR_DIMENSION, "Basis must have at least 1 component");
  CeedCheck(num_nodes > 0, ceed, CEED_ERROR_DIMENSION, "Basis must have at least 1 node");
  CeedCheck(num_qpts > 0, ceed, CEED_ERROR_DIMENSION, "Basis must have at least 1 quadrature point");

  CeedCall(CeedBasisGetTopologyDimension(topo, &dim));

  CeedCall(CeedCalloc(1, basis));
  CeedCall(CeedReferenceCopy(ceed, &(*basis)->ceed));
  (*basis)->ref_count       = 1;
  (*basis)->is_tensor_basis = false;
  (*basis)->dim             = dim;
  (*basis)->topo            = topo;
  (*basis)->num_comp        = num_comp;
  (*basis)->P               = P;
  (*basis)->Q               = Q;
  (*basis)->fe_space        = CEED_FE_SPACE_H1;
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
  @brief Create a non tensor-product basis for \f$H(\mathrm{div})\f$ discretizations

  @param[in]  ceed      Ceed object where the CeedBasis will be created
  @param[in]  topo      Topology of element (`CEED_TOPOLOGY_QUAD`, `CEED_TOPOLOGY_PRISM`, etc.), dimension of which is used in some array sizes below
  @param[in]  num_comp  Number of components (usually 1 for vectors in H(div) bases)
  @param[in]  num_nodes Total number of nodes (dofs per element)
  @param[in]  num_qpts  Total number of quadrature points
  @param[in]  interp    Row-major (dim * num_qpts * num_nodes) matrix expressing the values of basis functions at quadrature points
  @param[in]  div       Row-major (num_qpts * num_nodes) matrix expressing divergence of basis functions at quadrature points
  @param[in]  q_ref     Array of length num_qpts * dim holding the locations of quadrature points on the reference element
  @param[in]  q_weight  Array of length num_qpts holding the quadrature weights on the reference element
  @param[out] basis     Address of the variable where the newly created CeedBasis will be stored.

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedBasisCreateHdiv(Ceed ceed, CeedElemTopology topo, CeedInt num_comp, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                        const CeedScalar *div, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis *basis) {
  CeedInt Q = num_qpts, P = num_nodes, dim = 0;

  if (!ceed->BasisCreateHdiv) {
    Ceed delegate;

    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "Basis"));
    CeedCheck(delegate, ceed, CEED_ERROR_UNSUPPORTED, "Backend does not implement BasisCreateHdiv");
    CeedCall(CeedBasisCreateHdiv(delegate, topo, num_comp, num_nodes, num_qpts, interp, div, q_ref, q_weight, basis));
    return CEED_ERROR_SUCCESS;
  }

  CeedCheck(num_comp > 0, ceed, CEED_ERROR_DIMENSION, "Basis must have at least 1 component");
  CeedCheck(num_nodes > 0, ceed, CEED_ERROR_DIMENSION, "Basis must have at least 1 node");
  CeedCheck(num_qpts > 0, ceed, CEED_ERROR_DIMENSION, "Basis must have at least 1 quadrature point");

  CeedCall(CeedBasisGetTopologyDimension(topo, &dim));

  CeedCall(CeedCalloc(1, basis));
  CeedCall(CeedReferenceCopy(ceed, &(*basis)->ceed));
  (*basis)->ref_count       = 1;
  (*basis)->is_tensor_basis = false;
  (*basis)->dim             = dim;
  (*basis)->topo            = topo;
  (*basis)->num_comp        = num_comp;
  (*basis)->P               = P;
  (*basis)->Q               = Q;
  (*basis)->fe_space        = CEED_FE_SPACE_HDIV;
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
  @brief Create a non tensor-product basis for \f$H(\mathrm{curl})\f$ discretizations

  @param[in]  ceed      Ceed object where the CeedBasis will be created
  @param[in]  topo      Topology of element (`CEED_TOPOLOGY_QUAD`, `CEED_TOPOLOGY_PRISM`, etc.), dimension of which is used in some array sizes below
  @param[in]  num_comp  Number of components (usually 1 for vectors in H(curl) bases)
  @param[in]  num_nodes Total number of nodes (dofs per element)
  @param[in]  num_qpts  Total number of quadrature points
  @param[in]  interp    Row-major (dim * num_qpts * num_nodes) matrix expressing the values of basis functions at quadrature points
  @param[in]  curl      Row-major (curl_comp * num_qpts * num_nodes, curl_comp = 1 if dim < 3 else dim) matrix expressing curl of basis functions at
quadrature points
  @param[in]  q_ref     Array of length num_qpts * dim holding the locations of quadrature points on the reference element
  @param[in]  q_weight  Array of length num_qpts holding the quadrature weights on the reference element
  @param[out] basis     Address of the variable where the newly created CeedBasis will be stored.

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedBasisCreateHcurl(Ceed ceed, CeedElemTopology topo, CeedInt num_comp, CeedInt num_nodes, CeedInt num_qpts, const CeedScalar *interp,
                         const CeedScalar *curl, const CeedScalar *q_ref, const CeedScalar *q_weight, CeedBasis *basis) {
  CeedInt Q = num_qpts, P = num_nodes, dim = 0, curl_comp = 0;

  if (!ceed->BasisCreateHdiv) {
    Ceed delegate;

    CeedCall(CeedGetObjectDelegate(ceed, &delegate, "Basis"));
    CeedCheck(delegate, ceed, CEED_ERROR_UNSUPPORTED, "Backend does not implement BasisCreateHcurl");
    CeedCall(CeedBasisCreateHcurl(delegate, topo, num_comp, num_nodes, num_qpts, interp, curl, q_ref, q_weight, basis));
    return CEED_ERROR_SUCCESS;
  }

  CeedCheck(num_comp > 0, ceed, CEED_ERROR_DIMENSION, "Basis must have at least 1 component");
  CeedCheck(num_nodes > 0, ceed, CEED_ERROR_DIMENSION, "Basis must have at least 1 node");
  CeedCheck(num_qpts > 0, ceed, CEED_ERROR_DIMENSION, "Basis must have at least 1 quadrature point");

  CeedCall(CeedBasisGetTopologyDimension(topo, &dim));
  curl_comp = (dim < 3) ? 1 : dim;

  CeedCall(CeedCalloc(1, basis));
  CeedCall(CeedReferenceCopy(ceed, &(*basis)->ceed));
  (*basis)->ref_count       = 1;
  (*basis)->is_tensor_basis = false;
  (*basis)->dim             = dim;
  (*basis)->topo            = topo;
  (*basis)->num_comp        = num_comp;
  (*basis)->P               = P;
  (*basis)->Q               = Q;
  (*basis)->fe_space        = CEED_FE_SPACE_HCURL;
  CeedCall(CeedMalloc(Q * dim, &(*basis)->q_ref_1d));
  CeedCall(CeedMalloc(Q, &(*basis)->q_weight_1d));
  if (q_ref) memcpy((*basis)->q_ref_1d, q_ref, Q * dim * sizeof(q_ref[0]));
  if (q_weight) memcpy((*basis)->q_weight_1d, q_weight, Q * sizeof(q_weight[0]));
  CeedCall(CeedMalloc(dim * Q * P, &(*basis)->interp));
  CeedCall(CeedMalloc(curl_comp * Q * P, &(*basis)->curl));
  if (interp) memcpy((*basis)->interp, interp, dim * Q * P * sizeof(interp[0]));
  if (curl) memcpy((*basis)->curl, curl, curl_comp * Q * P * sizeof(curl[0]));
  CeedCall(ceed->BasisCreateHcurl(topo, dim, P, Q, interp, curl, q_ref, q_weight, *basis));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a CeedBasis for projection from the nodes of `basis_from` to the nodes of `basis_to`.

  Only `CEED_EVAL_INTERP` will be valid for the new basis, `basis_project`.
  For H^1 spaces, `CEED_EVAL_GRAD` will also be valid.
  The interpolation is given by `interp_project = interp_to^+ * interp_from`, where the pseudoinverse `interp_to^+` is given by QR
factorization.
  The gradient (for the H^1 case) is given by `grad_project = interp_to^+ * grad_from`.

  Note: `basis_from` and `basis_to` must have compatible quadrature spaces.

  Note: `basis_project` will have the same number of components as `basis_from`, regardless of the number of components that `basis_to` has.
        If `basis_from` has 3 components and `basis_to` has 5 components, then `basis_project` will have 3 components.

  @param[in]  basis_from    CeedBasis to prolong from
  @param[in]  basis_to      CeedBasis to prolong to
  @param[out] basis_project Address of the variable where the newly created CeedBasis will be stored.

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedBasisCreateProjection(CeedBasis basis_from, CeedBasis basis_to, CeedBasis *basis_project) {
  Ceed        ceed;
  bool        is_tensor;
  CeedInt     dim, num_comp;
  CeedScalar *q_ref, *q_weight, *interp_project, *grad_project;

  CeedCall(CeedBasisGetCeed(basis_to, &ceed));

  // Create projection matrix
  CeedCall(CeedBasisCreateProjectionMatrices(basis_from, basis_to, &interp_project, &grad_project));

  // Build basis
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
    // Even if basis_to and basis_from are not H1, the resulting basis is H1 for interpolation to work
    CeedInt          num_nodes_to, num_nodes_from;
    CeedElemTopology topo;

    CeedCall(CeedBasisGetTopology(basis_to, &topo));
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
  @brief Copy the pointer to a CeedBasis.

  Note: If the value of `basis_copy` passed into this function is non-NULL, then it is assumed that `basis_copy` is a pointer to a CeedBasis.
        This CeedBasis will be destroyed if `basis_copy` is the only reference to this CeedBasis.

  @param[in]     basis      CeedBasis to copy reference to
  @param[in,out] basis_copy Variable to store copied reference

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedBasisReferenceCopy(CeedBasis basis, CeedBasis *basis_copy) {
  if (basis != CEED_BASIS_NONE) CeedCall(CeedBasisReference(basis));
  CeedCall(CeedBasisDestroy(basis_copy));
  *basis_copy = basis;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief View a CeedBasis

  @param[in] basis  CeedBasis to view
  @param[in] stream Stream to view to, e.g., stdout

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedBasisView(CeedBasis basis, FILE *stream) {
  CeedInt          q_comp   = 0;
  CeedElemTopology topo     = basis->topo;
  CeedFESpace      fe_space = basis->fe_space;

  // Print FE space and element topology of the basis
  fprintf(stream, "CeedBasis in a %s on a %s element\n", CeedFESpaces[fe_space], CeedElemTopologies[topo]);
  if (basis->is_tensor_basis) {
    fprintf(stream, "  P: %" CeedInt_FMT "\n  Q: %" CeedInt_FMT "\n", basis->P_1d, basis->Q_1d);
  } else {
    fprintf(stream, "  P: %" CeedInt_FMT "\n  Q: %" CeedInt_FMT "\n", basis->P, basis->Q);
  }
  fprintf(stream, "  dimension: %" CeedInt_FMT "\n  field components: %" CeedInt_FMT "\n", basis->dim, basis->num_comp);
  // Print quadrature data, interpolation/gradient/divergence/curl of the basis
  if (basis->is_tensor_basis) {  // tensor basis
    CeedCall(CeedScalarView("qref1d", "\t% 12.8f", 1, basis->Q_1d, basis->q_ref_1d, stream));
    CeedCall(CeedScalarView("qweight1d", "\t% 12.8f", 1, basis->Q_1d, basis->q_weight_1d, stream));
    CeedCall(CeedScalarView("interp1d", "\t% 12.8f", basis->Q_1d, basis->P_1d, basis->interp_1d, stream));
    CeedCall(CeedScalarView("grad1d", "\t% 12.8f", basis->Q_1d, basis->P_1d, basis->grad_1d, stream));
  } else {  // non-tensor basis
    CeedCall(CeedScalarView("qref", "\t% 12.8f", 1, basis->Q * basis->dim, basis->q_ref_1d, stream));
    CeedCall(CeedScalarView("qweight", "\t% 12.8f", 1, basis->Q, basis->q_weight_1d, stream));
    CeedCall(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_INTERP, &q_comp));
    CeedCall(CeedScalarView("interp", "\t% 12.8f", q_comp * basis->Q, basis->P, basis->interp, stream));
    if (basis->grad) {
      CeedCall(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_GRAD, &q_comp));
      CeedCall(CeedScalarView("grad", "\t% 12.8f", q_comp * basis->Q, basis->P, basis->grad, stream));
    }
    if (basis->div) {
      CeedCall(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_DIV, &q_comp));
      CeedCall(CeedScalarView("div", "\t% 12.8f", q_comp * basis->Q, basis->P, basis->div, stream));
    }
    if (basis->curl) {
      CeedCall(CeedBasisGetNumQuadratureComponents(basis, CEED_EVAL_CURL, &q_comp));
      CeedCall(CeedScalarView("curl", "\t% 12.8f", q_comp * basis->Q, basis->P, basis->curl, stream));
    }
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Apply basis evaluation from nodes to quadrature points or vice versa

  @param[in]  basis      CeedBasis to evaluate
  @param[in]  num_elem   The number of elements to apply the basis evaluation to;
                           the backend will specify the ordering in CeedElemRestrictionCreateBlocked()
  @param[in]  t_mode    \ref CEED_NOTRANSPOSE to evaluate from nodes to quadrature points;
                          \ref CEED_TRANSPOSE to apply the transpose, mapping from quadrature points to nodes
  @param[in]  eval_mode \ref CEED_EVAL_NONE to use values directly,
                          \ref CEED_EVAL_INTERP to use interpolated values,
                          \ref CEED_EVAL_GRAD to use gradients,
                          \ref CEED_EVAL_DIV to use divergence,
                          \ref CEED_EVAL_CURL to use curl,
                          \ref CEED_EVAL_WEIGHT to use quadrature weights.
  @param[in]  u        Input CeedVector
  @param[out] v        Output CeedVector

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedBasisApply(CeedBasis basis, CeedInt num_elem, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector u, CeedVector v) {
  CeedInt  dim, num_comp, q_comp, num_nodes, num_qpts;
  CeedSize u_length = 0, v_length;

  CeedCall(CeedBasisGetDimension(basis, &dim));
  CeedCall(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCall(CeedBasisGetNumQuadratureComponents(basis, eval_mode, &q_comp));
  CeedCall(CeedBasisGetNumNodes(basis, &num_nodes));
  CeedCall(CeedBasisGetNumQuadraturePoints(basis, &num_qpts));
  CeedCall(CeedVectorGetLength(v, &v_length));
  if (u) CeedCall(CeedVectorGetLength(u, &u_length));

  CeedCheck(basis->Apply, basis->ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support BasisApply");

  // Check compatibility of topological and geometrical dimensions
  CeedCheck((t_mode == CEED_TRANSPOSE && v_length % num_nodes == 0 && u_length % num_qpts == 0) ||
                (t_mode == CEED_NOTRANSPOSE && u_length % num_nodes == 0 && v_length % num_qpts == 0),
            basis->ceed, CEED_ERROR_DIMENSION, "Length of input/output vectors incompatible with basis dimensions");

  // Check vector lengths to prevent out of bounds issues
  bool good_dims = true;
  switch (eval_mode) {
    case CEED_EVAL_NONE:
    case CEED_EVAL_INTERP:
    case CEED_EVAL_GRAD:
    case CEED_EVAL_DIV:
    case CEED_EVAL_CURL:
      good_dims =
          ((t_mode == CEED_TRANSPOSE && u_length >= num_elem * num_comp * num_qpts * q_comp && v_length >= num_elem * num_comp * num_nodes) ||
           (t_mode == CEED_NOTRANSPOSE && v_length >= num_elem * num_qpts * num_comp * q_comp && u_length >= num_elem * num_comp * num_nodes));
      break;
    case CEED_EVAL_WEIGHT:
      good_dims = v_length >= num_elem * num_qpts;
      break;
  }
  CeedCheck(good_dims, basis->ceed, CEED_ERROR_DIMENSION, "Input/output vectors too short for basis and evaluation mode");

  CeedCall(basis->Apply(basis, num_elem, t_mode, eval_mode, u, v));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Apply basis evaluation from nodes to arbitrary points

  @param[in]  basis      CeedBasis to evaluate
  @param[in]  num_points The number of points to apply the basis evaluation to
  @param[in]  t_mode    \ref CEED_NOTRANSPOSE to evaluate from nodes to points;
                          \ref CEED_TRANSPOSE to apply the transpose, mapping from points to nodes
  @param[in]  eval_mode \ref CEED_EVAL_INTERP to use interpolated values,
                          \ref CEED_EVAL_GRAD to use gradients
  @param[in]  x_ref    CeedVector holding reference coordinates of each point
  @param[in]  u        Input CeedVector, of length `num_nodes * num_comp` for `CEED_NOTRANSPOSE`
  @param[out] v        Output CeedVector, of length `num_points * num_q_comp` for `CEED_NOTRANSPOSE` with `CEED_EVAL_INTERP`

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedBasisApplyAtPoints(CeedBasis basis, CeedInt num_points, CeedTransposeMode t_mode, CeedEvalMode eval_mode, CeedVector x_ref, CeedVector u,
                           CeedVector v) {
  CeedInt  dim, num_comp, num_q_comp, num_nodes, P_1d = 1, Q_1d = 1;
  CeedSize x_length = 0, u_length = 0, v_length;

  CeedCall(CeedBasisGetDimension(basis, &dim));
  CeedCall(CeedBasisGetNumNodes1D(basis, &P_1d));
  CeedCall(CeedBasisGetNumQuadraturePoints1D(basis, &Q_1d));
  CeedCall(CeedBasisGetNumComponents(basis, &num_comp));
  CeedCall(CeedBasisGetNumQuadratureComponents(basis, eval_mode, &num_q_comp));
  CeedCall(CeedBasisGetNumNodes(basis, &num_nodes));
  CeedCall(CeedVectorGetLength(x_ref, &x_length));
  CeedCall(CeedVectorGetLength(v, &v_length));
  CeedCall(CeedVectorGetLength(u, &u_length));

  // Check compatibility of topological and geometrical dimensions
  CeedCheck((t_mode == CEED_TRANSPOSE && v_length % num_nodes == 0) || (t_mode == CEED_NOTRANSPOSE && u_length % num_nodes == 0), basis->ceed,
            CEED_ERROR_DIMENSION, "Length of input/output vectors incompatible with basis dimensions and number of points");

  // Check compatibility coordinates vector
  CeedCheck(x_length >= num_points * dim, basis->ceed, CEED_ERROR_DIMENSION,
            "Length of reference coordinate vector incompatible with basis dimension and number of points");

  // Check vector lengths to prevent out of bounds issues
  bool good_dims = false;
  switch (eval_mode) {
    case CEED_EVAL_INTERP:
      good_dims = ((t_mode == CEED_TRANSPOSE && (u_length >= num_points * num_q_comp || v_length >= num_nodes * num_comp)) ||
                   (t_mode == CEED_NOTRANSPOSE && (v_length >= num_points * num_q_comp || u_length >= num_nodes * num_comp)));
      break;
    case CEED_EVAL_GRAD:
      good_dims = ((t_mode == CEED_TRANSPOSE && (u_length >= num_points * num_q_comp * dim || v_length >= num_nodes * num_comp)) ||
                   (t_mode == CEED_NOTRANSPOSE && (v_length >= num_points * num_q_comp * dim || u_length >= num_nodes * num_comp)));
      break;
    case CEED_EVAL_NONE:
    case CEED_EVAL_WEIGHT:
    case CEED_EVAL_DIV:
    case CEED_EVAL_CURL:
      // LCOV_EXCL_START
      return CeedError(basis->ceed, CEED_ERROR_UNSUPPORTED, "Evaluation at arbitrary points not supported for %s", CeedEvalModes[eval_mode]);
      // LCOV_EXCL_STOP
  }
  CeedCheck(good_dims, basis->ceed, CEED_ERROR_DIMENSION, "Input/output vectors too short for basis and evaluation mode");

  // Backend method
  if (basis->ApplyAtPoints) {
    CeedCall(basis->ApplyAtPoints(basis, num_points, t_mode, eval_mode, x_ref, u, v));
    return CEED_ERROR_SUCCESS;
  }

  // Default implementation
  CeedCheck(basis->is_tensor_basis, basis->ceed, CEED_ERROR_UNSUPPORTED, "Evaluation at arbitrary points only supported for tensor product bases");
  CeedCheck(eval_mode == CEED_EVAL_INTERP || t_mode == CEED_NOTRANSPOSE, basis->ceed, CEED_ERROR_UNSUPPORTED, "%s evaluation only supported for %s",
            CeedEvalModes[eval_mode], CeedTransposeModes[CEED_NOTRANSPOSE]);
  if (!basis->basis_chebyshev) {
    // Build matrix mapping from quadrature point values to Chebyshev coefficients
    CeedScalar       *tau, *C, *I, *chebyshev_coeffs_1d;
    const CeedScalar *q_ref_1d;

    // Build coefficient matrix
    // -- Note: Clang-tidy needs this check because it does not understand the is_tensor_basis check above
    CeedCheck(P_1d > 0 && Q_1d > 0, basis->ceed, CEED_ERROR_INCOMPATIBLE, "Basis dimensions are malformed");
    CeedCall(CeedCalloc(Q_1d * Q_1d, &C));
    CeedCall(CeedBasisGetQRef(basis, &q_ref_1d));
    for (CeedInt i = 0; i < Q_1d; i++) CeedCall(CeedChebyshevPolynomialsAtPoint(q_ref_1d[i], Q_1d, &C[i * Q_1d]));

    // Inverse of coefficient matrix
    CeedCall(CeedCalloc(Q_1d * Q_1d, &chebyshev_coeffs_1d));
    CeedCall(CeedCalloc(Q_1d * Q_1d, &I));
    CeedCall(CeedCalloc(Q_1d, &tau));
    // -- QR Factorization, C = Q R
    CeedCall(CeedQRFactorization(basis->ceed, C, tau, Q_1d, Q_1d));
    // -- chebyshev_coeffs_1d = R_inv Q^T
    for (CeedInt i = 0; i < Q_1d; i++) I[i * Q_1d + i] = 1.0;
    // ---- Apply R_inv, chebyshev_coeffs_1d = I R_inv
    for (CeedInt i = 0; i < Q_1d; i++) {  // Row i
      chebyshev_coeffs_1d[Q_1d * i] = I[Q_1d * i] / C[0];
      for (CeedInt j = 1; j < Q_1d; j++) {  // Column j
        chebyshev_coeffs_1d[j + Q_1d * i] = I[j + Q_1d * i];
        for (CeedInt k = 0; k < j; k++) chebyshev_coeffs_1d[j + Q_1d * i] -= C[j + Q_1d * k] * chebyshev_coeffs_1d[k + Q_1d * i];
        chebyshev_coeffs_1d[j + Q_1d * i] /= C[j + Q_1d * j];
      }
    }
    // ---- Apply Q^T, chebyshev_coeffs_1d = R_inv Q^T
    CeedCall(CeedHouseholderApplyQ(chebyshev_coeffs_1d, C, tau, CEED_NOTRANSPOSE, Q_1d, Q_1d, Q_1d, 1, Q_1d));

    // Build basis mapping from nodes to Chebyshev coefficients
    CeedScalar       *chebyshev_interp_1d, *chebyshev_grad_1d, *chebyshev_q_weight_1d;
    const CeedScalar *interp_1d;

    CeedCall(CeedCalloc(P_1d * Q_1d, &chebyshev_interp_1d));
    CeedCall(CeedCalloc(P_1d * Q_1d, &chebyshev_grad_1d));
    CeedCall(CeedCalloc(Q_1d, &chebyshev_q_weight_1d));
    CeedCall(CeedBasisGetInterp1D(basis, &interp_1d));
    CeedCall(CeedMatrixMatrixMultiply(basis->ceed, chebyshev_coeffs_1d, interp_1d, chebyshev_interp_1d, Q_1d, P_1d, Q_1d));

    CeedCall(CeedVectorCreate(basis->ceed, num_comp * CeedIntPow(Q_1d, dim), &basis->vec_chebyshev));
    CeedCall(CeedBasisCreateTensorH1(basis->ceed, dim, num_comp, P_1d, Q_1d, chebyshev_interp_1d, chebyshev_grad_1d, q_ref_1d, chebyshev_q_weight_1d,
                                     &basis->basis_chebyshev));

    // Cleanup
    CeedCall(CeedFree(&C));
    CeedCall(CeedFree(&chebyshev_coeffs_1d));
    CeedCall(CeedFree(&I));
    CeedCall(CeedFree(&tau));
    CeedCall(CeedFree(&chebyshev_interp_1d));
    CeedCall(CeedFree(&chebyshev_grad_1d));
    CeedCall(CeedFree(&chebyshev_q_weight_1d));
  }

  // Create TensorContract object if needed, such as a basis from the GPU backends
  if (!basis->contract) {
    Ceed      ceed_ref;
    CeedBasis basis_ref = NULL;

    CeedCall(CeedInit("/cpu/self", &ceed_ref));
    // Only need matching tensor contraction dimensions, any type of basis will work
    CeedCall(CeedBasisCreateTensorH1Lagrange(ceed_ref, dim, num_comp, P_1d, Q_1d, CEED_GAUSS, &basis_ref));
    // Note - clang-tidy doesn't know basis_ref->contract must be valid here
    CeedCheck(basis_ref && basis_ref->contract, basis->ceed, CEED_ERROR_UNSUPPORTED,
              "Reference CPU ceed failed to create a tensor contraction object");
    CeedCall(CeedTensorContractReferenceCopy(basis_ref->contract, &basis->contract));
    CeedCall(CeedBasisDestroy(&basis_ref));
    CeedCall(CeedDestroy(&ceed_ref));
  }

  // Basis evaluation
  switch (t_mode) {
    case CEED_NOTRANSPOSE: {
      // Nodes to arbitrary points
      CeedScalar       *v_array;
      const CeedScalar *chebyshev_coeffs, *x_array_read;

      // -- Interpolate to Chebyshev coefficients
      CeedCall(CeedBasisApply(basis->basis_chebyshev, 1, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, u, basis->vec_chebyshev));

      // -- Evaluate Chebyshev polynomials at arbitrary points
      CeedCall(CeedVectorGetArrayRead(basis->vec_chebyshev, CEED_MEM_HOST, &chebyshev_coeffs));
      CeedCall(CeedVectorGetArrayRead(x_ref, CEED_MEM_HOST, &x_array_read));
      CeedCall(CeedVectorGetArrayWrite(v, CEED_MEM_HOST, &v_array));
      switch (eval_mode) {
        case CEED_EVAL_INTERP: {
          CeedScalar tmp[2][num_comp * CeedIntPow(Q_1d, dim)], chebyshev_x[Q_1d];

          // ---- Values at point
          for (CeedInt p = 0; p < num_points; p++) {
            CeedInt pre = num_comp * CeedIntPow(Q_1d, dim - 1), post = 1;

            for (CeedInt d = 0; d < dim; d++) {
              // ------ Tensor contract with current Chebyshev polynomial values
              CeedCall(CeedChebyshevPolynomialsAtPoint(x_array_read[p * dim + d], Q_1d, chebyshev_x));
              CeedCall(CeedTensorContractApply(basis->contract, pre, Q_1d, post, 1, chebyshev_x, t_mode, false,
                                               d == 0 ? chebyshev_coeffs : tmp[d % 2], d == (dim - 1) ? &v_array[p * num_comp] : tmp[(d + 1) % 2]));
              pre /= Q_1d;
              post *= 1;
            }
          }
          break;
        }
        case CEED_EVAL_GRAD: {
          CeedScalar tmp[2][num_comp * CeedIntPow(Q_1d, dim)], chebyshev_x[Q_1d];

          // ---- Values at point
          for (CeedInt p = 0; p < num_points; p++) {
            // Dim**2 contractions, apply grad when pass == dim
            for (CeedInt pass = 0; pass < dim; pass++) {
              CeedInt pre = num_comp * CeedIntPow(Q_1d, dim - 1), post = 1;

              for (CeedInt d = 0; d < dim; d++) {
                // ------ Tensor contract with current Chebyshev polynomial values
                if (pass == d) CeedCall(CeedChebyshevDerivativeAtPoint(x_array_read[p * dim + d], Q_1d, chebyshev_x));
                else CeedCall(CeedChebyshevPolynomialsAtPoint(x_array_read[p * dim + d], Q_1d, chebyshev_x));
                CeedCall(CeedTensorContractApply(basis->contract, pre, Q_1d, post, 1, chebyshev_x, t_mode, false,
                                                 d == 0 ? chebyshev_coeffs : tmp[d % 2],
                                                 d == (dim - 1) ? &v_array[p * num_comp * dim + pass] : tmp[(d + 1) % 2]));
                pre /= Q_1d;
                post *= 1;
              }
            }
          }
          break;
        }
        default:
          // Nothing to do, this won't occur
          break;
      }
      CeedCall(CeedVectorRestoreArrayRead(basis->vec_chebyshev, &chebyshev_coeffs));
      CeedCall(CeedVectorRestoreArrayRead(x_ref, &x_array_read));
      CeedCall(CeedVectorRestoreArray(v, &v_array));
      break;
    }
    case CEED_TRANSPOSE: {
      // Note: No switch on e_mode here because only CEED_EVAL_INTERP is supported at this time
      // Arbitrary points to nodes
      CeedScalar       *chebyshev_coeffs;
      const CeedScalar *u_array, *x_array_read;

      // -- Transpose of evaluation of Chebyshev polynomials at arbitrary points
      CeedCall(CeedVectorGetArrayWrite(basis->vec_chebyshev, CEED_MEM_HOST, &chebyshev_coeffs));
      CeedCall(CeedVectorGetArrayRead(x_ref, CEED_MEM_HOST, &x_array_read));
      CeedCall(CeedVectorGetArrayRead(u, CEED_MEM_HOST, &u_array));
      {
        CeedScalar tmp[2][num_comp * CeedIntPow(Q_1d, dim)], chebyshev_x[Q_1d];

        // ---- Values at point
        for (CeedInt p = 0; p < num_points; p++) {
          CeedInt pre = num_comp * 1, post = 1;

          for (CeedInt d = 0; d < dim; d++) {
            // ------ Tensor contract with current Chebyshev polynomial values
            CeedCall(CeedChebyshevPolynomialsAtPoint(x_array_read[p * dim + d], Q_1d, chebyshev_x));
            CeedCall(CeedTensorContractApply(basis->contract, pre, 1, post, Q_1d, chebyshev_x, t_mode, p > 0 && d == (dim - 1),
                                             d == 0 ? &u_array[p * num_comp] : tmp[d % 2], d == (dim - 1) ? chebyshev_coeffs : tmp[(d + 1) % 2]));
            pre /= 1;
            post *= Q_1d;
          }
        }
      }
      CeedCall(CeedVectorRestoreArray(basis->vec_chebyshev, &chebyshev_coeffs));
      CeedCall(CeedVectorRestoreArrayRead(x_ref, &x_array_read));
      CeedCall(CeedVectorRestoreArrayRead(u, &u_array));

      // -- Interpolate transpose from Chebyshev coefficients
      CeedCall(CeedBasisApply(basis->basis_chebyshev, 1, CEED_TRANSPOSE, CEED_EVAL_INTERP, basis->vec_chebyshev, v));
      break;
    }
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get Ceed associated with a CeedBasis

  @param[in]  basis CeedBasis
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

  @param[in]  basis CeedBasis
  @param[out] dim   Variable to store dimension of basis

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetDimension(CeedBasis basis, CeedInt *dim) {
  *dim = basis->dim;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get topology for given CeedBasis

  @param[in]  basis CeedBasis
  @param[out] topo  Variable to store topology of basis

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetTopology(CeedBasis basis, CeedElemTopology *topo) {
  *topo = basis->topo;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get number of components for given CeedBasis

  @param[in]  basis    CeedBasis
  @param[out] num_comp Variable to store number of components of basis

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetNumComponents(CeedBasis basis, CeedInt *num_comp) {
  *num_comp = basis->num_comp;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get total number of nodes (in dim dimensions) of a CeedBasis

  @param[in]  basis CeedBasis
  @param[out] P     Variable to store number of nodes

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedBasisGetNumNodes(CeedBasis basis, CeedInt *P) {
  *P = basis->P;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get total number of nodes (in 1 dimension) of a CeedBasis

  @param[in]  basis CeedBasis
  @param[out] P_1d  Variable to store number of nodes

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetNumNodes1D(CeedBasis basis, CeedInt *P_1d) {
  CeedCheck(basis->is_tensor_basis, basis->ceed, CEED_ERROR_MINOR, "Cannot supply P_1d for non-tensor basis");
  *P_1d = basis->P_1d;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get total number of quadrature points (in dim dimensions) of a CeedBasis

  @param[in]  basis CeedBasis
  @param[out] Q     Variable to store number of quadrature points

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedBasisGetNumQuadraturePoints(CeedBasis basis, CeedInt *Q) {
  *Q = basis->Q;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get total number of quadrature points (in 1 dimension) of a CeedBasis

  @param[in]  basis CeedBasis
  @param[out] Q_1d  Variable to store number of quadrature points

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetNumQuadraturePoints1D(CeedBasis basis, CeedInt *Q_1d) {
  CeedCheck(basis->is_tensor_basis, basis->ceed, CEED_ERROR_MINOR, "Cannot supply Q_1d for non-tensor basis");
  *Q_1d = basis->Q_1d;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get reference coordinates of quadrature points (in dim dimensions) of a CeedBasis

  @param[in]  basis CeedBasis
  @param[out] q_ref Variable to store reference coordinates of quadrature points

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetQRef(CeedBasis basis, const CeedScalar **q_ref) {
  *q_ref = basis->q_ref_1d;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get quadrature weights of quadrature points (in dim dimensions) of a CeedBasis

  @param[in]  basis    CeedBasis
  @param[out] q_weight Variable to store quadrature weights

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetQWeights(CeedBasis basis, const CeedScalar **q_weight) {
  *q_weight = basis->q_weight_1d;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get interpolation matrix of a CeedBasis

  @param[in]  basis  CeedBasis
  @param[out] interp Variable to store interpolation matrix

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetInterp(CeedBasis basis, const CeedScalar **interp) {
  if (!basis->interp && basis->is_tensor_basis) {
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

  @param[in]  basis     CeedBasis
  @param[out] interp_1d Variable to store interpolation matrix

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedBasisGetInterp1D(CeedBasis basis, const CeedScalar **interp_1d) {
  CeedCheck(basis->is_tensor_basis, basis->ceed, CEED_ERROR_MINOR, "CeedBasis is not a tensor product basis.");
  *interp_1d = basis->interp_1d;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get gradient matrix of a CeedBasis

  @param[in]  basis CeedBasis
  @param[out] grad  Variable to store gradient matrix

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetGrad(CeedBasis basis, const CeedScalar **grad) {
  if (!basis->grad && basis->is_tensor_basis) {
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

  @param[in]  basis   CeedBasis
  @param[out] grad_1d Variable to store gradient matrix

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetGrad1D(CeedBasis basis, const CeedScalar **grad_1d) {
  CeedCheck(basis->is_tensor_basis, basis->ceed, CEED_ERROR_MINOR, "CeedBasis is not a tensor product basis.");
  *grad_1d = basis->grad_1d;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get divergence matrix of a CeedBasis

  @param[in]  basis CeedBasis
  @param[out] div   Variable to store divergence matrix

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetDiv(CeedBasis basis, const CeedScalar **div) {
  CeedCheck(basis->div, basis->ceed, CEED_ERROR_MINOR, "CeedBasis does not have divergence matrix.");
  *div = basis->div;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get curl matrix of a CeedBasis

  @param[in]  basis CeedBasis
  @param[out] curl  Variable to store curl matrix

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedBasisGetCurl(CeedBasis basis, const CeedScalar **curl) {
  CeedCheck(basis->curl, basis->ceed, CEED_ERROR_MINOR, "CeedBasis does not have curl matrix.");
  *curl = basis->curl;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Destroy a CeedBasis

  @param[in,out] basis CeedBasis to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedBasisDestroy(CeedBasis *basis) {
  if (!*basis || *basis == CEED_BASIS_NONE || --(*basis)->ref_count > 0) {
    *basis = NULL;
    return CEED_ERROR_SUCCESS;
  }
  if ((*basis)->Destroy) CeedCall((*basis)->Destroy(*basis));
  CeedCall(CeedTensorContractDestroy(&(*basis)->contract));
  CeedCall(CeedFree(&(*basis)->q_ref_1d));
  CeedCall(CeedFree(&(*basis)->q_weight_1d));
  CeedCall(CeedFree(&(*basis)->interp));
  CeedCall(CeedFree(&(*basis)->interp_1d));
  CeedCall(CeedFree(&(*basis)->grad));
  CeedCall(CeedFree(&(*basis)->grad_1d));
  CeedCall(CeedFree(&(*basis)->div));
  CeedCall(CeedFree(&(*basis)->curl));
  CeedCall(CeedVectorDestroy(&(*basis)->vec_chebyshev));
  CeedCall(CeedBasisDestroy(&(*basis)->basis_chebyshev));
  CeedCall(CeedDestroy(&(*basis)->ceed));
  CeedCall(CeedFree(basis));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Construct a Gauss-Legendre quadrature

  @param[in]  Q           Number of quadrature points (integrates polynomials of degree 2*Q-1 exactly)
  @param[out] q_ref_1d    Array of length Q to hold the abscissa on [-1, 1]
  @param[out] q_weight_1d Array of length Q to hold the weights

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedGaussQuadrature(CeedInt Q, CeedScalar *q_ref_1d, CeedScalar *q_weight_1d) {
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

  @param[in]  Q           Number of quadrature points (integrates polynomials of degree 2*Q-3 exactly)
  @param[out] q_ref_1d    Array of length Q to hold the abscissa on [-1, 1]
  @param[out] q_weight_1d Array of length Q to hold the weights

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedLobattoQuadrature(CeedInt Q, CeedScalar *q_ref_1d, CeedScalar *q_weight_1d) {
  CeedScalar P0, P1, P2, dP2, d2P2, xi, wi, PI = 4.0 * atan(1.0);

  // Build q_ref_1d, q_weight_1d
  // Set endpoints
  CeedCheck(Q > 1, NULL, CEED_ERROR_DIMENSION, "Cannot create Lobatto quadrature with Q=%" CeedInt_FMT " < 2 points", Q);
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

/// @}
