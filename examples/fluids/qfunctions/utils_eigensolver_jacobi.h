// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Eigen system solver for symmetric NxN matrices. Modified from the CC0 code provided at https://github.com/jewettaij/jacobi_pd

#ifndef utils_eigensolver_jacobi_h
#define utils_eigensolver_jacobi_h

#include <ceed.h>
#include <math.h>

#include "utils.h"

// @typedef choose the criteria for sorting eigenvalues and eigenvectors
typedef enum eSortCriteria {
  SORT_NONE,
  SORT_DECREASING_EVALS,
  SORT_INCREASING_EVALS,
  SORT_DECREASING_ABS_EVALS,
  SORT_INCREASING_ABS_EVALS
} SortCriteria;

///@brief Find the off-diagonal index in row i whose absolute value is largest
///
/// @param[in] *A matrix
/// @param[in] i row index
/// @returns   Index of absolute largest off-diagonal element in row i
CEED_QFUNCTION_HELPER CeedInt MaxEntryRow(const CeedScalar *A, CeedInt N, CeedInt i) {
  CeedInt j_max = i + 1;
  for (CeedInt j = i + 2; j < N; j++)
    if (fabs(A[i * N + j]) > fabs(A[i * N + j_max])) j_max = j;
  return j_max;
}

/// @brief Find the indices (i_max, j_max) marking the location of the
///        entry in the matrix with the largest absolute value.  This
///        uses the max_idx_row[] array to find the answer in O(n) time.
///
/// @param[in]    *A    matrix
/// @param[inout] i_max row index
/// @param[inout] j_max column index
CEED_QFUNCTION_HELPER void MaxEntry(const CeedScalar *A, CeedInt N, CeedInt *max_idx_row, CeedInt *i_max, CeedInt *j_max) {
  *i_max               = 0;
  *j_max               = max_idx_row[*i_max];
  CeedScalar max_entry = fabs(A[*i_max * N + *j_max]);
  for (CeedInt i = 1; i < N - 1; i++) {
    CeedInt j = max_idx_row[i];
    if (fabs(A[i * N + j]) > max_entry) {
      max_entry = fabs(A[i * N + j]);
      *i_max    = i;
      *j_max    = j;
    }
  }
}

/// @brief Calculate the components of a rotation matrix which performs a
///        rotation in the i,j plane by an angle (θ) that (when multiplied on
///        both sides) will zero the ij'th element of A, so that afterwards
///        A[i][j] = 0.  The results will be stored in c, s, and t
///        (which store cos(θ), sin(θ), and tan(θ), respectively).
///
/// @param[in] *A matrix
/// @param[in] i row index
/// @param[in] j column index
CEED_QFUNCTION_HELPER void CalcRot(const CeedScalar *A, CeedInt N, CeedInt i, CeedInt j, CeedScalar *rotmat_cst) {
  rotmat_cst[2]      = 1.0;  // = tan(θ)
  CeedScalar A_jj_ii = (A[j * N + j] - A[i * N + i]);
  if (A_jj_ii != 0.0) {
    // kappa = (A[j][j] - A[i][i]) / (2*A[i][j])
    CeedScalar kappa = A_jj_ii;
    rotmat_cst[2]    = 0.0;
    CeedScalar A_ij  = A[i * N + j];
    if (A_ij != 0.0) {
      kappa /= (2.0 * A_ij);
      // t satisfies: t^2 + 2*t*kappa - 1 = 0
      // (choose the root which has the smaller absolute value)
      rotmat_cst[2] = 1.0 / (sqrt(1 + kappa * kappa) + fabs(kappa));
      if (kappa < 0.0) rotmat_cst[2] = -rotmat_cst[2];
    }
  }
  rotmat_cst[0] = 1.0 / sqrt(1 + rotmat_cst[2] * rotmat_cst[2]);
  rotmat_cst[1] = rotmat_cst[0] * rotmat_cst[2];
}

/// @brief  Perform a similarity transformation by multiplying matrix A on both
///         sides by a rotation matrix (and its transpose) to eliminate A[i][j].
/// @details This rotation matrix performs a rotation in the i,j plane by
///         angle θ.  This function assumes that c=cos(θ). s=sin(θ), t=tan(θ)
///         have been calculated in advance (using the CalcRot() function).
///         It also assumes that i<j.  The max_idx_row[] array is also updated.
///         To save time, since the matrix is symmetric, the elements
///         below the diagonal (ie. A[u][v] where u>v) are not computed.
/// @verbatim
///   A' = R^T * A * R
/// where R the rotation in the i,j plane and ^T denotes the transpose.
///                 i         j
///       _                             _
///      |  1                            |
///      |    .                          |
///      |      .                        |
///      |        1                      |
///      |          c   ...   s          |
///      |          .  .      .          |
/// R  = |          .    1    .          |
///      |          .      .  .          |
///      |          -s  ...   c          |
///      |                      1        |
///      |                        .      |
///      |                          .    |
///      |_                           1 _|
/// @endverbatim
///
/// Let A' denote the matrix A after multiplication by R^T and R.
/// The components of A' are:
///
/// @verbatim
///   A'_uv =  Σ_w  Σ_z   R_wu * A_wz * R_zv
/// @endverbatim
///
/// Note that a the rotation at location i,j will modify all of the matrix
/// elements containing at least one index which is either i or j
/// such as: A[w][i], A[i][w], A[w][j], A[j][w].
/// Check and see whether these modified matrix elements exceed the
/// corresponding values in max_idx_row[] array for that row.
/// If so, then update max_idx_row for that row.
/// This is somewhat complicated by the fact that we must only consider
/// matrix elements in the upper-right triangle strictly above the diagonal.
/// (ie. matrix elements whose second index is > the first index).
/// The modified elements we must consider are marked with an "X" below:
///
/// @verbatim
///                 i         j
///       _                             _
///      |  .       X         X          |
///      |    .     X         X          |
///      |      .   X         X          |
///      |        . X         X          |
///      |          X X X X X 0 X X X X  |  i
///      |            .       X          |
///      |              .     X          |
/// A  = |                .   X          |
///      |                  . X          |
///      |                    X X X X X  |  j
///      |                      .        |
///      |                        .      |
///      |                          .    |
///      |_                           . _|
/// @endverbatim
///
/// @param[in] *A matrix
/// @param[in] i row index
/// @param[in] j column index
CEED_QFUNCTION_HELPER void ApplyRot(CeedScalar *A, CeedInt N, CeedInt i, CeedInt j, CeedInt *max_idx_row, CeedScalar *rotmat_cst) {
  // Compute the diagonal elements of A which have changed:
  A[i * N + i] -= rotmat_cst[2] * A[i * N + j];
  A[j * N + j] += rotmat_cst[2] * A[i * N + j];
  // Note: This is algebraically equivalent to:
  // A[i][i] = c*c*A[i][i] + s*s*A[j][j] - 2*s*c*A[i][j]
  // A[j][j] = s*s*A[i][i] + c*c*A[j][j] + 2*s*c*A[i][j]

  // Update the off-diagonal elements of A which will change (above the diagonal)

  A[i * N + j] = 0.0;

  // compute A[w][i] and A[i][w] for all w!=i,considering above-diagonal elements
  for (CeedInt w = 0; w < i; w++) {                                              // 0 <= w <  i  <  j < N
    A[i * N + w] = A[w * N + i];                                                 // backup the previous value. store below diagonal (i>w)
    A[w * N + i] = rotmat_cst[0] * A[w * N + i] - rotmat_cst[1] * A[w * N + j];  // A[w][i], A[w][j] from previous iteration
    if (i == max_idx_row[w]) max_idx_row[w] = MaxEntryRow(A, N, w);
    else if (fabs(A[w * N + i]) > fabs(A[w * N + max_idx_row[w]])) max_idx_row[w] = i;
  }
  for (CeedInt w = i + 1; w < j; w++) {                                          // 0 <= i <  w  <  j < N
    A[w * N + i] = A[i * N + w];                                                 // backup the previous value. store below diagonal (w>i)
    A[i * N + w] = rotmat_cst[0] * A[i * N + w] - rotmat_cst[1] * A[w * N + j];  // A[i][w], A[w][j] from previous iteration
  }
  for (CeedInt w = j + 1; w < N; w++) {                                          // 0 <= i < j+1 <= w < N
    A[w * N + i] = A[i * N + w];                                                 // backup the previous value. store below diagonal (w>i)
    A[i * N + w] = rotmat_cst[0] * A[i * N + w] - rotmat_cst[1] * A[j * N + w];  // A[i][w], A[j][w] from previous iteration
  }

  // now that we're done modifying row i, we can update max_idx_row[i]
  max_idx_row[i] = MaxEntryRow(A, N, i);

  // compute A[w][j] and A[j][w] for all w!=j,considering above-diagonal elements
  for (CeedInt w = 0; w < i; w++) {                                              // 0 <=  w  <  i <  j < N
    A[w * N + j] = rotmat_cst[1] * A[i * N + w] + rotmat_cst[0] * A[w * N + j];  // A[i][w], A[w][j] from previous iteration
    if (j == max_idx_row[w]) max_idx_row[w] = MaxEntryRow(A, N, w);
    else if (fabs(A[w * N + j]) > fabs(A[w * N + max_idx_row[w]])) max_idx_row[w] = j;
  }
  for (CeedInt w = i + 1; w < j; w++) {                                          // 0 <= i+1 <= w <  j < N
    A[w * N + j] = rotmat_cst[1] * A[w * N + i] + rotmat_cst[0] * A[w * N + j];  // A[w][i], A[w][j] from previous iteration
    if (j == max_idx_row[w]) max_idx_row[w] = MaxEntryRow(A, N, w);
    else if (fabs(A[w * N + j]) > fabs(A[w * N + max_idx_row[w]])) max_idx_row[w] = j;
  }
  for (CeedInt w = j + 1; w < N; w++) {                                          // 0 <=  i  <  j <  w < N
    A[j * N + w] = rotmat_cst[1] * A[w * N + i] + rotmat_cst[0] * A[j * N + w];  // A[w][i], A[j][w] from previous iteration
  }
  // now that we're done modifying row j, we can update max_idx_row[j]
  max_idx_row[j] = MaxEntryRow(A, N, j);
}

///@brief Multiply matrix A on the LEFT side by a transposed rotation matrix R^T
///       This matrix performs a rotation in the i,j plane by angle θ  (where
///       the arguments "s" and "c" refer to cos(θ) and sin(θ), respectively).
/// @verbatim
///   A'_uv = Σ_w  R_wu * A_wv
/// @endverbatim
///
/// @param[in] *A matrix
/// @param[in] i row index
/// @param[in] j column index
CEED_QFUNCTION_HELPER void ApplyRotLeft(CeedScalar *A, CeedInt N, CeedInt i, CeedInt j, CeedScalar *rotmat_cst) {
  // Recall that c = cos(θ) and s = sin(θ)
  for (CeedInt v = 0; v < N; v++) {
    CeedScalar Aiv = A[i * N + v];
    A[i * N + v]   = rotmat_cst[0] * A[i * N + v] - rotmat_cst[1] * A[j * N + v];
    A[j * N + v]   = rotmat_cst[1] * Aiv + rotmat_cst[0] * A[j * N + v];
  }
}

/// @brief Sort the rows in evec according to the numbers in v (also sorted)
///
/// @param[inout] *eval vector containing the keys used for sorting
/// @param[inout] *evec matrix whose rows will be sorted according to v
/// @param[in]    n  size of the vector and matrix
/// @param[in]    s  sort decreasing order?
CEED_QFUNCTION_HELPER void SortRows(CeedScalar *eval, CeedScalar *evec, CeedInt N, SortCriteria sort_criteria) {
  if (sort_criteria == SORT_NONE) return;

  for (CeedInt i = 0; i < N - 1; i++) {
    CeedInt i_max = i;
    for (CeedInt j = i + 1; j < N; j++) {
      // find the "maximum" element in the array starting at position i+1
      switch (sort_criteria) {
        case SORT_DECREASING_EVALS:
          if (eval[j] > eval[i_max]) i_max = j;
          break;
        case SORT_INCREASING_EVALS:
          if (eval[j] < eval[i_max]) i_max = j;
          break;
        case SORT_DECREASING_ABS_EVALS:
          if (fabs(eval[j]) > fabs(eval[i_max])) i_max = j;
          break;
        case SORT_INCREASING_ABS_EVALS:
          if (fabs(eval[j]) < fabs(eval[i_max])) i_max = j;
          break;
        default:
          break;
      }
    }
    SwapScalar(&eval[i], &eval[i_max]);
    for (CeedInt k = 0; k < N; k++) SwapScalar(&evec[i * N + k], &evec[i_max * N + k]);
  }
}

/// @brief Calculate all the eigenvalues and eigevectors of a symmetric matrix
///        using the Jacobi eigenvalue algorithm:
///        https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm
/// @returns The number of Jacobi iterations attempted, which should be > 0.
///          If the return value is not strictly > 0 then convergence failed.
/// @note  To reduce the computation time further, set calc_evecs=false.
///        Additionally, note that the output evecs should be normalized. It
///        simply takes the Identity matrix and performs (isometric) rotations
///        on it, so divergence from normalized is due to finite-precision
///        arithmetic of the rotations.
//
// @param[in]  A              the matrix you wish to diagonalize (size NxN)
// @param[in]  N              size of the matrix
// @param[out] eval           store the eigenvalues here (size N)
// @param[out] evec           store the eigenvectors here (in rows, size NxN)
// @param[out] max_idx_row    work vector of size N-1
// @param[in]  sort_criteria  sort results?
// @param[in]  calc_evecs     calculate the eigenvectors?
// @param[in]  max_num_sweeps maximum number of iterations = max_num_sweeps * number of off-diagonals (N*(N-1)/2)
CEED_QFUNCTION_HELPER CeedInt Diagonalize(CeedScalar *A, CeedInt N, CeedScalar *eval, CeedScalar *evec, CeedInt *max_idx_row,
                                          SortCriteria sort_criteria, bool calc_evec, const CeedInt max_num_sweeps) {
  CeedScalar rotmat_cst[3] = {0.};  // cos(θ), sin(θ), and tan(θ),

  if (calc_evec)
    for (CeedInt i = 0; i < N; i++)
      for (CeedInt j = 0; j < N; j++) evec[i * N + j] = (i == j) ? 1.0 : 0.0;  // Set evec equal to the identity matrix

  for (CeedInt i = 0; i < N - 1; i++) max_idx_row[i] = MaxEntryRow(A, N, i);

  // -- Iteration --
  CeedInt n_iters;
  CeedInt max_num_iters = max_num_sweeps * N * (N - 1) / 2;
  for (n_iters = 1; n_iters <= max_num_iters; n_iters++) {
    CeedInt i, j;
    MaxEntry(A, N, max_idx_row, &i, &j);

    // If A[i][j] is small compared to A[i][i] and A[j][j], set it to 0.
    if ((A[i * N + i] + A[i * N + j] == A[i * N + i]) && (A[j * N + j] + A[i * N + j] == A[j * N + j])) {
      A[i * N + j]   = 0.0;
      max_idx_row[i] = MaxEntryRow(A, N, i);
    }

    if (A[i * N + j] == 0.0) break;

    CalcRot(A, N, i, j, rotmat_cst);                // Calculate the parameters of the rotation matrix.
    ApplyRot(A, N, i, j, max_idx_row, rotmat_cst);  // Apply this rotation to the A matrix.
    if (calc_evec) ApplyRotLeft(evec, N, i, j, rotmat_cst);
  }

  for (CeedInt i = 0; i < N; i++) eval[i] = A[i * N + i];

  // Optional: Sort results by eigenvalue.
  SortRows(eval, evec, N, sort_criteria);

  if ((n_iters > max_num_iters) && (N > 1))  // If we exceeded max_num_iters,
    return 0;                                // indicate an error occured.

  return n_iters;
}

// @brief Interface to Diagonalize for 3x3 systems
CEED_QFUNCTION_HELPER CeedInt Diagonalize3(CeedScalar A[3][3], CeedScalar eval[3], CeedScalar evec[3][3], CeedInt max_idx_row[3],
                                           SortCriteria sort_criteria, bool calc_evec, const CeedInt max_num_sweeps) {
  return Diagonalize((CeedScalar *)A, 3, (CeedScalar *)eval, (CeedScalar *)evec, (CeedInt *)max_idx_row, sort_criteria, calc_evec, max_num_sweeps);
}

#endif  // utils_eigensolver_jacobi_h
