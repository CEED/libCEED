// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef utils_h
#define utils_h

#include <ceed.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

CEED_QFUNCTION_HELPER CeedScalar Max(CeedScalar a, CeedScalar b) { return a < b ? b : a; }
CEED_QFUNCTION_HELPER CeedScalar Min(CeedScalar a, CeedScalar b) { return a < b ? a : b; }

CEED_QFUNCTION_HELPER void SwapScalar(CeedScalar *a, CeedScalar *b) {
  CeedScalar temp = *a;
  *a              = *b;
  *b              = temp;
}

CEED_QFUNCTION_HELPER CeedScalar Square(CeedScalar x) { return x * x; }
CEED_QFUNCTION_HELPER CeedScalar Cube(CeedScalar x) { return x * x * x; }

// @brief Scale vector of length N by scalar alpha
CEED_QFUNCTION_HELPER void ScaleN(CeedScalar *u, const CeedScalar alpha, const CeedInt N) {
  CeedPragmaSIMD for (CeedInt i = 0; i < N; i++) u[i] *= alpha;
}

// @brief Set vector of length N to a value alpha
CEED_QFUNCTION_HELPER void SetValueN(CeedScalar *u, const CeedScalar alpha, const CeedInt N) {
  CeedPragmaSIMD for (CeedInt i = 0; i < N; i++) u[i] = alpha;
}

// @brief Copy N elements from x to y
CEED_QFUNCTION_HELPER void CopyN(const CeedScalar *x, CeedScalar *y, const CeedInt N) { CeedPragmaSIMD for (CeedInt i = 0; i < N; i++) y[i] = x[i]; }

// @brief Copy 3x3 matrix from A to B
CEED_QFUNCTION_HELPER void CopyMat3(const CeedScalar A[3][3], CeedScalar B[3][3]) { CopyN((const CeedScalar *)A, (CeedScalar *)B, 9); }

// @brief Dot product of vectors with N elements
CEED_QFUNCTION_HELPER CeedScalar DotN(const CeedScalar *u, const CeedScalar *v, const CeedInt N) {
  CeedScalar output = 0;
  CeedPragmaSIMD for (CeedInt i = 0; i < N; i++) output += u[i] * v[i];
  return output;
}

// @brief Dot product of 3 element vectors
CEED_QFUNCTION_HELPER CeedScalar Dot3(const CeedScalar *u, const CeedScalar *v) { return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]; }

// @brief Cross product of vectors with 3 elements
CEED_QFUNCTION_HELPER void Cross3(const CeedScalar u[3], const CeedScalar v[3], CeedScalar w[3]) {
  w[0] = (u[1] * v[2]) - (u[2] * v[1]);
  w[1] = (u[2] * v[0]) - (u[0] * v[2]);
  w[2] = (u[0] * v[1]) - (u[1] * v[0]);
}

// @brief Curl of vector given its gradient
CEED_QFUNCTION_HELPER void Curl3(const CeedScalar gradient[3][3], CeedScalar v[3]) {
  v[0] = gradient[2][1] - gradient[1][2];
  v[1] = gradient[0][2] - gradient[2][0];
  v[2] = gradient[1][0] - gradient[0][1];
}

// @brief Matrix vector product, b = Ax + b. A is NxM, x is M, b is N
CEED_QFUNCTION_HELPER void MatVecNM(const CeedScalar *A, const CeedScalar *x, const CeedInt N, const CeedInt M, const CeedTransposeMode transpose_A,
                                    CeedScalar *b) {
  switch (transpose_A) {
    case CEED_NOTRANSPOSE:
      CeedPragmaSIMD for (CeedInt i = 0; i < N; i++) b[i] += DotN(&A[i * M], x, M);
      break;
    case CEED_TRANSPOSE:
      CeedPragmaSIMD for (CeedInt i = 0; i < M; i++) { CeedPragmaSIMD for (CeedInt j = 0; j < N; j++) b[i] += A[j * M + i] * x[j]; }
      break;
  }
}

// @brief 3x3 Matrix vector product  b = Ax + b.
CEED_QFUNCTION_HELPER void MatVec3(const CeedScalar A[3][3], const CeedScalar x[3], const CeedTransposeMode transpose_A, CeedScalar b[3]) {
  MatVecNM((const CeedScalar *)A, (const CeedScalar *)x, 3, 3, transpose_A, (CeedScalar *)b);
}

// @brief Matrix-Matrix product, B = DA + B, where D is diagonal.
// @details A is NxM, D is diagonal NxN, represented by a vector of length N, and B is NxM. Optionally, A may be transposed.
CEED_QFUNCTION_HELPER void MatDiagNM(const CeedScalar *A, const CeedScalar *D, const CeedInt N, const CeedInt M, const CeedTransposeMode transpose_A,
                                     CeedScalar *B) {
  switch (transpose_A) {
    case CEED_NOTRANSPOSE:
      CeedPragmaSIMD for (CeedInt i = 0; i < N; i++) { CeedPragmaSIMD for (CeedInt j = 0; j < M; j++) B[i * M + j] += D[i] * A[i * M + j]; }
      break;
    case CEED_TRANSPOSE:
      CeedPragmaSIMD for (CeedInt i = 0; i < M; i++) { CeedPragmaSIMD for (CeedInt j = 0; j < N; j++) B[i * N + j] += D[i] * A[j * M + i]; }
      break;
  }
}

// @brief 3x3 Matrix-Matrix product, B = DA + B, where D is diagonal.
// @details Optionally, A may be transposed.
CEED_QFUNCTION_HELPER void MatDiag3(const CeedScalar A[3][3], const CeedScalar D[3], const CeedTransposeMode transpose_A, CeedScalar B[3][3]) {
  MatDiagNM((const CeedScalar *)A, (const CeedScalar *)D, 3, 3, transpose_A, (CeedScalar *)B);
}
// @brief NxN Matrix-Matrix product, C = AB + C
CEED_QFUNCTION_HELPER void MatMatN(const CeedScalar *A, const CeedScalar *B, const CeedInt N, const CeedTransposeMode transpose_A,
                                   const CeedTransposeMode transpose_B, CeedScalar *C) {
  switch (transpose_A) {
    case CEED_NOTRANSPOSE:
      switch (transpose_B) {
        case CEED_NOTRANSPOSE:
          CeedPragmaSIMD for (CeedInt i = 0; i < N; i++) {
            CeedPragmaSIMD for (CeedInt j = 0; j < N; j++) {
              CeedPragmaSIMD for (CeedInt k = 0; k < N; k++) C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
          }
          break;
        case CEED_TRANSPOSE:
          CeedPragmaSIMD for (CeedInt i = 0; i < N; i++) {
            CeedPragmaSIMD for (CeedInt j = 0; j < N; j++) {
              CeedPragmaSIMD for (CeedInt k = 0; k < N; k++) C[i * N + j] += A[i * N + k] * B[j * N + k];
            }
          }
          break;
      }
      break;
    case CEED_TRANSPOSE:
      switch (transpose_B) {
        case CEED_NOTRANSPOSE:
          CeedPragmaSIMD for (CeedInt i = 0; i < N; i++) {
            CeedPragmaSIMD for (CeedInt j = 0; j < N; j++) {
              CeedPragmaSIMD for (CeedInt k = 0; k < N; k++) C[i * N + j] += A[k * N + i] * B[k * N + j];
            }
          }
          break;
        case CEED_TRANSPOSE:
          CeedPragmaSIMD for (CeedInt i = 0; i < N; i++) {
            CeedPragmaSIMD for (CeedInt j = 0; j < N; j++) {
              CeedPragmaSIMD for (CeedInt k = 0; k < N; k++) C[i * N + j] += A[k * N + i] * B[j * N + k];
            }
          }
          break;
      }
      break;
  }
}

// @brief 3x3 Matrix-Matrix product, C = AB + C
CEED_QFUNCTION_HELPER void MatMat3(const CeedScalar A[3][3], const CeedScalar B[3][3], const CeedTransposeMode transpose_A,
                                   const CeedTransposeMode transpose_B, CeedScalar C[3][3]) {
  MatMatN((const CeedScalar *)A, (const CeedScalar *)B, 3, transpose_A, transpose_B, (CeedScalar *)C);
}

// @brief Unpack Kelvin-Mandel notation symmetric tensor into full tensor
CEED_QFUNCTION_HELPER void KMUnpack(const CeedScalar v[6], CeedScalar A[3][3]) {
  const CeedScalar weight = 1 / sqrt(2.);
  A[0][0]                 = v[0];
  A[1][1]                 = v[1];
  A[2][2]                 = v[2];
  A[2][1] = A[1][2] = weight * v[3];
  A[2][0] = A[0][2] = weight * v[4];
  A[1][0] = A[0][1] = weight * v[5];
}

// @brief Pack full tensor into Kelvin-Mandel notation symmetric tensor
CEED_QFUNCTION_HELPER void KMPack(const CeedScalar A[3][3], CeedScalar v[6]) {
  const CeedScalar weight = sqrt(2.);
  v[0]                    = A[0][0];
  v[1]                    = A[1][1];
  v[2]                    = A[2][2];
  v[3]                    = A[2][1] * weight;
  v[4]                    = A[2][0] * weight;
  v[5]                    = A[1][0] * weight;
}

// @brief Calculate metric tensor from mapping, g_{ij} = xi_{k,i} xi_{k,j} = dXdx^T dXdx
CEED_QFUNCTION_HELPER void KMMetricTensor(const CeedScalar dXdx[3][3], CeedScalar km_g_ij[6]) {
  CeedScalar g_ij[3][3] = {{0.}};
  MatMat3(dXdx, dXdx, CEED_TRANSPOSE, CEED_NOTRANSPOSE, g_ij);
  KMPack(g_ij, km_g_ij);
}

// @brief Linear ramp evaluation
CEED_QFUNCTION_HELPER CeedScalar LinearRampCoefficient(CeedScalar amplitude, CeedScalar length, CeedScalar start, CeedScalar x) {
  if (x < start) {
    return amplitude;
  } else if (x < start + length) {
    return amplitude * ((x - start) * (-1 / length) + 1);
  } else {
    return 0;
  }
}

/**
  @brief Pack stored values at quadrature point

  @param[in]   Q              Number of quadrature points
  @param[in]   i              Current quadrature point
  @param[in]   start          Starting index to store components
  @param[in]   num_comp       Number of components to store
  @param[in]   values_at_qpnt Local values for quadrature point i
  @param[out]  stored         Stored values

  @return An error code: 0 - success, otherwise - failure
**/
CEED_QFUNCTION_HELPER int StoredValuesPack(CeedInt Q, CeedInt i, CeedInt start, CeedInt num_comp, const CeedScalar *values_at_qpnt,
                                           CeedScalar *stored) {
  for (CeedInt j = 0; j < num_comp; j++) stored[(start + j) * Q + i] = values_at_qpnt[j];

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Unpack stored values at quadrature point

  @param[in]   Q              Number of quadrature points
  @param[in]   i              Current quadrature point
  @param[in]   start          Starting index to store components
  @param[in]   num_comp       Number of components to store
  @param[in]   stored         Stored values
  @param[out]  values_at_qpnt Local values for quadrature point i

  @return An error code: 0 - success, otherwise - failure
**/
CEED_QFUNCTION_HELPER int StoredValuesUnpack(CeedInt Q, CeedInt i, CeedInt start, CeedInt num_comp, const CeedScalar *stored,
                                             CeedScalar *values_at_qpnt) {
  for (CeedInt j = 0; j < num_comp; j++) values_at_qpnt[j] = stored[(start + j) * Q + i];

  return CEED_ERROR_SUCCESS;
}

/**
  @brief Unpack 3D element q_data at quadrature point

  @param[in]   Q         Number of quadrature points
  @param[in]   i         Current quadrature point
  @param[in]   q_data    Pointer to q_data (generated by `setupgeo.h:Setup`)
  @param[out]  wdetJ     Quadrature weight times determinant of the mapping Jacobian
  @param[out]  dXdx      Inverse of the mapping Jacobian (shape [3][3])

  @return An error code: 0 - success, otherwise - failure
**/
CEED_QFUNCTION_HELPER int QdataUnpack_3D(CeedInt Q, CeedInt i, const CeedScalar *q_data, CeedScalar *wdetJ, CeedScalar dXdx[3][3]) {
  StoredValuesUnpack(Q, i, 0, 1, q_data, wdetJ);
  StoredValuesUnpack(Q, i, 1, 9, q_data, (CeedScalar *)dXdx);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Unpack boundary element q_data for 3D problem at quadrature point

  @param[in]   Q         Number of quadrature points
  @param[in]   i         Current quadrature point
  @param[in]   q_data    Pointer to q_data (generated by `setupgeo.h:Setup`)
  @param[out]  wdetJ     Quadrature weight times determinant of the mapping Jacobian, or `NULL`
  @param[out]  dXdx      Inverse of the mapping Jacobian (shape [2][3]), or `NULL`
  @param[out]  normal    Components of the normal vector (shape [3]), or `NULL`

  @return An error code: 0 - success, otherwise - failure
**/
CEED_QFUNCTION_HELPER int QdataBoundaryUnpack_3D(CeedInt Q, CeedInt i, const CeedScalar *q_data, CeedScalar *wdetJ, CeedScalar dXdx[2][3],
                                                 CeedScalar normal[3]) {
  if (wdetJ) StoredValuesUnpack(Q, i, 0, 1, q_data, wdetJ);
  if (normal) StoredValuesUnpack(Q, i, 1, 3, q_data, normal);
  if (dXdx) StoredValuesUnpack(Q, i, 4, 6, q_data, (CeedScalar *)dXdx);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Unpack 2D element q_data at quadrature point

  @param[in]   Q         Number of quadrature points
  @param[in]   i         Current quadrature point
  @param[in]   q_data    Pointer to q_data (generated by `setupgeo.h:Setup`)
  @param[out]  wdetJ     Quadrature weight times determinant of the mapping Jacobian
  @param[out]  dXdx      Inverse of the mapping Jacobian (shape [2][2])

  @return An error code: 0 - success, otherwise - failure
**/
CEED_QFUNCTION_HELPER int QdataUnpack_2D(CeedInt Q, CeedInt i, const CeedScalar *q_data, CeedScalar *wdetJ, CeedScalar dXdx[2][2]) {
  StoredValuesUnpack(Q, i, 0, 1, q_data, wdetJ);
  StoredValuesUnpack(Q, i, 1, 4, q_data, (CeedScalar *)dXdx);
  return CEED_ERROR_SUCCESS;
}

#endif  // utils_h
