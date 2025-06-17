// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for CUDA shared memory tensor product basis AtPoints templates
#include <ceed/types.h>

//------------------------------------------------------------------------------
// Chebyshev values
//------------------------------------------------------------------------------
template <int Q_1D>
inline __device__ void ChebyshevPolynomialsAtPoint(const CeedScalar x, CeedScalar *chebyshev_x) {
  chebyshev_x[0] = 1.0;
  chebyshev_x[1] = 2 * x;
  for (CeedInt i = 2; i < Q_1D; i++) chebyshev_x[i] = 2 * x * chebyshev_x[i - 1] - chebyshev_x[i - 2];
}

template <int Q_1D>
inline __device__ void ChebyshevDerivativeAtPoint(const CeedScalar x, CeedScalar *chebyshev_dx) {
  CeedScalar chebyshev_x[3];

  chebyshev_x[1]  = 1.0;
  chebyshev_x[2]  = 2 * x;
  chebyshev_dx[0] = 0.0;
  chebyshev_dx[1] = 2.0;
  for (CeedInt i = 2; i < Q_1D; i++) {
    chebyshev_x[(i + 1) % 3] = 2 * x * chebyshev_x[(i + 0) % 3] - chebyshev_x[(i + 2) % 3];
    chebyshev_dx[i]          = 2 * x * chebyshev_dx[i - 1] + 2 * chebyshev_x[(i + 0) % 3] - chebyshev_dx[i - 2];
  }
}

//------------------------------------------------------------------------------
// 1D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// 1D interpolate to points
//------------------------------------------------------------------------------
template <int NUM_COMP, int NUM_POINTS, int P_1D, int Q_1D>
inline __device__ void InterpAtPoints1d(SharedData_Cuda &data, const CeedInt p, const CeedScalar *__restrict__ r_C, const CeedScalar *r_X,
                                        CeedScalar *__restrict__ r_V) {
  CeedScalar chebyshev_x[Q_1D];

  for (CeedInt i = 0; i < NUM_COMP; i++) r_V[i] = 0.0;
  ChebyshevPolynomialsAtPoint<Q_1D>(r_X[0], chebyshev_x);
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    // Load coefficients
    if (data.t_id_x < Q_1D) data.slice[data.t_id_x] = r_C[comp];
    __syncthreads();
    // Contract x direction
    for (CeedInt i = 0; i < Q_1D; i++) {
      r_V[comp] += chebyshev_x[i] * data.slice[i];
    }
  }
}

//------------------------------------------------------------------------------
// 1D interpolate transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int NUM_POINTS, int P_1D, int Q_1D>
inline __device__ void InterpTransposeAtPoints1d(SharedData_Cuda &data, const CeedInt p, const CeedScalar *__restrict__ r_U, const CeedScalar *r_X,
                                                 CeedScalar *__restrict__ r_C) {
  CeedScalar chebyshev_x[Q_1D];

  ChebyshevPolynomialsAtPoint<Q_1D>(r_X[0], chebyshev_x);
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    // Clear shared memory
    if (data.t_id_x < Q_1D) data.slice[data.t_id_x] = 0.0;
    __syncthreads();
    // Contract x direction
    if (p < NUM_POINTS) {
      for (CeedInt i = 0; i < Q_1D; i++) {
        atomicAdd(&data.slice[comp * Q_1D + (i + data.t_id_x) % Q_1D], chebyshev_x[(i + data.t_id_x) % Q_1D] * r_U[comp]);
      }
    }
    // Pull from shared to register
    __syncthreads();
    if (data.t_id_x < Q_1D) r_C[comp] += data.slice[data.t_id_x];
  }
}

//------------------------------------------------------------------------------
// 1D derivatives at points
//------------------------------------------------------------------------------
template <int NUM_COMP, int NUM_POINTS, int P_1D, int Q_1D>
inline __device__ void GradAtPoints1d(SharedData_Cuda &data, const CeedInt p, const CeedScalar *__restrict__ r_C, const CeedScalar *r_X,
                                      CeedScalar *__restrict__ r_V) {
  CeedScalar chebyshev_x[Q_1D];

  ChebyshevDerivativeAtPoint<Q_1D>(r_X[0], chebyshev_x);
  for (CeedInt i = 0; i < NUM_COMP; i++) r_V[i] = 0.0;
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    // Load coefficients
    __syncthreads();
    if (data.t_id_x < Q_1D) data.slice[data.t_id_x] = r_C[comp];
    __syncthreads();
    // Contract x direction
    for (CeedInt i = 0; i < Q_1D; i++) {
      r_V[comp] += chebyshev_x[i] * data.slice[i];
    }
  }
}

//------------------------------------------------------------------------------
// 1D derivatives transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int NUM_POINTS, int P_1D, int Q_1D>
inline __device__ void GradTransposeAtPoints1d(SharedData_Cuda &data, const CeedInt p, const CeedScalar *__restrict__ r_U, const CeedScalar *r_X,
                                               CeedScalar *__restrict__ r_C) {
  CeedScalar chebyshev_x[Q_1D];

  ChebyshevDerivativeAtPoint<Q_1D>(r_X[0], chebyshev_x);
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    // Clear shared memory
    if (data.t_id_x < Q_1D) data.slice[data.t_id_x] = 0.0;
    __syncthreads();
    // Contract x direction
    if (p < NUM_POINTS) {
      for (CeedInt i = 0; i < Q_1D; i++) {
        atomicAdd(&data.slice[comp * Q_1D + (i + data.t_id_x) % Q_1D], chebyshev_x[(i + data.t_id_x) % Q_1D] * r_U[comp]);
      }
    }
    // Pull from shared to register
    __syncthreads();
    if (data.t_id_x < Q_1D) r_C[comp] += data.slice[data.t_id_x];
  }
}

//------------------------------------------------------------------------------
// 2D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// 2D interpolate to points
//------------------------------------------------------------------------------
template <int NUM_COMP, int NUM_POINTS, int P_1D, int Q_1D>
inline __device__ void InterpAtPoints2d(SharedData_Cuda &data, const CeedInt p, const CeedScalar *__restrict__ r_C, const CeedScalar *r_X,
                                        CeedScalar *__restrict__ r_V) {
  for (CeedInt i = 0; i < NUM_COMP; i++) r_V[i] = 0.0;
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    CeedScalar buffer[Q_1D];
    CeedScalar chebyshev_x[Q_1D];

    // Load coefficients
    __syncthreads();
    if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) data.slice[data.t_id_x + data.t_id_y * Q_1D] = r_C[comp];
    __syncthreads();
    // Contract x direction
    ChebyshevPolynomialsAtPoint<Q_1D>(r_X[0], chebyshev_x);
    for (CeedInt i = 0; i < Q_1D; i++) {
      buffer[i] = 0.0;
      for (CeedInt j = 0; j < Q_1D; j++) {
        buffer[i] += chebyshev_x[j] * data.slice[j + i * Q_1D];
      }
    }
    // Contract y direction
    ChebyshevPolynomialsAtPoint<Q_1D>(r_X[1], chebyshev_x);
    for (CeedInt i = 0; i < Q_1D; i++) {
      r_V[comp] += chebyshev_x[i] * buffer[i];
    }
  }
}

//------------------------------------------------------------------------------
// 2D interpolate transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int NUM_POINTS, int P_1D, int Q_1D>
inline __device__ void InterpTransposeAtPoints2d(SharedData_Cuda &data, const CeedInt p, const CeedScalar *__restrict__ r_U, const CeedScalar *r_X,
                                                 CeedScalar *__restrict__ r_C) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    CeedScalar buffer[Q_1D];
    CeedScalar chebyshev_x[Q_1D];

    // Clear shared memory
    if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) data.slice[data.t_id_x + data.t_id_y * Q_1D] = 0.0;
    __syncthreads();
    // Contract y direction
    ChebyshevPolynomialsAtPoint<Q_1D>(r_X[1], chebyshev_x);
    for (CeedInt i = 0; i < Q_1D; i++) {
      buffer[i] = chebyshev_x[i] * r_U[comp];
    }
    // Contract x direction
    ChebyshevPolynomialsAtPoint<Q_1D>(r_X[0], chebyshev_x);
    if (p < NUM_POINTS) {
      for (CeedInt i = 0; i < Q_1D; i++) {
        // Note: shifting to avoid atomic adds
        const CeedInt ii = (i + data.t_id_x) % Q_1D;

        for (CeedInt j = 0; j < Q_1D; j++) {
          const CeedInt jj = (j + data.t_id_y) % Q_1D;

          atomicAdd(&data.slice[jj + ii * Q_1D], chebyshev_x[jj] * buffer[ii]);
        }
      }
    }
    // Pull from shared to register
    __syncthreads();
    if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) r_C[comp] += data.slice[data.t_id_x + data.t_id_y * Q_1D];
  }
}

//------------------------------------------------------------------------------
// 2D derivatives at points
//------------------------------------------------------------------------------
template <int NUM_COMP, int NUM_POINTS, int P_1D, int Q_1D>
inline __device__ void GradAtPoints2d(SharedData_Cuda &data, const CeedInt p, const CeedScalar *__restrict__ r_C, const CeedScalar *r_X,
                                      CeedScalar *__restrict__ r_V) {
  for (CeedInt i = 0; i < NUM_COMP * 2; i++) r_V[i] = 0.0;
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    CeedScalar buffer[Q_1D];
    CeedScalar chebyshev_x[Q_1D];

    // Load coefficients
    __syncthreads();
    if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) data.slice[data.t_id_x + data.t_id_y * Q_1D] = r_C[comp];
    __syncthreads();
    for (CeedInt dim = 0; dim < 2; dim++) {
      // Contract x direction
      if (dim == 0) ChebyshevDerivativeAtPoint<Q_1D>(r_X[0], chebyshev_x);
      else ChebyshevPolynomialsAtPoint<Q_1D>(r_X[0], chebyshev_x);
      for (CeedInt i = 0; i < Q_1D; i++) {
        buffer[i] = 0.0;
        for (CeedInt j = 0; j < Q_1D; j++) {
          buffer[i] += chebyshev_x[j] * data.slice[j + i * Q_1D];
        }
      }
      // Contract y direction
      if (dim == 1) ChebyshevDerivativeAtPoint<Q_1D>(r_X[1], chebyshev_x);
      else ChebyshevPolynomialsAtPoint<Q_1D>(r_X[1], chebyshev_x);
      for (CeedInt i = 0; i < Q_1D; i++) {
        r_V[comp + dim * NUM_COMP] += chebyshev_x[i] * buffer[i];
      }
    }
  }
}

//------------------------------------------------------------------------------
// 2D derivatives transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int NUM_POINTS, int P_1D, int Q_1D>
inline __device__ void GradTransposeAtPoints2d(SharedData_Cuda &data, const CeedInt p, const CeedScalar *__restrict__ r_U, const CeedScalar *r_X,
                                               CeedScalar *__restrict__ r_C) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    CeedScalar buffer[Q_1D];
    CeedScalar chebyshev_x[Q_1D];

    // Clear shared memory
    if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) data.slice[data.t_id_x + data.t_id_y * Q_1D] = 0.0;
    __syncthreads();
    for (CeedInt dim = 0; dim < 2; dim++) {
      // Contract y direction
      if (dim == 1) ChebyshevDerivativeAtPoint<Q_1D>(r_X[1], chebyshev_x);
      else ChebyshevPolynomialsAtPoint<Q_1D>(r_X[1], chebyshev_x);
      for (CeedInt i = 0; i < Q_1D; i++) {
        buffer[i] = chebyshev_x[i] * r_U[comp + dim * NUM_COMP];
      }
      // Contract x direction
      if (dim == 0) ChebyshevDerivativeAtPoint<Q_1D>(r_X[0], chebyshev_x);
      else ChebyshevPolynomialsAtPoint<Q_1D>(r_X[0], chebyshev_x);
      if (p < NUM_POINTS) {
        for (CeedInt i = 0; i < Q_1D; i++) {
          // Note: shifting to avoid atomic adds
          const CeedInt ii = (i + data.t_id_x) % Q_1D;

          for (CeedInt j = 0; j < Q_1D; j++) {
            const CeedInt jj = (j + data.t_id_y) % Q_1D;

            atomicAdd(&data.slice[jj + ii * Q_1D], chebyshev_x[jj] * buffer[ii]);
          }
        }
      }
    }
    // Pull from shared to register
    __syncthreads();
    if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) r_C[comp] += data.slice[data.t_id_x + data.t_id_y * Q_1D];
  }
}

//------------------------------------------------------------------------------
// 3D
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// 3D interpolate to points
//------------------------------------------------------------------------------
template <int NUM_COMP, int NUM_POINTS, int P_1D, int Q_1D>
inline __device__ void InterpAtPoints3d(SharedData_Cuda &data, const CeedInt p, const CeedScalar *__restrict__ r_C, const CeedScalar *r_X,
                                        CeedScalar *__restrict__ r_V) {
  for (CeedInt i = 0; i < NUM_COMP; i++) r_V[i] = 0.0;
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    for (CeedInt k = 0; k < Q_1D; k++) {
      CeedScalar buffer[Q_1D];
      CeedScalar chebyshev_x[Q_1D];

      // Load coefficients
      __syncthreads();
      if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) data.slice[data.t_id_x + data.t_id_y * Q_1D] = r_C[k + comp * Q_1D];
      __syncthreads();
      // Contract x direction
      ChebyshevPolynomialsAtPoint<Q_1D>(r_X[0], chebyshev_x);
      for (CeedInt i = 0; i < Q_1D; i++) {
        buffer[i] = 0.0;
        for (CeedInt j = 0; j < Q_1D; j++) {
          buffer[i] += chebyshev_x[j] * data.slice[j + i * Q_1D];
        }
      }
      // Contract y and z direction
      ChebyshevPolynomialsAtPoint<Q_1D>(r_X[2], chebyshev_x);
      const CeedScalar z = chebyshev_x[k];

      ChebyshevPolynomialsAtPoint<Q_1D>(r_X[1], chebyshev_x);
      for (CeedInt i = 0; i < Q_1D; i++) {
        r_V[comp] += chebyshev_x[i] * buffer[i] * z;
      }
    }
  }
}

//------------------------------------------------------------------------------
// 3D interpolate transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int NUM_POINTS, int P_1D, int Q_1D>
inline __device__ void InterpTransposeAtPoints3d(SharedData_Cuda &data, const CeedInt p, const CeedScalar *__restrict__ r_U, const CeedScalar *r_X,
                                                 CeedScalar *__restrict__ r_C) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    for (CeedInt k = 0; k < Q_1D; k++) {
      CeedScalar buffer[Q_1D];
      CeedScalar chebyshev_x[Q_1D];

      // Clear shared memory
      if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) data.slice[data.t_id_x + data.t_id_y * Q_1D] = 0.0;
      __syncthreads();
      // Contract y and z direction
      ChebyshevPolynomialsAtPoint<Q_1D>(r_X[2], chebyshev_x);
      const CeedScalar z = chebyshev_x[k];

      ChebyshevPolynomialsAtPoint<Q_1D>(r_X[1], chebyshev_x);
      for (CeedInt i = 0; i < Q_1D; i++) {
        buffer[i] = chebyshev_x[i] * r_U[comp] * z;
      }
      // Contract x direction
      ChebyshevPolynomialsAtPoint<Q_1D>(r_X[0], chebyshev_x);
      if (p < NUM_POINTS) {
        for (CeedInt i = 0; i < Q_1D; i++) {
          // Note: shifting to avoid atomic adds
          const CeedInt ii = (i + data.t_id_x) % Q_1D;

          for (CeedInt j = 0; j < Q_1D; j++) {
            const CeedInt jj = (j + data.t_id_y) % Q_1D;

            atomicAdd(&data.slice[jj + ii * Q_1D], chebyshev_x[jj] * buffer[ii]);
          }
        }
      }
      // Pull from shared to register
      __syncthreads();
      if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) r_C[k + comp * Q_1D] += data.slice[data.t_id_x + data.t_id_y * Q_1D];
    }
  }
}

//------------------------------------------------------------------------------
// 3D derivatives at points
//------------------------------------------------------------------------------
template <int NUM_COMP, int NUM_POINTS, int P_1D, int Q_1D>
inline __device__ void GradAtPoints3d(SharedData_Cuda &data, const CeedInt p, const CeedScalar *__restrict__ r_C, const CeedScalar *r_X,
                                      CeedScalar *__restrict__ r_V) {
  for (CeedInt i = 0; i < NUM_COMP * 3; i++) r_V[i] = 0.0;
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    for (CeedInt k = 0; k < Q_1D; k++) {
      CeedScalar buffer[Q_1D];
      CeedScalar chebyshev_x[Q_1D];

      // Load coefficients
      __syncthreads();
      if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) data.slice[data.t_id_x + data.t_id_y * Q_1D] = r_C[k + comp * Q_1D];
      __syncthreads();
      for (CeedInt dim = 0; dim < 3; dim++) {
        // Contract x direction
        if (dim == 0) ChebyshevDerivativeAtPoint<Q_1D>(r_X[0], chebyshev_x);
        else ChebyshevPolynomialsAtPoint<Q_1D>(r_X[0], chebyshev_x);
        for (CeedInt i = 0; i < Q_1D; i++) {
          buffer[i] = 0.0;
          for (CeedInt j = 0; j < Q_1D; j++) {
            buffer[i] += chebyshev_x[j] * data.slice[j + i * Q_1D];
          }
        }
        // Contract y and z direction
        if (dim == 2) ChebyshevDerivativeAtPoint<Q_1D>(r_X[2], chebyshev_x);
        else ChebyshevPolynomialsAtPoint<Q_1D>(r_X[2], chebyshev_x);
        const CeedScalar z = chebyshev_x[k];

        if (dim == 1) ChebyshevDerivativeAtPoint<Q_1D>(r_X[1], chebyshev_x);
        else ChebyshevPolynomialsAtPoint<Q_1D>(r_X[1], chebyshev_x);
        for (CeedInt i = 0; i < Q_1D; i++) {
          r_V[comp + dim * NUM_COMP] += chebyshev_x[i] * buffer[i] * z;
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
// 3D derivatives transpose
//------------------------------------------------------------------------------
template <int NUM_COMP, int NUM_POINTS, int P_1D, int Q_1D>
inline __device__ void GradTransposeAtPoints3d(SharedData_Cuda &data, const CeedInt p, const CeedScalar *__restrict__ r_U, const CeedScalar *r_X,
                                               CeedScalar *__restrict__ r_C) {
  for (CeedInt comp = 0; comp < NUM_COMP; comp++) {
    for (CeedInt k = 0; k < Q_1D; k++) {
      CeedScalar buffer[Q_1D];
      CeedScalar chebyshev_x[Q_1D];

      // Clear shared memory
      if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) data.slice[data.t_id_x + data.t_id_y * Q_1D] = 0.0;
      __syncthreads();
      for (CeedInt dim = 0; dim < 3; dim++) {
        // Contract y and z direction
        if (dim == 2) ChebyshevDerivativeAtPoint<Q_1D>(r_X[2], chebyshev_x);
        else ChebyshevPolynomialsAtPoint<Q_1D>(r_X[2], chebyshev_x);
        const CeedScalar z = chebyshev_x[k];

        if (dim == 1) ChebyshevDerivativeAtPoint<Q_1D>(r_X[1], chebyshev_x);
        else ChebyshevPolynomialsAtPoint<Q_1D>(r_X[1], chebyshev_x);
        for (CeedInt i = 0; i < Q_1D; i++) {
          buffer[i] = chebyshev_x[i] * r_U[comp + dim * NUM_COMP] * z;
        }
        // Contract x direction
        if (dim == 0) ChebyshevDerivativeAtPoint<Q_1D>(r_X[0], chebyshev_x);
        else ChebyshevPolynomialsAtPoint<Q_1D>(r_X[0], chebyshev_x);
        if (p < NUM_POINTS) {
          for (CeedInt i = 0; i < Q_1D; i++) {
            // Note: shifting to avoid atomic adds
            const CeedInt ii = (i + data.t_id_x) % Q_1D;

            for (CeedInt j = 0; j < Q_1D; j++) {
              const CeedInt jj = (j + data.t_id_y) % Q_1D;

              atomicAdd(&data.slice[jj + ii * Q_1D], chebyshev_x[jj] * buffer[ii]);
            }
          }
        }
      }
      // Pull from shared to register
      __syncthreads();
      if (data.t_id_x < Q_1D && data.t_id_y < Q_1D) r_C[k + comp * Q_1D] += data.slice[data.t_id_x + data.t_id_y * Q_1D];
    }
  }
}
