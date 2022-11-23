// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../../kernel-defines.hpp"

const char *occa_tensor_basis_2d_gpu_source = STRINGIFY_SOURCE(

    typedef CeedScalar * dofArray @dim(P1D, P1D, BASIS_COMPONENT_COUNT, elementCount);
    typedef const CeedScalar *const_dofArray @dim(P1D, P1D, BASIS_COMPONENT_COUNT, elementCount);

    typedef CeedScalar * quadArray @dim(Q1D, Q1D, elementCount, BASIS_COMPONENT_COUNT, 2);
    typedef const CeedScalar *const_quadArray @dim(Q1D, Q1D, elementCount, BASIS_COMPONENT_COUNT, 2);

    typedef CeedScalar * sharedBufferArray @dim(MAX_PQ, MAX_PQ, ELEMENTS_PER_BLOCK); typedef const CeedScalar *quadToDof @dim(P1D, Q1D);
    typedef CeedScalar * elementWeightArray @dim(Q1D, Q1D, elementCount);

    //---[ Utility Methods ]----------------
    inline void readDofs(const int element, const int component, const int px, const int py, const_dofArray U, CeedScalar *Up) {
      // Zero out extra entries
      *Up = ((px < P1D) && (py < P1D) ? U(px, py, component, element) : 0.0);
    }

    inline void writeDofs(const int element, const int component, const int px, const int py, const CeedScalar Vp, dofArray V) {
      if ((px < P1D) && (py < P1D)) {
        V(px, py, component, element) = Vp;
      }
    }

    inline void readQuads(const int elementCount, const int element, const int component, const int qx, const int qy, const int dim,
                          const_quadArray U, CeedScalar *Uq) { *Uq = U(qx, qy, element, component, dim); }

    inline void writeQuads(const int elementCount, const int element, const int component, const int qx, const int qy, const int dim,
                           const CeedScalar Vq, quadArray V) { V(qx, qy, element, component, dim) = Vq; }

    inline void contractX(const int qx, const int qy, const int localElement, sharedBufferArray sharedBuffer, quadToDof B, const CeedScalar U,
                          CeedScalar *V) {
      sharedBuffer(qx, qy, localElement) = U;
      *V                                 = 0.0;
      @barrier();
      for (int p = 0; p < P1D; ++p) {
        *V += B(p, qx) * sharedBuffer(p, qy, localElement);
      }
      @barrier();
    }

    inline void contractY(const int qx, const int qy, const int localElement, sharedBufferArray sharedBuffer, quadToDof B, const CeedScalar U,
                          CeedScalar *V) {
      sharedBuffer(qx, qy, localElement) = U;
      *V                                 = 0.0;
      @barrier();
      for (int p = 0; p < P1D; ++p) {
        *V += B(p, qy) * sharedBuffer(qx, p, localElement);
      }
      @barrier();
    }

    inline void contractTransposeX(const int px, const int py, const int localElement, sharedBufferArray sharedBuffer, quadToDof B,
                                   const CeedScalar U, CeedScalar *V) {
      sharedBuffer(px, py, localElement) = U;
      *V                                 = 0.0;
      @barrier();
      for (int q = 0; q < Q1D; ++q) {
        *V += B(px, q) * sharedBuffer(q, py, localElement);
      }
      @barrier();
    }

    inline void contractTransposeY(const int px, const int py, const int localElement, sharedBufferArray sharedBuffer, quadToDof B,
                                   const CeedScalar U, CeedScalar *V) {
      sharedBuffer(px, py, localElement) = U;
      *V                                 = 0.0;
      @barrier();
      for (int q = 0; q < Q1D; ++q) {
        *V += B(py, q) * sharedBuffer(px, q, localElement);
      }
      @barrier();
    }

    //---[ Kernels ]------------------------
    @kernel void interp(const CeedInt elementCount, quadToDof B, const CeedScalar *U, CeedScalar *V) {
      for (int elementOffset = 0; elementOffset < elementCount; elementOffset += ELEMENTS_PER_BLOCK; @outer) {
        @shared CeedScalar sharedBuffer[MAX_PQ * MAX_PQ * ELEMENTS_PER_BLOCK];

        for (int localElement = 0; localElement < ELEMENTS_PER_BLOCK; ++localElement; @inner) {
          const int element = elementOffset + localElement;
          for (int qy = 0; qy < Q1D; ++qy; @inner) {
            for (int qx = 0; qx < Q1D; ++qx; @inner) {
              if (element < elementCount) {
                for (int component = 0; component < BASIS_COMPONENT_COUNT; ++component) {
                  CeedScalar r1, r2;
                  if (!TRANSPOSE) {
                    readDofs(element, component, qx, qy, U, &r1);
                    contractX(qx, qy, localElement, sharedBuffer, B, r1, &r2);
                    contractY(qx, qy, localElement, sharedBuffer, B, r2, &r1);
                    writeQuads(elementCount, element, component, qx, qy, 0, r1, V);
                  } else {
                    readQuads(elementCount, element, component, qx, qy, 0, U, &r1);
                    contractTransposeY(qx, qy, localElement, sharedBuffer, B, r1, &r2);
                    contractTransposeX(qx, qy, localElement, sharedBuffer, B, r2, &r1);
                    writeDofs(element, component, qx, qy, r1, V);
                  }
                }
              }
            }
          }
        }
      }
    }

    @kernel void grad(const CeedInt elementCount, quadToDof B, quadToDof Bx, const CeedScalar *U, CeedScalar *V) {
      for (int elementOffset = 0; elementOffset < elementCount; elementOffset += ELEMENTS_PER_BLOCK; @outer) {
        @shared CeedScalar sharedBuffer[MAX_PQ * MAX_PQ * ELEMENTS_PER_BLOCK];

        for (int localElement = 0; localElement < ELEMENTS_PER_BLOCK; ++localElement; @inner) {
          const int element = elementOffset + localElement;
          for (int qy = 0; qy < Q1D; ++qy; @inner) {
            for (int qx = 0; qx < Q1D; ++qx; @inner) {
              if (element < elementCount) {
                for (int component = 0; component < BASIS_COMPONENT_COUNT; ++component) {
                  CeedScalar r1, r2, r3;
                  if (!TRANSPOSE) {
                    readDofs(element, component, qx, qy, U, &r1);
                    contractX(qx, qy, localElement, sharedBuffer, Bx, r1, &r2);
                    contractY(qx, qy, localElement, sharedBuffer, B, r2, &r3);
                    writeQuads(elementCount, element, component, qx, qy, 0, r3, V);
                    contractX(qx, qy, localElement, sharedBuffer, B, r1, &r2);
                    contractY(qx, qy, localElement, sharedBuffer, Bx, r2, &r3);
                    writeQuads(elementCount, element, component, qx, qy, 1, r3, V);
                  } else {
                    readQuads(elementCount, element, component, qx, qy, 0, U, &r1);
                    contractTransposeY(qx, qy, localElement, sharedBuffer, B, r1, &r2);
                    contractTransposeX(qx, qy, localElement, sharedBuffer, Bx, r2, &r3);
                    readQuads(elementCount, element, component, qx, qy, 1, U, &r1);
                    contractTransposeY(qx, qy, localElement, sharedBuffer, Bx, r1, &r2);
                    contractTransposeX(qx, qy, localElement, sharedBuffer, B, r2, &r1);
                    writeDofs(element, component, qx, qy, r1 + r3, V);
                  }
                }
              }
            }
          }
        }
      }
    }

    @kernel void weight(const CeedInt elementCount, const CeedScalar *qWeights1D, elementWeightArray W) {
      for (int elementOffset = 0; elementOffset < elementCount; elementOffset += ELEMENTS_PER_BLOCK; @outer) {
        for (int element = elementOffset; element < (elementOffset + ELEMENTS_PER_BLOCK); ++element; @outer) {
          for (int qy = 0; qy < Q1D; ++qy; @inner) {
            for (int qx = 0; qx < Q1D; ++qx; @inner) {
              W(qx, qy, element) = qWeights1D[qx] * qWeights1D[qy];
            }
          }
        }
      }
    }

);
