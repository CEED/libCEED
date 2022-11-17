// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../../kernel-defines.hpp"

const char *occa_tensor_basis_1d_gpu_source = STRINGIFY_SOURCE(

    typedef CeedScalar * dofArray @dim(P1D, BASIS_COMPONENT_COUNT, elementCount);
    typedef const CeedScalar *const_dofArray @dim(P1D, BASIS_COMPONENT_COUNT, elementCount);

    typedef CeedScalar * quadArray @dim(Q1D, elementCount, BASIS_COMPONENT_COUNT);
    typedef const CeedScalar *const_quadArray @dim(Q1D, elementCount, BASIS_COMPONENT_COUNT);

    typedef CeedScalar * sharedBufferArray @dim(MAX_PQ, ELEMENTS_PER_BLOCK); typedef const CeedScalar *quadToDof @dim(P1D, Q1D);
    typedef CeedScalar * elementWeightArray @dim(Q1D, elementCount);

    //---[ Utility Methods ]----------------
    inline void readDofs(const int element, const int localElement, const int component, const int p, const_dofArray U,
                         sharedBufferArray sharedBuffer) {
      // Zero out extra entries
      sharedBuffer(p, localElement) = ((p < P1D) ? U(p, component, element) : 0.0);
    }

    inline void writeDofs(const int element, const int component, const int p, const CeedScalar Vp, dofArray V) {
      if (p < P1D) {
        V(p, component, element) = Vp;
      }
    }

    inline void readQuads(const int elementCount, const int element, const int localElement, const int component, const int q, const_quadArray U,
                          sharedBufferArray sharedBuffer) { sharedBuffer(q, localElement) = U(q, element, component); }

    inline void writeQuads(const int elementCount, const int element, const int component, const int q, const CeedScalar Vq, quadArray V) {
      V(q, element, component) = Vq;
    }

    inline void contractX(const int q, const int localElement, sharedBufferArray sharedBuffer, quadToDof B, CeedScalar &V) {
      V = 0.0;
      for (int p = 0; p < P1D; ++p) {
        V += B(p, q) * sharedBuffer(p, localElement);
      }
    }

    inline void contractTransposeX(const int p, const int localElement, sharedBufferArray sharedBuffer, quadToDof B, CeedScalar &V) {
      V = 0.0;
      for (int q = 0; q < Q1D; ++q) {
        V += B(p, q) * sharedBuffer(q, localElement);
      }
    }

    //---[ Kernels ]------------------------
    @kernel void interp(const CeedInt elementCount, quadToDof B, const CeedScalar *U, CeedScalar *V) {
      for (int elementOffset = 0; elementOffset < elementCount; elementOffset += ELEMENTS_PER_BLOCK; @outer) {
        @shared CeedScalar sharedBuffer[MAX_PQ * ELEMENTS_PER_BLOCK];

        for (int localElement = 0; localElement < ELEMENTS_PER_BLOCK; ++localElement; @inner) {
          for (int q = 0; q < Q1D; ++q; @inner) {
            const int element = elementOffset + localElement;
            if (element < elementCount) {
              for (int component = 0; component < BASIS_COMPONENT_COUNT; ++component) {
                CeedScalar r;
                if (!TRANSPOSE) {
                  readDofs(element, localElement, component, q, U, sharedBuffer);
                  contractX(q, localElement, sharedBuffer, B, r);
                  writeQuads(elementCount, element, component, q, r, V);
                } else {
                  readQuads(elementCount, element, localElement, component, q, U, sharedBuffer);
                  contractTransposeX(q, localElement, sharedBuffer, B, r);
                  writeDofs(element, component, q, r, V);
                }
              }
            }
          }
        }
      }
    }

    @kernel void grad(const CeedInt elementCount, quadToDof B, quadToDof Bx, const CeedScalar *U, CeedScalar *V) {
      for (int elementOffset = 0; elementOffset < elementCount; elementOffset += ELEMENTS_PER_BLOCK; @outer) {
        @shared CeedScalar sharedBuffer[MAX_PQ * ELEMENTS_PER_BLOCK];

        for (int localElement = 0; localElement < ELEMENTS_PER_BLOCK; ++localElement; @inner) {
          for (int q = 0; q < Q1D; ++q; @inner) {
            const int element = elementOffset + localElement;
            if (element < elementCount) {
              for (int component = 0; component < BASIS_COMPONENT_COUNT; ++component) {
                CeedScalar r;
                if (!TRANSPOSE) {
                  readDofs(element, localElement, component, q, U, sharedBuffer);
                  contractX(q, localElement, sharedBuffer, Bx, r);
                  writeQuads(elementCount, element, component, q, r, V);
                } else {
                  readQuads(elementCount, element, localElement, component, q, U, sharedBuffer);
                  contractTransposeX(q, localElement, sharedBuffer, Bx, r);
                  writeDofs(element, component, q, r, V);
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
          for (int q = 0; q < Q1D; ++q; @inner) {
            W(q, element) = qWeights1D[q];
          }
        }
      }
    }

);
