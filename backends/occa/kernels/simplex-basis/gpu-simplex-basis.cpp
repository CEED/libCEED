// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../kernel-defines.hpp"

const char *occa_simplex_basis_gpu_source = STRINGIFY_SOURCE(

    @directive("#if TRANSPOSE") typedef CeedScalar * dofArray @dim(Q, elementCount, BASIS_COMPONENT_COUNT, DIM);
    typedef CeedScalar * quadArray @dim(P, BASIS_COMPONENT_COUNT, elementCount, DIM);
    @directive("#else") typedef CeedScalar * dofArray @dim(P, BASIS_COMPONENT_COUNT, elementCount, DIM);
    typedef CeedScalar * quadArray @dim(Q, elementCount, BASIS_COMPONENT_COUNT, DIM); @directive("#endif")

                                                                                          typedef CeedScalar *
                                                                                      quadToDof @dim(P, Q);
    typedef CeedScalar * dQuadToDof @dim(P, Q, DIM); typedef CeedScalar * elementWeightArray @dim(Q, elementCount);

    @kernel void interp(const CeedInt elementCount, const quadToDof B, const dofArray U, quadArray V) {
      for (int elementOffset = 0; elementOffset < elementCount; elementOffset += ELEMENTS_PER_BLOCK; @outer) {
        @shared CeedScalar s_B[P * Q] @dim(P, Q);

        // Store weights in shared memory
        for (int i = 0; i < MAX_PQ; ++i; @inner) {
          for (int j = i; j < (P * Q); j += MAX_PQ) {
            s_B[j] = B[j];
          }
        }

        for (int localElement = 0; localElement < ELEMENTS_PER_BLOCK; ++localElement) {
          for (int i = 0; i < MAX_PQ; ++i; @inner) {
            const int element = elementOffset + localElement;
            if (element < elementCount) {
              // Element operation
              for (int component = 0; component < BASIS_COMPONENT_COUNT; ++component) {
                if (!TRANSPOSE) {
                  const int q = i;
                  if (q < Q) {
                    CeedScalar v = 0;
                    for (int p = 0; p < P; ++p) {
                      v += s_B(p, q) * U(p, component, element, 0);
                    }
                    V(q, element, component, 0) = v;
                  }
                } else {
                  const int p = i;
                  if (p < P) {
                    CeedScalar v = 0;
                    for (int q = 0; q < Q; ++q) {
                      v += s_B(p, q) * U(q, element, component, 0);
                    }
                    V(p, component, element, 0) = v;
                  }
                }
              }
            }
          }
        }
      }
    }

    @kernel void grad(const CeedInt elementCount, const dQuadToDof Bx, const dofArray U, quadArray V) {
      for (int elementOffset = 0; elementOffset < elementCount; elementOffset += ELEMENTS_PER_BLOCK; @outer) {
        @shared CeedScalar s_Bx[Q * P * DIM] @dim(P, Q, DIM);

        // Store weights in shared memory
        for (int i = 0; i < MAX_PQ; ++i; @inner) {
          for (int j = i; j < (P * Q * DIM); j += MAX_PQ) {
            s_Bx[j] = Bx[j];
          }
        }

        for (int localElement = 0; localElement < ELEMENTS_PER_BLOCK; ++localElement) {
          for (int i = 0; i < MAX_PQ; ++i; @inner) {
            const int element = elementOffset + localElement;
            if (element < elementCount) {
              // Element operation
              for (int component = 0; component < BASIS_COMPONENT_COUNT; ++component) {
                if (!TRANSPOSE) {
                  const int q = i;
                  if (q < Q) {
                    CeedScalar v[DIM];
                    for (int dim = 0; dim < DIM; ++dim) {
                      v[dim] = 0;
                    }

                    for (int p = 0; p < P; ++p) {
                      const CeedScalar u = U(p, component, element, 0);
                      for (int dim = 0; dim < DIM; ++dim) {
                        v[dim] += s_Bx(p, q, dim) * u;
                      }
                    }

                    for (int dim = 0; dim < DIM; ++dim) {
                      V(q, element, component, dim) = v[dim];
                    }
                  }
                } else {
                  const int p = i;
                  if (p < P) {
                    CeedScalar v = 0;
                    for (int dim = 0; dim < DIM; ++dim) {
                      for (int q = 0; q < Q; ++q) {
                        v += s_Bx(p, q, dim) * U(q, element, component, dim);
                      }
                    }
                    V(p, component, element, 0) = v;
                  }
                }
              }
            }
          }
        }
      }
    }

    @kernel void weight(const CeedInt elementCount, const CeedScalar *qWeights, elementWeightArray W) {
      for (int elementOffset = 0; elementOffset < elementCount; elementOffset += ELEMENTS_PER_BLOCK; @outer) {
        @shared CeedScalar s_qWeights[Q];

        for (int q = 0; q < Q; ++q; @inner) {
          s_qWeights[q] = qWeights[q];
        }

        for (int localElement = 0; localElement < ELEMENTS_PER_BLOCK; ++localElement) {
          const int element = elementOffset + localElement;
          if (element < elementCount) {
            for (int q = 0; q < Q; ++q; @inner) {
              W(q, element) = s_qWeights[q];
            }
          }
        }
      }
    }

);
