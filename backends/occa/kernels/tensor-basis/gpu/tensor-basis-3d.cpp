// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../../kernel-defines.hpp"

const char *occa_tensor_basis_3d_gpu_source = STRINGIFY_SOURCE(

    typedef CeedScalar * dofArray @dim(P1D, P1D, P1D, BASIS_COMPONENT_COUNT, elementCount);
    typedef const CeedScalar *const_dofArray @dim(P1D, P1D, P1D, BASIS_COMPONENT_COUNT, elementCount);

    typedef CeedScalar * quadArray @dim(Q1D, Q1D, Q1D, elementCount, BASIS_COMPONENT_COUNT, 3);
    typedef const CeedScalar *const_quadArray @dim(Q1D, Q1D, Q1D, elementCount, BASIS_COMPONENT_COUNT, 3);

    typedef CeedScalar * sharedBufferArray @dim(MAX_PQ, MAX_PQ, BASIS_COMPONENT_COUNT); typedef const CeedScalar *quadToDof @dim(P1D, Q1D);
    typedef CeedScalar * elementWeightArray @dim(Q1D, Q1D, Q1D, elementCount);

    //---[ Utility Methods ]----------------
    inline void add(const CeedScalar *U, CeedScalar *V) {
      for (int q = 0; q < Q1D; q++) {
        V[q] += U[q];
      }
    }

    inline void readDofs(const int element, const int component, const int px, const int py, const_dofArray U, CeedScalar *Up) {
      // Zero out extra entries
      for (int pz = 0; pz < P1D; ++pz) {
        Up[pz] = ((px < P1D) && (py < P1D) ? U(px, py, pz, component, element) : 0.0);
      }
      for (int q = P1D; q < Q1D; ++q) {
        Up[q] = 0.0;
      }
    }

    inline void writeDofs(const int element, const int component, const int px, const int py, const CeedScalar *Vp, dofArray V) {
      if ((px < P1D) && (py < P1D)) {
        for (int pz = 0; pz < P1D; ++pz) {
          V(px, py, pz, component, element) = Vp[pz];
        }
      }
    }

    inline void readQuads(const int elementCount, const int element, const int component, const int qx, const int qy, const int dim,
                          const_quadArray U, CeedScalar *Uq) {
      for (int qz = 0; qz < Q1D; ++qz) {
        Uq[qz] = U(qx, qy, qz, element, component, dim);
      }
    }

    inline void writeQuads(const int elementCount, const int element, const int component, const int qx, const int qy, const int dim,
                           const CeedScalar *Vq, quadArray V) {
      for (int qz = 0; qz < Q1D; ++qz) {
        V(qx, qy, qz, element, component, dim) = Vq[qz];
      }
    }

    inline void contractX(const int qx, const int qy, const int component, sharedBufferArray sharedBuffer, quadToDof B, const CeedScalar *Uq,
                          CeedScalar *Vp) {
      for (int pz = 0; pz < P1D; ++pz) {
        sharedBuffer(qx, qy, component) = Uq[pz];
        Vp[pz]                          = 0.0;
        @barrier();
        for (int p = 0; p < P1D; ++p) {
          Vp[pz] += B(p, qx) * sharedBuffer(p, qy, component);
        }
        @barrier();
      }
    }

    inline void contractY(const int qx, const int qy, const int component, sharedBufferArray sharedBuffer, quadToDof B, const CeedScalar *Uq,
                          CeedScalar *Vp) {
      for (int pz = 0; pz < P1D; ++pz) {
        sharedBuffer(qx, qy, component) = Uq[pz];
        Vp[pz]                          = 0.0;
        @barrier();
        for (int p = 0; p < P1D; ++p) {
          Vp[pz] += B(p, qy) * sharedBuffer(qx, p, component);
        }
        @barrier();
      }
    }

    inline void contractZ(const int qx, const int qy, quadToDof B, const CeedScalar *Up, CeedScalar *Vq) {
      for (int qz = 0; qz < Q1D; ++qz) {
        Vq[qz] = 0.0;
        for (int p = 0; p < P1D; ++p) {
          Vq[qz] += B(p, qz) * Up[p];
        }
      }
    }

    inline void contractTransposeX(const int px, const int py, const int component, sharedBufferArray sharedBuffer, quadToDof B, const CeedScalar *Up,
                                   CeedScalar *Vp) {
      for (int pz = 0; pz < P1D; ++pz) {
        sharedBuffer(px, py, component) = Up[pz];
        Vp[pz]                          = 0.0;
        @barrier();
        if (px < P1D) {
          for (int qx = 0; qx < Q1D; ++qx) {
            Vp[pz] += B(px, qx) * sharedBuffer(qx, py, component);
          }
        }
        @barrier();
      }
    }

    inline void contractTransposeY(const int px, const int py, const int component, sharedBufferArray sharedBuffer, quadToDof B, const CeedScalar *Up,
                                   CeedScalar *Vp) {
      for (int pz = 0; pz < P1D; ++pz) {
        sharedBuffer(px, py, component) = Up[pz];
        Vp[pz]                          = 0.0;
        @barrier();
        if (py < P1D) {
          for (int qy = 0; qy < Q1D; ++qy) {
            Vp[pz] += B(py, qy) * sharedBuffer(px, qy, component);
          }
        }
        @barrier();
      }
    }

    inline void contractTransposeZ(const int px, const int py, quadToDof B, const CeedScalar *Uq, CeedScalar *Vq) {
      for (int pz = 0; pz < P1D; ++pz) {
        Vq[pz] = 0.0;
        for (int qz = 0; qz < Q1D; ++qz) {
          Vq[pz] += B(pz, qz) * Uq[qz];
        }
      }
    }

    //---[ Kernels ]------------------------
    @kernel void interp(const CeedInt elementCount, quadToDof B, const CeedScalar *U, CeedScalar *V) {
      for (int element = 0; element < elementCount; ++element; @outer) {
        @shared CeedScalar sharedBuffer[MAX_PQ * MAX_PQ * BASIS_COMPONENT_COUNT];

        for (int component = 0; component < BASIS_COMPONENT_COUNT; ++component; @inner) {
          for (int qy = 0; qy < Q1D; ++qy; @inner) {
            for (int qx = 0; qx < Q1D; ++qx; @inner) {
              if (element < elementCount) {
                CeedScalar r1[MAX_PQ], r2[MAX_PQ];
                for (int q = 0; q < Q1D; ++q) {
                  r1[q] = 0.0;
                  r2[q] = 0.0;
                }

                if (!TRANSPOSE) {
                  readDofs(element, component, qx, qy, U, r1);
                  contractX(qx, qy, component, sharedBuffer, B, r1, r2);
                  contractY(qx, qy, component, sharedBuffer, B, r2, r1);
                  contractZ(qx, qy, B, r1, r2);
                  writeQuads(elementCount, element, component, qx, qy, 0, r2, V);
                } else {
                  readQuads(elementCount, element, component, qx, qy, 0, U, r1);
                  contractTransposeZ(qx, qy, B, r1, r2);
                  contractTransposeY(qx, qy, component, sharedBuffer, B, r2, r1);
                  contractTransposeX(qx, qy, component, sharedBuffer, B, r1, r2);
                  writeDofs(element, component, qx, qy, r2, V);
                }
              }
            }
          }
        }
      }
    }

    @kernel void grad(const CeedInt elementCount, quadToDof B, quadToDof Bx, const CeedScalar *U, CeedScalar *V) {
      for (int element = 0; element < elementCount; ++element; @outer) {
        @shared CeedScalar sharedBuffer[MAX_PQ * MAX_PQ * BASIS_COMPONENT_COUNT];

        for (int component = 0; component < BASIS_COMPONENT_COUNT; ++component; @inner) {
          for (int qy = 0; qy < Q1D; ++qy; @inner) {
            for (int qx = 0; qx < Q1D; ++qx; @inner) {
              if (element < elementCount) {
                CeedScalar r1[MAX_PQ], r2[MAX_PQ], r3[MAX_PQ];

                if (!TRANSPOSE) {
                  readDofs(element, component, qx, qy, U, r1);
                  // Dx
                  contractX(qx, qy, component, sharedBuffer, Bx, r1, r2);
                  contractY(qx, qy, component, sharedBuffer, B, r2, r3);
                  contractZ(qx, qy, B, r3, r2);
                  writeQuads(elementCount, element, component, qx, qy, 0, r2, V);
                  // Dy
                  contractX(qx, qy, component, sharedBuffer, B, r1, r2);
                  contractY(qx, qy, component, sharedBuffer, Bx, r2, r3);
                  contractZ(qx, qy, B, r3, r2);
                  writeQuads(elementCount, element, component, qx, qy, 1, r2, V);
                  // Dz
                  contractX(qx, qy, component, sharedBuffer, B, r1, r2);
                  contractY(qx, qy, component, sharedBuffer, B, r2, r3);
                  contractZ(qx, qy, Bx, r3, r2);
                  writeQuads(elementCount, element, component, qx, qy, 2, r2, V);
                } else {
                  // Dx
                  readQuads(elementCount, element, component, qx, qy, 0, U, r1);
                  contractTransposeZ(qx, qy, B, r1, r3);
                  contractTransposeY(qx, qy, component, sharedBuffer, B, r3, r1);
                  contractTransposeX(qx, qy, component, sharedBuffer, Bx, r1, r2);
                  // Dy
                  readQuads(elementCount, element, component, qx, qy, 1, U, r1);
                  contractTransposeZ(qx, qy, B, r1, r3);
                  contractTransposeY(qx, qy, component, sharedBuffer, Bx, r3, r1);
                  contractTransposeX(qx, qy, component, sharedBuffer, B, r1, r3);
                  add(r3, r2);
                  // Dz
                  readQuads(elementCount, element, component, qx, qy, 2, U, r1);
                  contractTransposeZ(qx, qy, Bx, r1, r3);
                  contractTransposeY(qx, qy, component, sharedBuffer, B, r3, r1);
                  contractTransposeX(qx, qy, component, sharedBuffer, B, r1, r3);
                  add(r3, r2);
                  writeDofs(element, component, qx, qy, r2, V);
                }
              }
            }
          }
        }
      }
    }

    @kernel void weight(const CeedInt elementCount, const CeedScalar *qWeights1D, elementWeightArray W) {
      for (int element = 0; element < elementCount; ++element; @outer) {
        for (int qz = 0; qz < Q1D; ++qz; @inner) {
          for (int qy = 0; qy < Q1D; ++qy; @inner) {
            for (int qx = 0; qx < Q1D; ++qx) {
              if (element < elementCount) {
                W(qx, qy, qz, element) = qWeights1D[qx] * qWeights1D[qy] * qWeights1D[qz];
              }
            }
          }
        }
      }
    }

);
