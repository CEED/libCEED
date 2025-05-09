// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../../kernel-defines.hpp"

const char *occa_tensor_basis_3d_cpu_function_source = STRINGIFY_SOURCE(

    @directive("#define TENSOR_FUNCTION(FUNCTION_NAME) tensor_3d_ ## FUNCTION_NAME ## _Q ## Q1D ## _P ## P1D")

        inline void TENSOR_FUNCTION(interpElement)(const CeedScalar *B @dim(P1D, Q1D), const CeedScalar *Ue @dim(P1D, P1D, P1D),
                                                   CeedScalar *Ve @dim(Q1D, Q1D, Q1D)) {
          for (int qz = 0; qz < Q1D; ++qz) {
            for (int qy = 0; qy < Q1D; ++qy) {
              for (int qx = 0; qx < Q1D; ++qx) {
                Ve(qx, qy, qz) = 0;
              }
            }
          }

          for (int pz = 0; pz < P1D; ++pz) {
            CeedScalar V_xy[Q1D][Q1D];
            for (int qy = 0; qy < Q1D; ++qy) {
              for (int qx = 0; qx < Q1D; ++qx) {
                V_xy[qy][qx] = 0;
              }
            }

            for (int py = 0; py < P1D; ++py) {
              CeedScalar V_x[Q1D];
              for (int qx = 0; qx < Q1D; ++qx) {
                V_x[qx] = 0;
              }

              for (int px = 0; px < P1D; ++px) {
                const CeedScalar Up = Ue(px, py, pz);
                for (int qx = 0; qx < Q1D; ++qx) {
                  V_x[qx] += B(px, qx) * Up;
                }
              }

              for (int qy = 0; qy < Q1D; ++qy) {
                const CeedScalar wy = B(py, qy);
                for (int qx = 0; qx < Q1D; ++qx) {
                  V_xy[qy][qx] += wy * V_x[qx];
                }
              }
            }

            for (int qz = 0; qz < Q1D; ++qz) {
              const CeedScalar wz = B(pz, qz);
              for (int qy = 0; qy < Q1D; ++qy) {
                for (int qx = 0; qx < Q1D; ++qx) {
                  Ve(qx, qy, qz) += wz * V_xy[qy][qx];
                }
              }
            }
          }
        }

    inline void TENSOR_FUNCTION(interpElementTranspose)(const CeedScalar *B @dim(P1D, Q1D), const CeedScalar *Ue @dim(Q1D, Q1D, Q1D),
                                                        CeedScalar *Ve @dim(P1D, P1D, P1D)) {
      for (int pz = 0; pz < P1D; ++pz) {
        for (int py = 0; py < P1D; ++py) {
          for (int px = 0; px < P1D; ++px) {
            Ve(px, py, pz) = 0;
          }
        }
      }

      for (int qz = 0; qz < Q1D; ++qz) {
        CeedScalar V_xy[P1D][P1D];
        for (int py = 0; py < P1D; ++py) {
          for (int px = 0; px < P1D; ++px) {
            V_xy[py][px] = 0;
          }
        }

        for (int qy = 0; qy < Q1D; ++qy) {
          CeedScalar V_x[P1D];
          for (int px = 0; px < P1D; ++px) {
            V_x[px] = 0;
          }

          for (int qx = 0; qx < Q1D; ++qx) {
            const CeedScalar Uq = Ue(qx, qy, qz);
            for (int px = 0; px < P1D; ++px) {
              V_x[px] += B(px, qx) * Uq;
            }
          }

          for (int py = 0; py < P1D; ++py) {
            const CeedScalar wy = B(py, qy);
            for (int px = 0; px < P1D; ++px) {
              V_xy[py][px] += wy * V_x[px];
            }
          }
        }

        for (int pz = 0; pz < P1D; ++pz) {
          const CeedScalar wz = B(pz, qz);
          for (int py = 0; py < P1D; ++py) {
            for (int px = 0; px < P1D; ++px) {
              Ve(px, py, pz) += wz * V_xy[py][px];
            }
          }
        }
      }
    }

    inline void TENSOR_FUNCTION(gradElement)(const CeedScalar *B @dim(P1D, Q1D), const CeedScalar *Bx @dim(P1D, Q1D),
                                             const CeedScalar *Ue @dim(P1D, P1D, P1D), CeedScalar *Ve_x @dim(Q1D, Q1D, Q1D),
                                             CeedScalar *Ve_y @dim(Q1D, Q1D, Q1D), CeedScalar *Ve_z @dim(Q1D, Q1D, Q1D)) {
      for (int qz = 0; qz < Q1D; ++qz) {
        for (int qy = 0; qy < Q1D; ++qy) {
          for (int qx = 0; qx < Q1D; ++qx) {
            Ve_x(qx, qy, qz) = 0;
            Ve_y(qx, qy, qz) = 0;
            Ve_z(qx, qy, qz) = 0;
          }
        }
      }

      for (int pz = 0; pz < P1D; ++pz) {
        CeedScalar gradXY[Q1D][Q1D][3];
        for (int qy = 0; qy < Q1D; ++qy) {
          for (int qx = 0; qx < Q1D; ++qx) {
            gradXY[qy][qx][0] = 0;
            gradXY[qy][qx][1] = 0;
            gradXY[qy][qx][2] = 0;
          }
        }

        for (int py = 0; py < P1D; ++py) {
          CeedScalar gradX[Q1D][2];
          for (int qx = 0; qx < Q1D; ++qx) {
            gradX[qx][0] = 0;
            gradX[qx][1] = 0;
          }

          for (int px = 0; px < P1D; ++px) {
            const CeedScalar Up = Ue(px, py, pz);
            for (int qx = 0; qx < Q1D; ++qx) {
              gradX[qx][0] += Up * B(px, qx);
              gradX[qx][1] += Up * Bx(px, qx);
            }
          }

          for (int qy = 0; qy < Q1D; ++qy) {
            const CeedScalar wy  = B(py, qy);
            const CeedScalar wDy = Bx(py, qy);
            for (int qx = 0; qx < Q1D; ++qx) {
              const CeedScalar wx  = gradX[qx][0];
              const CeedScalar wDx = gradX[qx][1];
              gradXY[qy][qx][0] += wDx * wy;
              gradXY[qy][qx][1] += wx * wDy;
              gradXY[qy][qx][2] += wx * wy;
            }
          }
        }

        for (int qz = 0; qz < Q1D; ++qz) {
          const CeedScalar wz  = B(pz, qz);
          const CeedScalar wDz = Bx(pz, qz);
          for (int qy = 0; qy < Q1D; ++qy) {
            for (int qx = 0; qx < Q1D; ++qx) {
              Ve_x(qx, qy, qz) += gradXY[qy][qx][0] * wz;
              Ve_y(qx, qy, qz) += gradXY[qy][qx][1] * wz;
              Ve_z(qx, qy, qz) += gradXY[qy][qx][2] * wDz;
            }
          }
        }
      }
    }

    inline void TENSOR_FUNCTION(gradElementTranspose)(const CeedScalar *B @dim(P1D, Q1D), const CeedScalar *Bx @dim(P1D, Q1D),
                                                      const CeedScalar *Ue_x @dim(Q1D, Q1D, Q1D), const CeedScalar *Ue_y @dim(Q1D, Q1D, Q1D),
                                                      const CeedScalar *Ue_z @dim(Q1D, Q1D, Q1D), CeedScalar *Ve @dim(P1D, P1D, P1D)) {
      for (int pz = 0; pz < P1D; ++pz) {
        for (int py = 0; py < P1D; ++py) {
          for (int px = 0; px < P1D; ++px) {
            Ve(px, py, pz) = 0;
          }
        }
      }

      for (int qz = 0; qz < Q1D; ++qz) {
        CeedScalar gradXY[P1D][P1D][3];
        for (int py = 0; py < P1D; ++py) {
          for (int px = 0; px < P1D; ++px) {
            gradXY[py][px][0] = 0;
            gradXY[py][px][1] = 0;
            gradXY[py][px][2] = 0;
          }
        }

        for (int qy = 0; qy < Q1D; ++qy) {
          CeedScalar gradX[P1D][3];
          for (int px = 0; px < P1D; ++px) {
            gradX[px][0] = 0;
            gradX[px][1] = 0;
            gradX[px][2] = 0;
          }

          for (int qx = 0; qx < Q1D; ++qx) {
            const CeedScalar Ux = Ue_x(qx, qy, qz);
            const CeedScalar Uy = Ue_y(qx, qy, qz);
            const CeedScalar Uz = Ue_z(qx, qy, qz);
            for (int px = 0; px < P1D; ++px) {
              const CeedScalar wx  = B(px, qx);
              const CeedScalar wDx = Bx(px, qx);
              gradX[px][0] += Ux * wDx;
              gradX[px][1] += Uy * wx;
              gradX[px][2] += Uz * wx;
            }
          }

          for (int py = 0; py < P1D; ++py) {
            const CeedScalar wy  = B(py, qy);
            const CeedScalar wDy = Bx(py, qy);
            for (int px = 0; px < P1D; ++px) {
              gradXY[py][px][0] += gradX[px][0] * wy;
              gradXY[py][px][1] += gradX[px][1] * wDy;
              gradXY[py][px][2] += gradX[px][2] * wy;
            }
          }
        }

        for (int pz = 0; pz < P1D; ++pz) {
          const CeedScalar wz  = B(pz, qz);
          const CeedScalar wDz = Bx(pz, qz);
          for (int py = 0; py < P1D; ++py) {
            for (int px = 0; px < P1D; ++px) {
              Ve(px, py, pz) += ((gradXY[py][px][0] * wz) + (gradXY[py][px][1] * wz) + (gradXY[py][px][2] * wDz));
            }
          }
        }
      }
    }

    inline void TENSOR_FUNCTION(weightElement)(const CeedScalar *qWeights1D, CeedScalar *We @dim(Q1D, Q1D, Q1D)) {
      for (int qz = 0; qz < Q1D; ++qz) {
        const CeedScalar wz = qWeights1D[qz];
        for (int qy = 0; qy < Q1D; ++qy) {
          const CeedScalar wy = qWeights1D[qy];
          for (int qx = 0; qx < Q1D; ++qx) {
            We(qx, qy, qz) = qWeights1D[qx] * wy * wz;
          }
        }
      }
    }

);

const char *occa_tensor_basis_3d_cpu_kernel_source = STRINGIFY_SOURCE(

    @kernel void interp(const CeedInt elementCount, const CeedScalar *B, const CeedScalar *U, CeedScalar *V) {
      for (int element = 0; element < elementCount; ++element; @outer) {
        for (int component = 0; component < BASIS_COMPONENT_COUNT; ++component; @inner) {
          if (!TRANSPOSE) {
            const CeedScalar *Ue @dim(P1D, P1D, P1D, BASIS_COMPONENT_COUNT, elementCount) = U;
            CeedScalar       *Ve @dim(Q1D, Q1D, Q1D, elementCount, BASIS_COMPONENT_COUNT) = V;

            TENSOR_FUNCTION(interpElement)(B, &Ue(0, 0, 0, component, element), &Ve(0, 0, 0, element, component));
          } else {
            const CeedScalar *Ue @dim(Q1D, Q1D, Q1D, elementCount, BASIS_COMPONENT_COUNT) = U;
            CeedScalar       *Ve @dim(P1D, P1D, P1D, BASIS_COMPONENT_COUNT, elementCount) = V;

            TENSOR_FUNCTION(interpElementTranspose)(B, &Ue(0, 0, 0, element, component), &Ve(0, 0, 0, component, element));
          }
        }
      }
    }

    @kernel void grad(const CeedInt elementCount, const CeedScalar *B, const CeedScalar *Bx, const CeedScalar *U, CeedScalar *V) {
      for (int element = 0; element < elementCount; ++element; @outer) {
        for (int component = 0; component < BASIS_COMPONENT_COUNT; ++component; @inner) {
          if (!TRANSPOSE) {
            const CeedScalar *Ue @dim(P1D, P1D, P1D, BASIS_COMPONENT_COUNT, elementCount)    = U;
            CeedScalar       *Ve @dim(Q1D, Q1D, Q1D, elementCount, BASIS_COMPONENT_COUNT, 3) = V;

            TENSOR_FUNCTION(gradElement)
            (B, Bx, &Ue(0, 0, 0, component, element), &Ve(0, 0, 0, element, component, 0), &Ve(0, 0, 0, element, component, 1),
             &Ve(0, 0, 0, element, component, 2));
          } else {
            const CeedScalar *Ue @dim(Q1D, Q1D, Q1D, elementCount, BASIS_COMPONENT_COUNT, 3) = U;
            CeedScalar       *Ve @dim(P1D, P1D, P1D, BASIS_COMPONENT_COUNT, elementCount)    = V;

            TENSOR_FUNCTION(gradElementTranspose)
            (B, Bx, &Ue(0, 0, 0, element, component, 0), &Ue(0, 0, 0, element, component, 1), &Ue(0, 0, 0, element, component, 2),
             &Ve(0, 0, 0, component, element));
          }
        }
      }
    }

    @kernel void weight(const CeedInt elementCount, const CeedScalar *qWeights1D, CeedScalar *W @dim(Q1D, Q1D, Q1D, elementCount)) {
      @tile(32, @outer, @inner) for (int element = 0; element < elementCount; ++element) {
        TENSOR_FUNCTION(weightElement)(qWeights1D, &W(0, 0, 0, element));
      }
    }

);
