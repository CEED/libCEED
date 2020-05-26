// Copyright (c) 2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include "../../kernel-defines.hpp"


const char *occa_tensor_basis_2d_cpu_function_source = STRINGIFY_SOURCE(

@directive("#define TENSOR_FUNCTION(FUNCTION_NAME) tensor_2d_ ## FUNCTION_NAME ## _Q ## Q1D ## _P ## P1D")

inline void TENSOR_FUNCTION(interpElement)(
  @restrict const CeedScalar *B @dim(P1D, Q1D),
  @restrict const CeedScalar *Ue @dim(P1D, P1D),
  @restrict CeedScalar *Ve @dim(Q1D, Q1D)
) {
  for (int qy = 0; qy < Q1D; ++qy) {
    for (int qx = 0; qx < Q1D; ++qx) {
      Ve(qx, qy) = 0;
    }
  }

  for (int py = 0; py < P1D; ++py) {
    CeedScalar V_x[Q1D];
    for (int qx = 0; qx < Q1D; ++qx) {
      V_x[qx] = 0;
    }

    for (int px = 0; px < P1D; ++px) {
      const CeedScalar Up = Ue(px, py);
      for (int qx = 0; qx < Q1D; ++qx) {
        V_x[qx] += B(px, qx) * Up;
      }
    }

    for (int qy = 0; qy < Q1D; ++qy) {
      const CeedScalar w = B(py, qy);
      for (int qx = 0; qx < Q1D; ++qx) {
        Ve(qx, qy) += w * V_x[qx];
      }
    }
  }
}

inline void TENSOR_FUNCTION(interpElementTranspose)(
  @restrict const CeedScalar *B @dim(P1D, Q1D),
  @restrict const CeedScalar *Ue @dim(Q1D, Q1D),
  @restrict CeedScalar *Ve @dim(P1D, P1D)
) {
  for (int py = 0; py < P1D; ++py) {
    for (int px = 0; px < P1D; ++px) {
      Ve(px, py) = 0;
    }
  }

  for (int qy = 0; qy < Q1D; ++qy) {
    CeedScalar V_x[P1D];
    for (int py = 0; py < P1D; ++py) {
      V_x[py] = 0;
    }

    for (int qx = 0; qx < Q1D; ++qx) {
      const CeedScalar Up = Ue(qx, qy);
      for (int px = 0; px < P1D; ++px) {
        V_x[px] += B(px, qx) * Up;
      }
    }

    for (int py = 0; py < P1D; ++py) {
      const CeedScalar w = B(py, qy);
      for (int px = 0; px < P1D; ++px) {
        Ve(px, py) += w * V_x[px];
      }
    }
  }
}

inline void TENSOR_FUNCTION(gradElement)(
  @restrict const CeedScalar *B @dim(P1D, Q1D),
  @restrict const CeedScalar *Bx @dim(P1D, Q1D),
  @restrict const CeedScalar *Ue @dim(P1D, P1D),
  @restrict CeedScalar *Ve_x @dim(Q1D, Q1D),
  @restrict CeedScalar *Ve_y @dim(Q1D, Q1D)
) {
  CeedScalar grad[Q1D][Q1D][2];
  for (int qy = 0; qy < Q1D; ++qy) {
    for (int qx = 0; qx < Q1D; ++qx) {
      grad[qy][qx][0] = 0;
      grad[qy][qx][1] = 0;
    }
  }

  for (int py = 0; py < P1D; ++py) {
    CeedScalar gradX[Q1D][2];
    for (int qx = 0; qx < Q1D; ++qx) {
      gradX[qx][0] = 0;
      gradX[qx][1] = 0;
    }

    for (int px = 0; px < P1D; ++px) {
      const CeedScalar Up = Ue(px, py);
      for (int qx = 0; qx < Q1D; ++qx) {
        gradX[qx][0] += Up * B(px, qx);
        gradX[qx][1] += Up * Bx(px, qx);
      }
    }

    for (int qy = 0; qy < Q1D; ++qy) {
      const CeedScalar wx  = B(py, qy);
      const CeedScalar wDx = Bx(py, qy);
      for (int qx = 0; qx < Q1D; ++qx) {
        grad[qy][qx][0] += gradX[qx][1] * wx;
        grad[qy][qx][1] += gradX[qx][0] * wDx;
      }
    }
  }
  for (int qy = 0; qy < Q1D; ++qy) {
    for (int qx = 0; qx < Q1D; ++qx) {
      Ve_x(qx, qy) = grad[qy][qx][0];
      Ve_y(qx, qy) = grad[qy][qx][1];
    }
  }
}

inline void TENSOR_FUNCTION(gradElementTranspose)(
  @restrict const CeedScalar *B @dim(P1D, Q1D),
  @restrict const CeedScalar *Bx @dim(P1D, Q1D),
  @restrict const CeedScalar *Ue_x @dim(Q1D, Q1D),
  @restrict const CeedScalar *Ue_y @dim(Q1D, Q1D),
  @restrict CeedScalar *Ve @dim(P1D, P1D)
) {
  for (int py = 0; py < P1D; ++py) {
    for (int px = 0; px < P1D; ++px) {
      Ve(px, py) = 0.0;
    }
  }

  for (int qy = 0; qy < Q1D; ++qy) {
    CeedScalar gradX[P1D][2];
    for (int px = 0; px < P1D; ++px) {
      gradX[px][0] = 0;
      gradX[px][1] = 0;
    }

    for (int qx = 0; qx < Q1D; ++qx) {
      const CeedScalar Ux = Ue_x(qy, qx);
      const CeedScalar Uy = Ue_y(qy, qx);
      for (int px = 0; px < P1D; ++px) {
        const CeedScalar wx  = B(px, qx);
        const CeedScalar wDx = Bx(px, qx);
        gradX[px][0] += Ux * wx;
        gradX[px][1] += Uy * wDx;
      }
    }

    for (int py = 0; py < P1D; ++py) {
      const CeedScalar wy  = B(py, qy);
      const CeedScalar wDy = Bx(py, qy);
      for (int px = 0; px < P1D; ++px) {
        Ve(px, py) += ((gradX[px][1] * wy) +
                       (gradX[px][0] * wDy));
      }
    }
  }
}

inline void TENSOR_FUNCTION(weightElement)(
  @restrict const CeedScalar *qWeights1D,
  @restrict CeedScalar *We @dim(Q1D, Q1D)
) {
  for (int qy = 0; qy < Q1D; ++qy) {
    const CeedScalar wy = qWeights1D[qy];
    for (int qx = 0; qx < Q1D; ++qx) {
      We(qx, qy) = qWeights1D[qx] * wy;
    }
  }
}

);

const char *occa_tensor_basis_2d_cpu_kernel_source = STRINGIFY_SOURCE(

@kernel void interp(const CeedInt elementCount,
                    @restrict const CeedScalar *B,
                    @restrict const CeedScalar *U,
                    @restrict CeedScalar *V) {
  for (int element = 0; element < elementCount; ++element; @outer) {
    for (int component = 0; component < BASIS_COMPONENT_COUNT; ++component; @inner) {
      if (!TRANSPOSE) {
        const CeedScalar *Ue @dim(P1D, P1D, BASIS_COMPONENT_COUNT, elementCount) = U;
        CeedScalar *Ve @dim(Q1D, Q1D, elementCount, BASIS_COMPONENT_COUNT) = V;

        TENSOR_FUNCTION(interpElement)(
          B,
          &Ue(0, 0, component, element),
          &Ve(0, 0, element, component)
        );
      } else {
        const CeedScalar *Ue @dim(Q1D, Q1D, elementCount, BASIS_COMPONENT_COUNT) = U;
        CeedScalar *Ve @dim(P1D, P1D, BASIS_COMPONENT_COUNT, elementCount) = V;

        TENSOR_FUNCTION(interpElementTranspose)(
          B,
          &Ue(0, 0, element, component),
          &Ve(0, 0, component, element)
        );
      }
    }
  }
}

@kernel void grad(const CeedInt elementCount,
                  @restrict const CeedScalar *B,
                  @restrict const CeedScalar *Bx,
                  @restrict const CeedScalar *U,
                  @restrict CeedScalar *V) {
  for (int element = 0; element < elementCount; ++element; @outer) {
    for (int component = 0; component < BASIS_COMPONENT_COUNT; ++component; @inner) {
      if (!TRANSPOSE) {
        const CeedScalar *Ue @dim(P1D, P1D, BASIS_COMPONENT_COUNT, elementCount) = U;
        CeedScalar *Ve @dim(Q1D, Q1D, elementCount, BASIS_COMPONENT_COUNT, 2) = V;

        TENSOR_FUNCTION(gradElement)(
          B,
          Bx,
          &Ue(0, 0, component, element),
          &Ve(0, 0, element, component, 0),
          &Ve(0, 0, element, component, 1)
        );
      } else {
        const CeedScalar *Ue @dim(Q1D, Q1D, elementCount, BASIS_COMPONENT_COUNT, 2) = U;
        CeedScalar *Ve @dim(P1D, P1D, BASIS_COMPONENT_COUNT, elementCount) = V;

        TENSOR_FUNCTION(gradElementTranspose)(
          B,
          Bx,
          &Ue(0, 0, element, component, 0),
          &Ue(0, 0, element, component, 1),
          &Ve(0, 0, component, element)
        );
      }
    }
  }
}

@kernel void weight(const CeedInt elementCount,
                    @restrict const CeedScalar *qWeights1D,
                    @restrict CeedScalar *W @dim(Q1D, Q1D, elementCount)) {
  @tile(32, @outer, @inner)
  for (int element = 0; element < elementCount; ++element) {
    TENSOR_FUNCTION(weightElement)(
      qWeights1D,
      &W(0, 0, element)
    );
  }
}

);
