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

const char *occa_tensor_basis_1d_cpu_function_source = STRINGIFY_SOURCE(

@directive("#define TENSOR_FUNCTION(FUNCTION_NAME) tensor_1d_ ## FUNCTION_NAME ## _Q ## Q1D ## _P ## P1D")

inline void TENSOR_FUNCTION(interpElement)(
  const CeedScalar *B @dim(P1D, Q1D),
  const CeedScalar *Ue,
  CeedScalar *Ve
) {
  for (int q = 0; q < Q1D; ++q) {
    CeedScalar Vq = 0;
    for (int p = 0; p < P1D; ++p) {
      Vq += B(p, q) * Ue[p];
    }
    Ve[q] = Vq;
  }
}

inline void TENSOR_FUNCTION(interpElementTranspose)(
  const CeedScalar *B @dim(P1D, Q1D),
  const CeedScalar *Ue,
  CeedScalar *Ve
) {
  for (int p = 0; p < P1D; ++p) {
    CeedScalar Vp = 0;
    for (int q = 0; q < Q1D; ++q) {
      Vp += B(p, q) * Ue[q];
    }
    Ve[p] = Vp;
  }
}

inline void TENSOR_FUNCTION(gradElement)(
  const CeedScalar *B  @dim(P1D, Q1D),
  const CeedScalar *Bx @dim(P1D, Q1D),
  const CeedScalar *Ue,
  CeedScalar *Ve
) {
  for (int q = 0; q < Q1D; ++q) {
    CeedScalar Vq = 0;
    for (int p = 0; p < P1D; ++p) {
      Vq += Bx(p, q) * Ue[p];
    }
    Ve[q] = Vq;
  }
}

inline void TENSOR_FUNCTION(gradElementTranspose)(
  const CeedScalar *B @dim(P1D, Q1D),
  const CeedScalar *Bx @dim(P1D, Q1D),
  const CeedScalar *Ue,
  CeedScalar *Ve
) {
  for (int p = 0; p < P1D; ++p) {
    CeedScalar Vp = 0;
    for (int q = 0; q < Q1D; ++q) {
      Vp += Bx(p, q) * Ue[q];
    }
    Ve[p] = Vp;
  }
}

inline void TENSOR_FUNCTION(weightElement)(
  const CeedScalar *qWeights1D,
  CeedScalar *We
) {
  for (int q = 0; q < Q1D; ++q) {
    We[q] = qWeights1D[q];
  }
}

);

const char *occa_tensor_basis_1d_cpu_kernel_source = STRINGIFY_SOURCE(

@kernel void interp(const CeedInt elementCount,
                    const CeedScalar *B,
                    const CeedScalar *U,
                    CeedScalar *V) {
  for (int element = 0; element < elementCount; ++element; @outer) {
    for (int component = 0; component < BASIS_COMPONENT_COUNT; ++component; @inner) {
      if (!TRANSPOSE) {
        const CeedScalar *Ue @dim(P1D, BASIS_COMPONENT_COUNT, elementCount) = U;
        CeedScalar *Ve @dim(Q1D, elementCount, BASIS_COMPONENT_COUNT) = V;

        TENSOR_FUNCTION(interpElement)(
          B,
          &Ue(0, component, element),
          &Ve(0, element, component)
        );
      } else {
        const CeedScalar *Ue @dim(Q1D, elementCount, BASIS_COMPONENT_COUNT) = U;
        CeedScalar *Ve @dim(P1D, BASIS_COMPONENT_COUNT, elementCount) = V;

        TENSOR_FUNCTION(interpElementTranspose)(
          B,
          &Ue(0, element, component),
          &Ve(0, component, element)
        );
      }
    }
  }
}

@kernel void grad(const CeedInt elementCount,
                  const CeedScalar *B,
                  const CeedScalar *Bx,
                  const CeedScalar *U,
                  CeedScalar *V) {
  for (int element = 0; element < elementCount; ++element; @outer) {
    for (int component = 0; component < BASIS_COMPONENT_COUNT; ++component; @inner) {
      if (!TRANSPOSE) {
        const CeedScalar *Ue @dim(P1D, BASIS_COMPONENT_COUNT, elementCount) = U;
        CeedScalar *Ve @dim(Q1D, elementCount, BASIS_COMPONENT_COUNT) = V;

        TENSOR_FUNCTION(gradElement)(
          B,
          Bx,
          &Ue(0, component, element),
          &Ve(0, element, component)
        );
      } else {
        const CeedScalar *Ue @dim(Q1D, elementCount, BASIS_COMPONENT_COUNT) = U;
        CeedScalar *Ve @dim(P1D, BASIS_COMPONENT_COUNT, elementCount) = V;

        TENSOR_FUNCTION(gradElementTranspose)(
          B,
          Bx,
          &Ue(0, element, component),
          &Ve(0, component, element)
        );
      }
    }
  }
}

@kernel void weight(const CeedInt elementCount,
                    const CeedScalar *qWeights1D,
                    CeedScalar *W @dim(Q1D, elementCount)) {
  @tile(32, @outer, @inner)
  for (int element = 0; element < elementCount; ++element) {
    TENSOR_FUNCTION(weightElement)(
      qWeights1D,
      &W(0, element)
    );
  }
}

);
