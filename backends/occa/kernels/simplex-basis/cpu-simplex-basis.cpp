// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "../kernel-defines.hpp"

const char *occa_simplex_basis_cpu_function_source = STRINGIFY_SOURCE(

    @directive("#define SIMPLEX_FUNCTION(FUNCTION_NAME) simplex_ ## DIM ## d_ ## FUNCTION_NAME ## _Q ## Q ## _P ## P")

        inline void SIMPLEX_FUNCTION(interpElement)(const CeedScalar *B @dim(P, Q), const CeedScalar *Ue, CeedScalar *Ve) {
          for (int q = 0; q < Q; ++q) {
            CeedScalar v = 0;
            for (int p = 0; p < P; ++p) {
              v += B(p, q) * Ue[p];
            }
            Ve[q] = v;
          }
        }

    inline void SIMPLEX_FUNCTION(interpElementTranspose)(const CeedScalar *B @dim(P, Q), const CeedScalar *Ue, CeedScalar *Ve) {
      for (int p = 0; p < P; ++p) {
        CeedScalar v = 0;
        for (int q = 0; q < Q; ++q) {
          v += B(p, q) * Ue[q];
        }
        Ve[p] = v;
      }
    }

    inline void SIMPLEX_FUNCTION(gradElement)(const CeedScalar *Bx @dim(P, Q, DIM), const CeedScalar *Ue, CeedScalar *Ve, ) {
      for (int q = 0; q < Q; ++q) {
        CeedScalar v[DIM];
        for (int dim = 0; dim < DIM; ++dim) {
          v[dim] = 0;
        }

        for (int p = 0; p < P; ++p) {
          const CeedScalar u = Ue[p];
          for (int dim = 0; dim < DIM; ++dim) {
            v[dim] += Bx(p, q, dim) * u;
          }
        }

        for (int dim = 0; dim < DIM; ++dim) {
          Ve[dim * Q + q] = v[dim];
        }
      }
    }

    inline void SIMPLEX_FUNCTION(gradElementTranspose)(const CeedScalar *Bx @dim(P, Q, DIM), const CeedScalar *Ue, CeedScalar *Ve) {
      for (int p = 0; p < P; ++p) {
        CeedScalar v = 0;
        for (int dim = 0; dim < DIM; ++dim) {
          for (int q = 0; q < Q; ++q) {
            v += Bx(p, q, dim) * Ue[dim * Q + q];
          }
        }
        Ve[p] = v;
      }
    }

    inline void SIMPLEX_FUNCTION(weightElement)(const CeedScalar *qWeights, CeedScalar *We) {
      for (int q = 0; q < Q; ++q) {
        We[q] = qWeights[q];
      }
    }

);

const char *occa_simplex_basis_cpu_kernel_source = STRINGIFY_SOURCE(

    @kernel void interp(const CeedInt elementCount, const CeedScalar *B, const CeedScalar *U, CeedScalar *V) {
      for (int element = 0; element < elementCount; ++element; @outer) {
        for (int component = 0; component < BASIS_COMPONENT_COUNT; ++component; @inner) {
          if (!TRANSPOSE) {
            const CeedScalar *Ue @dim(P, BASIS_COMPONENT_COUNT, elementCount) = U;
            CeedScalar       *Ve @dim(Q, elementCount, BASIS_COMPONENT_COUNT) = V;

            SIMPLEX_FUNCTION(interpElement)(B, &Ue(0, component, element), &Ve(0, element, component));
          } else {
            const CeedScalar *Ue @dim(Q, elementCount, BASIS_COMPONENT_COUNT) = U;
            CeedScalar       *Ve @dim(P, BASIS_COMPONENT_COUNT, elementCount) = V;

            SIMPLEX_FUNCTION(interpElementTranspose)(B, &Ue(0, element, component), &Ve(0, component, element));
          }
        }
      }
    }

    @kernel void grad(const CeedInt elementCount, const CeedScalar *Bx, const CeedScalar *U, CeedScalar *V) {
      for (int element = 0; element < elementCount; ++element; @outer) {
        for (int component = 0; component < BASIS_COMPONENT_COUNT; ++component; @inner) {
          if (!TRANSPOSE) {
            const CeedScalar *Ue @dim(P, BASIS_COMPONENT_COUNT, elementCount)       = U;
            CeedScalar       *_Ve @dim(Q, elementCount, BASIS_COMPONENT_COUNT, DIM) = V;

            CeedScalar Ve[DIM][Q];
            for (int dim = 0; dim < DIM; ++dim) {
              for (int q = 0; q < Q; ++q) {
                Ve[dim][q] = _Ve(q, element, component, dim);
              }
            }

            SIMPLEX_FUNCTION(gradElement)(Bx, &Ue(0, component, element), (CeedScalar *)Ve);

            for (int dim = 0; dim < DIM; ++dim) {
              for (int q = 0; q < Q; ++q) {
                _Ve(q, element, component, dim) = Ve[dim][q];
              }
            }
          } else {
            const CeedScalar *_Ue @dim(Q, elementCount, BASIS_COMPONENT_COUNT, DIM) = U;
            CeedScalar       *Ve @dim(P, BASIS_COMPONENT_COUNT, elementCount)       = V;

            CeedScalar Ue[DIM][Q];
            for (int dim = 0; dim < DIM; ++dim) {
              for (int q = 0; q < Q; ++q) {
                Ue[dim][q] = _Ue(q, element, component, dim);
              }
            }

            SIMPLEX_FUNCTION(gradElementTranspose)(Bx, (CeedScalar *)Ue, &Ve(0, component, element));
          }
        }
      }
    }

    @kernel void weight(const CeedInt elementCount, const CeedScalar *qWeights, CeedScalar *W @dim(Q, elementCount)) {
      @tile(32, @outer, @inner) for (int element = 0; element < elementCount; ++element) {
        SIMPLEX_FUNCTION(weightElement)(qWeights, &W(0, element));
      }
    }

);
