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

#include "./kernel-defines.hpp"

// Kernels are based on the cuda backend from LLNL and VT groups
//
// Expects the following types to be defined:
// - CeedInt
// - CeedScalar
//
// Expects the following constants to be defined:
// - COMPONENT_COUNT  : CeedInt
// - ELEMENT_SIZE     : CeedInt
// - NODE_COUNT       : CeedInt
// - TILE_SIZE        : int
// - USES_INDICES     : bool
// - STRIDE_TYPE      : ceed::occa::StrideType
// - NODE_STRIDE      : Optional[CeedInt]
// - COMPONENT_STRIDE : Optional[CeedInt]
// - ELEMENT_STRIDE   : CeedInt

const char *occa_elem_restriction_source = STRINGIFY_SOURCE(

typedef CeedScalar *QuadVector @dim(ELEMENT_SIZE, COMPONENT_COUNT, elementCount);

@kernel
void applyRestriction(const CeedInt elementCount,
                      @restrict const CeedInt *indices,
                      @restrict const CeedScalar *u,
                      @restrict QuadVector v) {
  @tile(TILE_SIZE, @outer, @inner)
  for (int element = 0; element < elementCount; ++element) {
    for (int node = 0; node < ELEMENT_SIZE; ++node) {

@directive("#if USES_INDICES")
      const CeedInt index = indices[node + (element * ELEMENT_SIZE)];
@directive("#else")
      const CeedInt index = node + (element * ELEMENT_SIZE);
@directive("#endif")

      for (int c = 0; c < COMPONENT_COUNT; ++c) {
@directive("#if STRIDE_TYPE == COMPONENT_STRIDED")
        const CeedScalar u_i = u[index + (c * COMPONENT_STRIDE)];
@directive("#else")
        const CeedInt u_node = index % ELEMENT_SIZE;
        const CeedInt u_element = index / ELEMENT_SIZE;

        const CeedScalar u_i = u[
          (u_node * NODE_STRIDE)
          + (c * COMPONENT_STRIDE)
          + (u_element * ELEMENT_STRIDE)
        ];
@directive("#endif")

        v(node, c, element) = u_i;
      }
    }
  }
}

@kernel
void applyRestrictionTranspose(const CeedInt elementCount,
                               @restrict const CeedInt *quadIndices,
                               @restrict const CeedInt *dofOffsets,
                               @restrict const CeedInt *dofIndices,
                               @restrict const QuadVector u,
                               @restrict CeedScalar *v) {

  @directive("#if USES_INDICES")

  @tile(TILE_SIZE, @outer, @inner)
  for (int n = 0; n < NODE_COUNT; ++n) {
    CeedScalar vComp[COMPONENT_COUNT];

    const CeedInt vIndex = quadIndices[n];
    const CeedInt offsetStart = dofOffsets[n];
    const CeedInt offsetEnd = dofOffsets[n + 1];

    for (int c = 0; c < COMPONENT_COUNT; ++c) {
      vComp[c] = 0;
    }

    for (CeedInt i = offsetStart; i < offsetEnd; ++i) {
      const CeedInt index = dofIndices[i];

      const int node = (index % ELEMENT_SIZE);
      const int element = (index / ELEMENT_SIZE);

      for (int c = 0; c < COMPONENT_COUNT; ++c) {
        vComp[c] += u(node, c, element);
      }
    }

    for (int c = 0; c < COMPONENT_COUNT; ++c) {
      v[vIndex + (c * COMPONENT_STRIDE)] += vComp[c];
    }
  }

@directive("#else")

  @tile(TILE_SIZE, @outer, @inner)
  for (int element = 0; element < elementCount; ++element) {
    for (int node = 0; node < ELEMENT_SIZE; ++node) {
      for (int c = 0; c < COMPONENT_COUNT; ++c) {
        v[
          (node * NODE_STRIDE)
          + (c * COMPONENT_STRIDE)
          + (element * ELEMENT_STRIDE)
        ] += u(node, c, element);
      }
    }
  }

@directive("#endif")

}

);
