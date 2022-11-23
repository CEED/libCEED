// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "./kernel-defines.hpp"

// Kernels are based on the cuda backend from LLNL and VT groups
//
// Expects the following types to be defined:
// - CeedInt
// - CeedScalar
//
// Expects the following constants to be defined:
// - COMPONENT_COUNT            : CeedInt
// - ELEMENT_SIZE               : CeedInt
// - NODE_COUNT                 : CeedInt
// - TILE_SIZE                  : int
// - USES_INDICES               : bool
// - STRIDE_TYPE                : ceed::occa::StrideType
// - NODE_STRIDE                : Optional[CeedInt]
// - COMPONENT_STRIDE           : Optional[CeedInt]
// - ELEMENT_STRIDE             : Optional[CeedInt]
// - UNSTRIDED_COMPONENT_STRIDE : Optional[CeedInt]

const char *occa_elem_restriction_source = STRINGIFY_SOURCE(

    @directive("#define PRINT_KERNEL_HASHES 0")

            typedef CeedScalar *
        QuadVector @dim(ELEMENT_SIZE, COMPONENT_COUNT, elementCount);

    @kernel void applyRestriction(const CeedInt elementCount, const CeedInt *indices, CeedScalar *u, QuadVector v) {
      @tile(TILE_SIZE, @outer, @inner) for (int element = 0; element < elementCount; ++element) {
        @directive("#if PRINT_KERNEL_HASHES")
            // Print to see which kernel is being run
            if (element == 0) {
          printf("\n\napplyRestriction Kernel: " OKL_KERNEL_HASH "\n\n");
        }
        @directive("#endif")

            @directive("#if USES_INDICES") for (int node = 0; node < ELEMENT_SIZE; ++node) {
          const CeedInt index = indices[node + (element * ELEMENT_SIZE)];

          for (int c = 0; c < COMPONENT_COUNT; ++c) {
            v(node, c, element) = u[index + (c * UNSTRIDED_COMPONENT_STRIDE)];
          }
        }
        @directive("#else") for (int node = 0; node < ELEMENT_SIZE; ++node) {
          for (int c = 0; c < COMPONENT_COUNT; ++c) {
            v(node, c, element) = u[(node * NODE_STRIDE) + (c * COMPONENT_STRIDE) + (element * ELEMENT_STRIDE)];
          }
        }
        @directive("#endif")
      }
    }

    @directive("#if USES_INDICES")

        @kernel void applyRestrictionTranspose(const CeedInt elementCount, const CeedInt *quadIndices, const CeedInt *dofOffsets,
                                               const CeedInt *dofIndices, const QuadVector u, CeedScalar *v) {
          @tile(TILE_SIZE, @outer, @inner) for (int n = 0; n < NODE_COUNT; ++n) {
            @directive("#if PRINT_KERNEL_HASHES")
                // Print to see which kernel is being run
                if (n == 0) {
              printf("\n\napplyRestrictionTranspose Kernel: " OKL_KERNEL_HASH "\n\n");
            }
            @directive("#endif")

                CeedScalar vComp[COMPONENT_COUNT];

            // Prefetch index information
            const CeedInt vIndex      = quadIndices[n];
            const CeedInt offsetStart = dofOffsets[n];
            const CeedInt offsetEnd   = dofOffsets[n + 1];

            for (int c = 0; c < COMPONENT_COUNT; ++c) {
              vComp[c] = 0;
            }

            // Aggregate by component
            for (CeedInt i = offsetStart; i < offsetEnd; ++i) {
              const CeedInt index = dofIndices[i];

              const int node    = (index % ELEMENT_SIZE);
              const int element = (index / ELEMENT_SIZE);

              for (int c = 0; c < COMPONENT_COUNT; ++c) {
                vComp[c] += u(node, c, element);
              }
            }

            // Update dofs by component
            for (int c = 0; c < COMPONENT_COUNT; ++c) {
              v[vIndex + (c * UNSTRIDED_COMPONENT_STRIDE)] += vComp[c];
            }
          }
        }

    @directive("#else")  // USES_INDICES = false

    @kernel void applyRestrictionTranspose(const CeedInt elementCount, const CeedInt *quadIndices, const CeedInt *dofOffsets,
                                           const CeedInt *dofIndices, const QuadVector u, CeedScalar *v) {
      @tile(TILE_SIZE, @outer, @inner) for (int element = 0; element < elementCount; ++element) {
        @directive("#if PRINT_KERNEL_HASHES")
            // Print to see which kernel is being run
            if (element == 0) {
          printf("\n\napplyRestrictionTranspose Kernel: " OKL_KERNEL_HASH "\n\n");
        }
        @directive("#endif")

            for (int node = 0; node < ELEMENT_SIZE; ++node) {
          for (int c = 0; c < COMPONENT_COUNT; ++c) {
            v[(node * NODE_STRIDE) + (c * COMPONENT_STRIDE) + (element * ELEMENT_STRIDE)] += u(node, c, element);
          }
        }
      }
    }

    @directive("#endif")  // USES_INDICES

);
