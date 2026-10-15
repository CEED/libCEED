// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed.h>
#include <ceed/backend.h>
#include <cuda.h>

#include <sstream>
#include <string>

typedef struct {
  const CeedScalar *source;
  size_t            count, offset;
} CeedBasisCoefficientEntry_Cuda_gen;

// Each field binds at most an interpolation and a gradient matrix. Populated
// modules belong to one operator; only immutable compiled bytes may be reused.
typedef struct {
  size_t                             capacity, count;
  CeedInt                            num_entries;
  CeedBasisCoefficientEntry_Cuda_gen entries[4 * CEED_FIELD_MAX];
} CeedBasisCoefficients_Cuda_gen;

static void CeedBasisCoefficientsEmitBinding_Cuda_gen(CeedBasisCoefficients_Cuda_gen *coefficients, std::ostringstream &code, Tab &tab,
                                                      const std::string &role, const std::string &suffix, const CeedScalar *source, size_t count) {
  size_t offset = coefficients->count;

  for (CeedInt i = 0; i < coefficients->num_entries; i++) {
    const CeedBasisCoefficientEntry_Cuda_gen *entry = &coefficients->entries[i];

    if (entry->source == source && entry->count == count) {
      offset = entry->offset;
      break;
    }
  }
  if (offset == coefficients->count) {
    if (count > coefficients->capacity - coefficients->count || coefficients->num_entries == 4 * CEED_FIELD_MAX) {
      code << tab << "const CeedScalar *c_" << role << suffix << " = s_" << role << suffix << ";\n";
      return;
    }
    coefficients->entries[coefficients->num_entries++] = {source, count, offset};
    coefficients->count += count;
  }
  code << "#if CEED_GEN_CONSTANTS\n"
       << tab << "const CeedScalar *c_" << role << suffix << " = ceed_basis_coefficients + " << offset << ";\n#else\n"
       << tab << "const CeedScalar *c_" << role << suffix << " = s_" << role << suffix << ";\n#endif\n";
}

static std::string CeedBasisCoefficientsDeclaration_Cuda_gen(const CeedBasisCoefficients_Cuda_gen *coefficients) {
  if (!coefficients->count) return "";
  return "#if CEED_GEN_CONSTANTS\nextern \"C\" __constant__ CeedScalar ceed_basis_coefficients[" + std::to_string(coefficients->count) +
         "];\n#endif\n";
}

static int CeedBasisCoefficientsUpload_Cuda_gen(const CeedBasisCoefficients_Cuda_gen *coefficients, Ceed ceed, CUmodule module) {
  if (!coefficients->count) return CEED_ERROR_SUCCESS;
  CUdeviceptr destination;
  size_t      bytes;

  CeedCallCuda(ceed, cuModuleGetGlobal(&destination, &bytes, module, "ceed_basis_coefficients"));
  CeedCheck(bytes == coefficients->count * sizeof(CeedScalar), ceed, CEED_ERROR_BACKEND, "Generated basis coefficient symbol size mismatch");
  for (CeedInt i = 0; i < coefficients->num_entries; i++) {
    const CeedBasisCoefficientEntry_Cuda_gen *entry = &coefficients->entries[i];

    CeedCallCuda(ceed, cuMemcpyDtoD(destination + entry->offset * sizeof(CeedScalar), reinterpret_cast<CUdeviceptr>(entry->source),
                                    entry->count * sizeof(CeedScalar)));
  }
  return CEED_ERROR_SUCCESS;
}
