// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed.h>
#include <ceed/backend.h>
#include <hip/hip_runtime.h>

#include <sstream>
#include <string>

typedef struct {
  const CeedScalar *source;
  size_t            count, offset;
} CeedBasisCoefficientEntry_Hip_gen;

// Each field binds at most an interpolation and a gradient matrix. Populated
// modules belong to one operator; only immutable compiled bytes may be reused.
typedef struct {
  size_t                            capacity, count;
  CeedInt                           num_entries;
  CeedBasisCoefficientEntry_Hip_gen entries[4 * CEED_FIELD_MAX];
} CeedBasisCoefficients_Hip_gen;

static void CeedBasisCoefficientsEmitBinding_Hip_gen(CeedBasisCoefficients_Hip_gen *coefficients, std::ostringstream &code, Tab &tab,
                                                     const std::string &role, const std::string &suffix, const CeedScalar *source, size_t count) {
  size_t offset = coefficients->count;

  for (CeedInt i = 0; i < coefficients->num_entries; i++) {
    const CeedBasisCoefficientEntry_Hip_gen *entry = &coefficients->entries[i];

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

static std::string CeedBasisCoefficientsDeclaration_Hip_gen(const CeedBasisCoefficients_Hip_gen *coefficients) {
  if (!coefficients->count) return "";
  return "#if CEED_GEN_CONSTANTS\nextern \"C\" __constant__ CeedScalar ceed_basis_coefficients[" + std::to_string(coefficients->count) +
         "] = {0};\n#endif\n";
}

static int CeedBasisCoefficientsUpload_Hip_gen(const CeedBasisCoefficients_Hip_gen *coefficients, Ceed ceed, hipModule_t module) {
  if (!coefficients->count) return CEED_ERROR_SUCCESS;
  hipDeviceptr_t destination;
  size_t         bytes;

  CeedCallHip(ceed, hipModuleGetGlobal(&destination, &bytes, module, "ceed_basis_coefficients"));
  CeedCheck(bytes == coefficients->count * sizeof(CeedScalar), ceed, CEED_ERROR_BACKEND, "Generated basis coefficient symbol size mismatch");
  for (CeedInt i = 0; i < coefficients->num_entries; i++) {
    const CeedBasisCoefficientEntry_Hip_gen *entry = &coefficients->entries[i];

    CeedCallHip(ceed, hipMemcpy(static_cast<char *>(destination) + entry->offset * sizeof(CeedScalar), entry->source,
                                entry->count * sizeof(CeedScalar), hipMemcpyDeviceToDevice));
  }
  CeedCallHip(ceed, hipStreamSynchronize(NULL));
  return CEED_ERROR_SUCCESS;
}
