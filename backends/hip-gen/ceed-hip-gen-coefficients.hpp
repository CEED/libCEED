// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
#pragma once

#include <ceed.h>
#include <hip/hip_runtime.h>

#include <sstream>
#include <string>
#include <vector>

// Mutable module storage belongs to one operator. Reuse only immutable compiled
// bytes, never a loaded module populated with another operator's basis values.
class BasisCoefficients_Hip_gen {
 public:
  explicit BasisCoefficients_Hip_gen(size_t capacity) : capacity_(capacity), count_(0) {}

  void EmitBinding(std::ostringstream &code, const std::string &role, const std::string &suffix, const CeedScalar *source, size_t count) {
    size_t offset = count_;
    for (const auto &entry : entries_) {
      if (entry.source == source && entry.count == count) {
        offset = entry.offset;
        break;
      }
    }
    if (offset == count_) {
      if (count > capacity_ - count_) {
        code << "const CeedScalar *c_" << role << suffix << " = s_" << role << suffix << ";\n";
        return;
      }
      entries_.push_back({source, count, offset});
      count_ += count;
    }
    code << "#if CEED_GEN_CONSTANTS\nconst CeedScalar *c_" << role << suffix << " = ceed_basis_coefficients + " << offset
         << ";\n#else\nconst CeedScalar *c_" << role << suffix << " = s_" << role << suffix << ";\n#endif\n";
  }

  std::string Declaration() const {
    if (!count_) return "";
    // HIPRTC needs an explicit definition to emit a loadable constant symbol.
    return "#if CEED_GEN_CONSTANTS\nextern \"C\" __constant__ CeedScalar ceed_basis_coefficients[" + std::to_string(count_) + "] = {0};\n#endif\n";
  }

  int Upload(Ceed ceed, hipModule_t module) {
    if (!count_) return CEED_ERROR_SUCCESS;
    hipDeviceptr_t destination;
    size_t         bytes;
    CeedCallHip(ceed, hipModuleGetGlobal(&destination, &bytes, module, "ceed_basis_coefficients"));
    CeedCheck(bytes == count_ * sizeof(CeedScalar), ceed, CEED_ERROR_BACKEND, "Generated basis coefficient symbol size mismatch");
    for (const auto &entry : entries_) {
      CeedCallHip(ceed, hipMemcpy(static_cast<char *>(destination) + entry.offset * sizeof(CeedScalar), entry.source,
                                  entry.count * sizeof(CeedScalar), hipMemcpyDeviceToDevice));
    }
    CeedCallHip(ceed, hipStreamSynchronize(NULL));
    return CEED_ERROR_SUCCESS;
  }

 private:
  struct Entry {
    const CeedScalar *source;
    size_t            count, offset;
  };
  size_t             capacity_, count_;
  std::vector<Entry> entries_;
};
