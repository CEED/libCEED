#include <ceed.h>

// -----------------------------------------------------------------------------
// Redefine QFunction Macro
// -----------------------------------------------------------------------------
#undef CEED_QFUNCTION
#define CEED_QFUNCTION(name) extern int name

// -----------------------------------------------------------------------------
// QFunction Sources
// -----------------------------------------------------------------------------
#include "ex3-volume.h"

// Export the wrapper functions
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT CeedInt wrap_build_mass_diff(void *ctx, const CeedInt Q,
                                  const CeedScalar *const *in,
                                  CeedScalar *const *out) {
  return build_mass_diff(ctx, Q, in, out);
}

EXPORT CeedInt wrap_apply_mass_diff(void *ctx, const CeedInt Q,
                                  const CeedScalar *const *in,
                                  CeedScalar *const *out) {
  return apply_mass_diff(ctx, Q, in, out);
}
