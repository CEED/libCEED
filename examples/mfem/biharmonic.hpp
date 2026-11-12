#ifndef BIHARMONIC_HPP
#define BIHARMONIC_HPP

#include "biharmonic.h"
#include <ceed.h>

/// CEED QFunction for building quadrature data for diffusion
CEED_QFUNCTION(f_build_diff)(void *ctx, CeedInt Q,
                             const CeedScalar *const *in,
                             CeedScalar *const *out);

/// CEED QFunction for applying diffusion operator
CEED_QFUNCTION(f_apply_diff)(void *ctx, CeedInt Q,
                             const CeedScalar *const *in,
                             CeedScalar *const *out);

#endif // BIHARMONIC_HPP
