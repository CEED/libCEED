// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>

// H(curl) basis for order 2 Nédélec (first kind) triangle element in 2D
// See: https://defelement.com/elements/examples/triangle-N1curl-2.html
static void BuildHcurl2DSimplex(CeedScalar *q_ref, CeedScalar *q_weight, CeedScalar *interp, CeedScalar *curl) {
  CeedInt P = 8, Q = 4;

  q_ref[0]    = 1. / 3.;
  q_ref[4]    = 1. / 3.;
  q_ref[1]    = 0.2;
  q_ref[5]    = 0.2;
  q_ref[2]    = 0.2;
  q_ref[6]    = 0.6;
  q_ref[3]    = 0.6;
  q_ref[7]    = 0.2;
  q_weight[0] = -25. / 96.;
  q_weight[1] = 25. / 96.;
  q_weight[2] = 27. / 96.;
  q_weight[3] = 25. / 96.;

  // Loop over quadrature points
  for (int i = 0; i < Q; i++) {
    CeedScalar x1 = q_ref[0 * Q + i], x2 = q_ref[1 * Q + i];
    // Interp
    interp[(i + 0) * P + 0] = 2. * x2 * (1. - 4. * x1);
    interp[(i + Q) * P + 0] = 4. * x1 * (2. * x1 - 1.);
    interp[(i + 0) * P + 1] = 4. * x2 * (1. - 2. * x2);
    interp[(i + Q) * P + 1] = 2. * x1 * (4. * x2 - 1.);
    interp[(i + 0) * P + 2] = 2. * x2 * (-4. * x1 - 4. * x2 + 3.);
    interp[(i + Q) * P + 2] = 8. * x1 * x1 + 8. * x1 * x2 - 12. * x1 - 6. * x2 + 4.;
    interp[(i + 0) * P + 3] = 4. * x2 * (2. * x2 - 1.);
    interp[(i + Q) * P + 3] = -8. * x1 * x2 + 2. * x1 + 6. * x2 - 2.;
    interp[(i + 0) * P + 4] = 8. * x1 * x2 - 6. * x1 + 8. * x2 * x2 - 12. * x2 + 4.;
    interp[(i + Q) * P + 4] = 2. * x1 * (-4. * x1 - 4. * x2 + 3.);
    interp[(i + 0) * P + 5] = -8. * x1 * x2 + 6. * x1 + 2. * x2 - 2.;
    interp[(i + Q) * P + 5] = 4. * x1 * (2. * x1 - 1.);
    interp[(i + 0) * P + 6] = 8. * x2 * (-x1 - 2. * x2 + 2.);
    interp[(i + Q) * P + 6] = 8. * x1 * (x1 + 2. * x2 - 1.);
    interp[(i + 0) * P + 7] = 8. * x2 * (2. * x1 + x2 - 1.);
    interp[(i + Q) * P + 7] = 8. * x1 * (-2. * x1 - x2 + 2.);
    // Curl
    curl[i * P + 0] = 24. * x1 - 6.;
    curl[i * P + 1] = 24. * x2 - 6.;
    curl[i * P + 2] = 24. * x1 + 24. * x2 - 18.;
    curl[i * P + 3] = -24. * x2 + 6.;
    curl[i * P + 4] = -24. * x1 - 24. * x2 + 18.;
    curl[i * P + 5] = 24. * x1 - 6.;
    curl[i * P + 6] = 24. * x1 + 48. * x2 - 24.;
    curl[i * P + 7] = -48. * x1 - 24. * x1 + 24.;
  }
}
