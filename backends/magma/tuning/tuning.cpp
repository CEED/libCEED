// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

// clang-format off
// Triplets of {P, Q, dim}. For now, includes standard H1 spaces on triangles and tetrahedra with p such that P <= 40,
// and quadrature rules of degree 2p and 2p - 2. These can be expanded to more element types and quadrature rules in the
// future.
constexpr static std::array<std::array<int, 3>, 22> PQ_VALUES = {
    {{3, 1, 2}, {6,  3,  2}, {10, 6,  2}, {15, 12, 2}, {21, 16, 2}, {28, 25, 2}, {36, 33, 2},
     {3, 3, 2}, {6,  6,  2}, {10, 12, 2}, {15, 16, 2}, {21, 25, 2}, {28, 33, 2},
     {4, 1, 3}, {10, 4,  3}, {20, 11, 3}, {20, 14, 3}, {35, 24, 3},
     {4, 4, 3}, {10, 11, 3}, {10, 14, 3}, {20, 24, 3}}
};
// clang-format on

constexpr static std::array<std::pair<int, int>, 7> N_VALUES = {
    {{1024, 200}, {5120, 200}, {10240, 100}, {51200, 100}, {102400, 50}, {512000, 50}, {1024000, 25}}
};

using Clock    = std::chrono::steady_clock;
using Duration = std::chrono::duration<double>;

int main(int argc, char **argv) {
  Ceed ceed;

  std::random_device               rand_device;
  std::default_random_engine       rand_engine(rand_device());
  std::uniform_real_distribution<> rand_dist(0.0, 1.0);
  auto                             generate_random = [&rand_dist, &rand_engine]() { return rand_dist(rand_engine); };

  if (argc < 2) {
    printf("Usage: ./tuning <CEED_RESOURCE>");
    return 1;
  }
  CeedInit(argv[1], &ceed);
  CeedSetErrorHandler(ceed, CeedErrorStore);

  for (const auto [P, Q, dim] : PQ_VALUES) {
    CeedBasis  basis;
    CeedVector u, v;

    std::vector<double> q_ref(dim * Q, 0.0), q_weight(Q, 0.0), interp(P * Q), grad(P * Q * dim);
    std::generate(interp.begin(), interp.end(), generate_random);
    std::generate(grad.begin(), grad.end(), generate_random);

    CeedBasisCreateH1(ceed, (dim < 3) ? CEED_TOPOLOGY_TRIANGLE : CEED_TOPOLOGY_TET, 1, P, Q, interp.data(), grad.data(), q_ref.data(),
                      q_weight.data(), &basis);

    for (const auto [N, NUM_TRIALS] : N_VALUES) {
      double data_interp_n = 0.0, data_interp_t = 0.0, data_grad_n = 0.0, data_grad_t = 0.0;
      int    ierr;

      // Interp
      {
        CeedVectorCreate(ceed, P * N, &u);
        CeedVectorCreate(ceed, Q * N, &v);

        // NoTranspose
        CeedVectorSetValue(u, 1.0);
        CeedVectorSetValue(v, 0.0);
        ierr = CeedBasisApply(basis, N, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, u, v);
        if (!ierr) {
          const auto start = Clock::now();
          for (int trial = 0; trial < NUM_TRIALS; trial++) {
            CeedBasisApply(basis, N, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, u, v);
          }
          data_interp_n = std::chrono::duration_cast<Duration>(Clock::now() - start).count();
        }

        // Transpose
        CeedVectorSetValue(u, 1.0);
        CeedVectorSetValue(v, 0.0);
        ierr = CeedBasisApply(basis, N, CEED_TRANSPOSE, CEED_EVAL_INTERP, v, u);
        if (!ierr) {
          const auto start = Clock::now();
          for (int trial = 0; trial < NUM_TRIALS; trial++) {
            CeedBasisApply(basis, N, CEED_TRANSPOSE, CEED_EVAL_INTERP, v, u);
          }
          data_interp_t = std::chrono::duration_cast<Duration>(Clock::now() - start).count();
        }

        CeedVectorDestroy(&u);
        CeedVectorDestroy(&v);
      }

      // Grad
      {
        CeedVectorCreate(ceed, P * N, &u);
        CeedVectorCreate(ceed, dim * Q * N, &v);

        // NoTranspose
        CeedVectorSetValue(u, 1.0);
        CeedVectorSetValue(v, 0.0);
        ierr = CeedBasisApply(basis, N, CEED_NOTRANSPOSE, CEED_EVAL_GRAD, u, v);
        if (!ierr) {
          const auto start = Clock::now();
          for (int trial = 0; trial < NUM_TRIALS; trial++) {
            CeedBasisApply(basis, N, CEED_NOTRANSPOSE, CEED_EVAL_GRAD, u, v);
          }
          data_grad_n = std::chrono::duration_cast<Duration>(Clock::now() - start).count();
        }

        // Transpose
        CeedVectorSetValue(u, 1.0);
        CeedVectorSetValue(v, 0.0);
        ierr = CeedBasisApply(basis, N, CEED_TRANSPOSE, CEED_EVAL_GRAD, v, u);
        if (!ierr) {
          const auto start = Clock::now();
          for (int trial = 0; trial < NUM_TRIALS; trial++) {
            CeedBasisApply(basis, N, CEED_TRANSPOSE, CEED_EVAL_GRAD, v, u);
          }
          data_grad_t = std::chrono::duration_cast<Duration>(Clock::now() - start).count();
        }

        CeedVectorDestroy(&u);
        CeedVectorDestroy(&v);
      }

      // Postprocess and log the data
      const double  interp_flops = P * Q * (double)N;
      const double  grad_flops   = P * Q * dim * (double)N;
      constexpr int width = 12, precision = 2;
      // clang-format off
      std::printf("%-*d%-*d%-*d%-*d%-*d%*.*f\n",
                  width, P, width, Q, width, N, width, 1, width, 0, width, precision,
                  (data_interp_n > 0.0) ? 1e-6 * NUM_TRIALS * interp_flops / data_interp_n : 0.0);
      std::printf("%-*d%-*d%-*d%-*d%-*d%*.*f\n",
                  width, P, width, Q, width, N, width, 1, width, 1, width, precision,
                  (data_interp_t > 0.0) ? 1e-6 * NUM_TRIALS * interp_flops / data_interp_t : 0.0);
      std::printf("%-*d%-*d%-*d%-*d%-*d%*.*f\n",
                  width, P, width, Q, width, N, width, dim, width, 0, width, precision,
                  (data_grad_n > 0.0) ? 1e-6 * NUM_TRIALS * grad_flops / data_grad_n : 0.0);
      std::printf("%-*d%-*d%-*d%-*d%-*d%*.*f\n",
                  width, P, width, Q, width, N, width, dim, width, 1, width, precision,
                  (data_grad_t > 0.0) ? 1e-6 * NUM_TRIALS * grad_flops / data_grad_t : 0.0);
      // clang-format on
    }

    CeedBasisDestroy(&basis);
  }

  CeedDestroy(&ceed);
  return 0;
}
