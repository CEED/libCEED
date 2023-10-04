// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
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

// XX TODO WIP: Add other quadrature orders, prism/pyramid, ...
// clang-format off
constexpr static std::array<std::array<int, 3>, 11> PQ_VALUES = {
    {{3, 1, 2}, {6, 3,  2}, {10, 6,  2}, {15, 12, 2}, {21, 16, 2}, {28, 25, 2}, {36, 33, 2},
     {4, 1, 3}, {10, 4, 3}, {20, 11, 3}, {35, 24, 3}}
};
// clang-format on

constexpr static std::array<int, 7> N_VALUES = {1024, 5120, 10240, 51200, 102400, 512000, 1024000};

constexpr int NUM_TRIALS = 25;

using Clock    = std::chrono::steady_clock;
using Duration = std::chrono::duration<double>;

int main(int argc, char **argv) {
  Ceed ceed;

  std::random_device               rand_device;
  std::default_random_engine       rand_engine(rand_device());
  std::uniform_real_distribution<> rand_dist(0.0, 1.0);
  auto                             generate_random = [&rand_dist, &rand_engine]() { return rand_dist(rand_engine); };

  CeedInit((argc < 2) ? "/gpu/cuda/magma" : argv[1], &ceed);
  CeedSetErrorHandler(ceed, CeedErrorStore);

  for (const auto [P, Q, dim] : PQ_VALUES) {
    CeedBasis  basis;
    CeedVector u, v;

    std::vector<double> q_ref(dim * Q, 0.0), q_weight(Q, 0.0), interp(P * Q), grad(P * Q * dim);
    std::generate(interp.begin(), interp.end(), generate_random);
    std::generate(grad.begin(), grad.end(), generate_random);

    CeedBasisCreateH1(ceed, (dim < 3) ? CEED_TOPOLOGY_TRIANGLE : CEED_TOPOLOGY_TET, 1, P, Q, interp.data(), grad.data(), q_ref.data(),
                      q_weight.data(), &basis);

    for (const auto N : N_VALUES) {
      double data_interp_n = 0.0, data_interp_t = 0.0, data_grad_n = 0.0, data_grad_t = 0.0;

      // Interp
      {
        CeedVectorCreate(ceed, P * N, &u);
        CeedVectorCreate(ceed, Q * N, &v);

        // NoTranspose
        CeedVectorSetValue(u, 1.0);
        for (int trial = 0; trial <= NUM_TRIALS; trial++) {
          CeedVectorSetValue(v, 0.0);

          const auto start = Clock::now();
          int        ierr  = CeedBasisApply(basis, N, CEED_NOTRANSPOSE, CEED_EVAL_INTERP, u, v);
          if (ierr) {
            break;
          }
          if (trial > 0) {
            data_interp_n += std::chrono::duration_cast<Duration>(Clock::now() - start).count();
          }
        }

        // Transpose
        CeedVectorSetValue(v, 1.0);
        for (int trial = 0; trial <= NUM_TRIALS; trial++) {
          CeedVectorSetValue(u, 0.0);

          const auto start = Clock::now();
          int        ierr  = CeedBasisApply(basis, N, CEED_TRANSPOSE, CEED_EVAL_INTERP, v, u);
          if (ierr) {
            break;
          }
          if (trial > 0) {
            data_interp_t += std::chrono::duration_cast<Duration>(Clock::now() - start).count();
          }
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
        for (int trial = 0; trial < NUM_TRIALS; trial++) {
          CeedVectorSetValue(v, 0.0);

          const auto start = Clock::now();
          int        ierr  = CeedBasisApply(basis, N, CEED_NOTRANSPOSE, CEED_EVAL_GRAD, u, v);
          if (ierr) {
            break;
          }
          if (trial > 0) {
            data_grad_n += std::chrono::duration_cast<Duration>(Clock::now() - start).count();
          }
        }

        // Transpose
        CeedVectorSetValue(v, 1.0);
        for (int trial = 0; trial < NUM_TRIALS; trial++) {
          CeedVectorSetValue(u, 0.0);

          const auto start = Clock::now();
          int        ierr  = CeedBasisApply(basis, N, CEED_TRANSPOSE, CEED_EVAL_GRAD, v, u);
          if (ierr) {
            break;
          }
          if (trial > 0) {
            data_grad_t += std::chrono::duration_cast<Duration>(Clock::now() - start).count();
          }
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
                  width, P, width, N, width, Q, width, 1, width, 0, width, precision,
                  (data_interp_n > 0.0) ? NUM_TRIALS * interp_flops / data_interp_n * 1.0e-6 : 0.0);
      std::printf("%-*d%-*d%-*d%-*d%-*d%*.*f\n",
                  width, P, width, N, width, Q, width, 1, width, 1, width, precision,
                  (data_interp_t > 0.0) ? NUM_TRIALS * interp_flops / data_interp_t * 1.0e-6 : 0.0);
      std::printf("%-*d%-*d%-*d%-*d%-*d%*.*f\n",
                  width, P, width, N, width, Q, width, dim, width, 0, width, precision,
                  (data_grad_n > 0.0) ? NUM_TRIALS * grad_flops / data_grad_n * 1.0e-6 : 0.0);
      std::printf("%-*d%-*d%-*d%-*d%-*d%*.*f\n",
                  width, P, width, N, width, Q, width, dim, width, 1, width, precision,
                  (data_grad_n > 0.0) ? NUM_TRIALS * grad_flops / data_grad_n * 1.0e-6 : 0.0);
      // clang-format on
    }

    CeedBasisDestroy(&basis);
  }

  CeedDestroy(&ceed);
  return 0;
}
