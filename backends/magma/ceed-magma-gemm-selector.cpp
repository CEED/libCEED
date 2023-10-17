// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <array>
#include <limits>
#include <vector>

#include "ceed-magma-gemm-selector.h"

#include "tuning/indices.h"
#ifdef CEED_MAGMA_USE_HIP
#include "tuning/mi100.h"
#include "tuning/mi250x.h"
#include "tuning/mi250x_grad_rtc.h"
#include "tuning/mi250x_interp_rtc.h"
#else
#include "tuning/a100.h"
#include "tuning/a100_grad_rtc.h"
#include "tuning/a100_interp_rtc.h"
#include "tuning/v100.h"
#endif

////////////////////////////////////////////////////////////////////////////////
#ifdef CEED_MAGMA_USE_HIP
static inline auto gemm_selector_get_data(int gpu_arch, char precision, char trans_A) -> decltype(dgemm_nn_mi250x) {
  if (gpu_arch >= 910) {
    // gfx90a or newer
    return (precision == 's') ? ((trans_A == 'n') ? sgemm_nn_mi250x : sgemm_tn_mi250x) : ((trans_A == 'n') ? dgemm_nn_mi250x : dgemm_tn_mi250x);
  } else {
    // gfx908 or older
    return (precision == 's') ? ((trans_A == 'n') ? sgemm_nn_mi100 : sgemm_tn_mi100) : ((trans_A == 'n') ? dgemm_nn_mi100 : dgemm_tn_mi100);
  }
}
#else
static inline auto gemm_selector_get_data(int gpu_arch, char precision, char trans_A) -> decltype(dgemm_nn_a100) {
  if (gpu_arch >= 800) {
    // sm80 or newer
    return (precision == 's') ? ((trans_A == 'n') ? sgemm_nn_a100 : sgemm_tn_a100) : ((trans_A == 'n') ? dgemm_nn_a100 : dgemm_tn_a100);
  } else {
    // sm70 or older
    return (precision == 's') ? ((trans_A == 'n') ? sgemm_nn_v100 : sgemm_tn_v100) : ((trans_A == 'n') ? dgemm_nn_v100 : dgemm_tn_v100);
  }
}
#endif

////////////////////////////////////////////////////////////////////////////////
void gemm_selector(int gpu_arch, char precision, char trans_A, int m, int n, int k, int *n_batch, int *use_magma) {
  const auto &data = gemm_selector_get_data(gpu_arch, precision, trans_A);
  int         ir   = -1;
  double      norm = std::numeric_limits<double>::max();

  for (size_t i = 0; i < data.size(); i++) {
    const int &im = data[i][M_INDEX];
    const int &in = data[i][N_INDEX];
    const int &ik = data[i][K_INDEX];

    double mdiff = (double)(im - m);
    double ndiff = (double)(in - n);
    double kdiff = (double)(ik - k);
    double nrm   = mdiff * mdiff + ndiff * ndiff + kdiff * kdiff;

    if (nrm < norm) {
      norm = nrm;
      ir   = i;
    }

    if (im == m && in == n && ik == k) {
      // The input (m, n, k) exactly matches a record in `data`, no need to search further
      break;
    }
  }

  if (ir >= 0) {
    // If the closest match indicates that n = n_batch, that means calling the regular non-batch GEMM.
    // So n_batch is set to n instead of the 'n_batch' entry of the matching record.
    int n_       = data[ir][N_INDEX];
    int n_batch_ = data[ir][N_BATCH_INDEX];
    *n_batch     = (n_ == n_batch_) ? n : n_batch_;
    *use_magma   = data[ir][USE_MAGMA_INDEX];
  } else {
    *n_batch   = n;
    *use_magma = 0;
  }
}

//////////////////////////////////////////////////////////////////////////////
#ifdef CEED_MAGMA_USE_HIP
static inline auto nontensor_rtc_get_data(int gpu_arch, char trans_A, int q_comp) -> decltype(dinterp_n_mi250x) {
  if (q_comp == 1) {
    return (trans_A == 'n') ? dinterp_n_mi250x : dinterp_t_mi250x;
  } else {
    return (trans_A == 'n') ? dgrad_n_mi250x : dgrad_t_mi250x;
  }
}
#else
static inline auto nontensor_rtc_get_data(int gpu_arch, char trans_A, int q_comp) -> decltype(dinterp_n_a100) {
  if (q_comp == 1) {
    return (trans_A == 'n') ? dinterp_n_a100 : dinterp_t_a100;
  } else {
    return (trans_A == 'n') ? dgrad_n_a100 : dgrad_t_a100;
  }
}
#endif

////////////////////////////////////////////////////////////////////////////////
CeedInt nontensor_rtc_get_nb(int gpu_arch, char trans_A, int q_comp, int P, int Q, int n) {
  const auto &data = nontensor_rtc_get_data(gpu_arch, trans_A, q_comp);
  int         ir   = -1;
  double      norm = std::numeric_limits<double>::max();
  CeedInt     m    = (trans_A == 'n') ? Q : P;
  CeedInt     k    = (trans_A == 'n') ? P : Q;

  for (size_t i = 0; i < data.size(); i++) {
    const int &im = data[i][M_INDEX_RTC];
    const int &in = data[i][N_INDEX_RTC];
    const int &ik = data[i][K_INDEX_RTC];

    double mdiff = (double)(im - m);
    double ndiff = (double)(in - n);
    double kdiff = (double)(ik - k);
    double nrm   = mdiff * mdiff + ndiff * ndiff + kdiff * kdiff;

    if (nrm < norm) {
      norm = nrm;
      ir   = i;
    }

    if (im == m && in == n && ik == k) {
      // The input (m, n, k) exactly matches a record in `data`, no need to search further
      break;
    }
  }

  return (ir >= 0) ? data[ir][NB_INDEX_RTC] : 1;
}
