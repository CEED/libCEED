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
#include "tuning/mi100_rtc.h"
#include "tuning/mi250x.h"
#include "tuning/mi250x_rtc.h"
#else
#include "tuning/a100.h"
#include "tuning/a100_rtc.h"
#include "tuning/h100_rtc.h"
#include "tuning/v100.h"
#include "tuning/v100_rtc.h"
#endif

// These definitions to force a certain parameter when generating autotuning data offline
// #define CEED_AUTOTUNE_GEMM_SELECTOR_N_BATCH
// #define CEED_AUTOTUNE_GEMM_SELECTOR_USE_MAGMA
// #define CEED_AUTOTUNE_RTC_NB

////////////////////////////////////////////////////////////////////////////////
#ifdef CEED_MAGMA_USE_HIP
static inline auto gemm_selector_get_data(int gpu_arch, char precision, char trans_A) -> decltype(dgemm_nn_mi100) {
  if (gpu_arch >= 910) {
    // gfx90a or newer
    return (precision == 's') ? ((trans_A == 'n') ? sgemm_nn_mi250x : sgemm_tn_mi250x) : ((trans_A == 'n') ? dgemm_nn_mi250x : dgemm_tn_mi250x);
  } else {
    // gfx908 or older
    return (precision == 's') ? ((trans_A == 'n') ? sgemm_nn_mi100 : sgemm_tn_mi100) : ((trans_A == 'n') ? dgemm_nn_mi100 : dgemm_tn_mi100);
  }
}
#else
static inline auto gemm_selector_get_data(int gpu_arch, char precision, char trans_A) -> decltype(dgemm_nn_v100) {
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
#if defined(CEED_AUTOTUNE_GEMM_SELECTOR_N_BATCH) && defined(CEED_AUTOTUNE_GEMM_SELECTOR_USE_MAGMA)
  *n_batch   = CEED_AUTOTUNE_GEMM_SELECTOR_N_BATCH;
  *use_magma = CEED_AUTOTUNE_GEMM_SELECTOR_USE_MAGMA;
#else
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
#endif
}

//////////////////////////////////////////////////////////////////////////////
#ifdef CEED_MAGMA_USE_HIP
static inline auto nontensor_rtc_get_data(int gpu_arch, char trans_A) -> decltype(drtc_n_mi100) {
  if (gpu_arch >= 910) {
    // gfx90a or newer
    return (trans_A == 'n') ? drtc_n_mi250x : drtc_t_mi250x;
  } else {
    // gfx908 or older
    return (trans_A == 'n') ? drtc_n_mi100 : drtc_t_mi100;
  }
}
#else
static inline auto nontensor_rtc_get_data(int gpu_arch, char trans_A) -> decltype(drtc_n_v100) {
  if (gpu_arch >= 900) {
    // sm90 or newer
    return (trans_A == 'n') ? drtc_n_h100 : drtc_t_h100;
  } else if (gpu_arch >= 800) {
    // sm80 or newer
    return (trans_A == 'n') ? drtc_n_a100 : drtc_t_a100;
  } else {
    // sm70 or older
    return (trans_A == 'n') ? drtc_n_v100 : drtc_t_v100;
  }
}
#endif

////////////////////////////////////////////////////////////////////////////////
CeedInt nontensor_rtc_get_nb(int gpu_arch, char trans_A, int q_comp, int P, int Q, int N) {
#ifdef CEED_AUTOTUNE_RTC_NB
  return CEED_AUTOTUNE_RTC_NB;
#else
  const auto &data = nontensor_rtc_get_data(gpu_arch, trans_A);
  int         ir   = -1;
  double      norm = std::numeric_limits<double>::max();

  for (size_t i = 0; i < data.size(); i++) {
    // Only seach exact matches for q_comp
    if (q_comp != data[i][Q_COMP_INDEX_RTC]) {
      continue;
    }

    const int &iP = data[i][P_INDEX_RTC];
    const int &iQ = data[i][Q_INDEX_RTC];
    const int &iN = data[i][N_INDEX_RTC];

    double Pdiff = (double)(iP - P);
    double Qdiff = (double)(iQ - Q);
    double Ndiff = (double)(iN - N);
    double nrm   = Pdiff * Pdiff + Qdiff * Qdiff + Ndiff * Ndiff;

    if (nrm < norm) {
      norm = nrm;
      ir   = i;
    }

    if (iP == P && iQ == Q && iN == N) {
      // The input (P, Q, N) exactly matches a record in `data`, no need to search further
      break;
    }
  }

  return (ir >= 0) ? data[ir][NB_INDEX_RTC] : 1;
#endif
}
