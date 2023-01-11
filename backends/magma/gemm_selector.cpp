#include <stdio.h>
#include <sys/time.h>

#include <array>
#include <limits>
#include <vector>

#include "./gemm_tuning/indices.h"
#include "ceed-magma.h"
#ifdef CEED_MAGMA_USE_HIP
#include"./gemm_tuning/mi100.h"
#include"./gemm_tuning/mi250x.h"
#include"./gemm_tuning/mi250x_interp_rtc.h"
#include"./gemm_tuning/mi250x_grad_rtc.h"
#else
#include"./gemm_tuning/a100.h"
#include"./gemm_tuning/v100.h"
#include"./gemm_tuning/a100_interp_rtc.h"
#include"./gemm_tuning/a100_grad_rtc.h"
#endif

////////////////////////////////////////////////////////////////////////////////
static void *gemm_selector_get_data(int gpu_arch, char precision, char transA) {
// a default
#ifdef CEED_MAGMA_USE_HIP
  void *data = (void *)&sgemm_nn_mi250x;
#else
  void *data = (void *)&sgemm_nn_a100;
#endif

#ifdef CEED_MAGMA_USE_HIP
  if (gpu_arch >= 910) {
    // gfx90a or newer
    data = (precision == 's') ? ((transA == 'n') ? (void *)&sgemm_nn_mi250x : (void *)&sgemm_tn_mi250x)
                              : ((transA == 'n') ? (void *)&dgemm_nn_mi250x : (void *)&dgemm_tn_mi250x);
  } else {
    // gfx908 or older
    data = (precision == 's') ? ((transA == 'n') ? (void *)&sgemm_nn_mi100 : (void *)&sgemm_tn_mi100)
                              : ((transA == 'n') ? (void *)&dgemm_nn_mi100 : (void *)&dgemm_tn_mi100);
  }
#else
  if (gpu_arch >= 800) {
    // sm80 or newer
    data = (precision == 's') ? ((transA == 'n') ? (void *)&sgemm_nn_a100 : (void *)&sgemm_tn_a100)
                              : ((transA == 'n') ? (void *)&dgemm_nn_a100 : (void *)&dgemm_tn_a100);
  } else {
    // sm70 or older
    data = (precision == 's') ? ((transA == 'n') ? (void *)&sgemm_nn_v100 : (void *)&sgemm_tn_v100)
                              : ((transA == 'n') ? (void *)&dgemm_nn_v100 : (void *)&dgemm_tn_v100);
  }
#endif

  return data;
}

////////////////////////////////////////////////////////////////////////////////
void gemm_selector(int gpu_arch, char precision, char transA, int m, int n, int k, int *nbatch, int *use_magma) {
  // defaults
  *nbatch                                            = n;
  *use_magma                                         = 0;
  std::vector<std::array<int, RECORD_LENGTH> > *data = NULL;
  data = (std::vector<std::array<int, RECORD_LENGTH> > *)gemm_selector_get_data(gpu_arch, precision, transA);

  int    ir   = -1;
  double norm = std::numeric_limits<double>::max();
  for (size_t i = 0; i < data->size(); i++) {
    int im = (*data)[i][M_INDEX];
    int in = (*data)[i][N_INDEX];
    int ik = (*data)[i][K_INDEX];

    double mdiff = (double)(im - m);
    double ndiff = (double)(in - n);
    double kdiff = (double)(ik - k);

    double nrm = sqrt(mdiff * mdiff + ndiff * ndiff + kdiff * kdiff);

    if (nrm < norm) {
      norm = nrm;
      ir   = i;
    }

    if (nrm == 0) {
      // the input (m, n, k) exactly matches a record in `data`
      // no need to search further
      break;
    }
  }

  if( ir >= 0 ) {
    *use_magma = (*data)[ir][USE_MAGMA_INDEX];
    // if the closest match indicates that n = nbatch,
    // that means calling the regular non-batch gemm.
    // So nbatch is set to n instead of the 'nbatch'
    // entry of the matching record
    int n_      = (*data)[ir][N_INDEX];
    int nbatch_ = (*data)[ir][N_BATCH_INDEX];
    *nbatch     = (n_ == nbatch_) ? n : nbatch_;
  }
}

////////////////////////////////////////////////////////////////////////////////
static void*
nontensor_rtc_get_data(
  int gpu_arch, char precision,
  CeedEvalMode emode, CeedTransposeMode tmode)
{
  // a default
  #ifdef CEED_MAGMA_USE_HIP
  void* data = (void*)&dinterp_n_mi250x;
  #else
  void* data = (void*)&dinterp_n_a100;
  #endif

  #ifdef CEED_MAGMA_USE_HIP
  if( emode == CEED_EVAL_INTERP ) {
    data = ( tmode == CEED_TRANSPOSE ) ?
           (void*)&dinterp_t_mi250x:
           (void*)&dinterp_n_mi250x;
  }
  else if( emode == CEED_EVAL_GRAD ) {
    data = ( tmode == CEED_TRANSPOSE ) ?
           (void*)&dgrad_t_mi250x:
           (void*)&dgrad_n_mi250x;
  }
  #else
  if( emode == CEED_EVAL_INTERP ) {
    data = ( tmode == CEED_TRANSPOSE ) ?
           (void*)&dinterp_t_a100:
           (void*)&dinterp_n_a100;
  }
  else if( emode == CEED_EVAL_GRAD ) {
    data = ( tmode == CEED_TRANSPOSE ) ?
           (void*)&dgrad_t_a100:
           (void*)&dgrad_n_a100;
  }
  #endif

  return data;
}

////////////////////////////////////////////////////////////////////////////////
CeedInt nontensor_rtc_get_nb(
        int gpu_arch, char precision,
        CeedEvalMode emode, CeedTransposeMode tmode,
        int P_, int N, int Q_ )
{
    CeedInt P  = ( tmode == CEED_TRANSPOSE ) ? P_ : Q_;
    CeedInt Q  = ( tmode == CEED_TRANSPOSE ) ? Q_ : P_;
    CeedInt NB = 1;

    std::vector< std::array<int, RECORD_LENGTH_RTC> > *data = NULL;
    data =  (std::vector< std::array<int, RECORD_LENGTH_RTC> >*)
            nontensor_rtc_get_data(gpu_arch, precision, emode, tmode);

    int ir = -1;
    double norm = std::numeric_limits<double>::max();
    for(size_t i = 0; i < data->size(); i++) {
        int ip = (*data)[i][M_INDEX_RTC];
        int in = (*data)[i][N_INDEX_RTC];
        int iq = (*data)[i][K_INDEX_RTC];

        double pdiff = (double)(ip-P);
        double ndiff = (double)(in-N);
        double qdiff = (double)(iq-Q);
        double nrm = sqrt( pdiff*pdiff + ndiff*ndiff + qdiff*qdiff );

        if( nrm < norm ) {
            norm = nrm;
            ir = i;
        }

        if( nrm == 0 ) {
            // the input (m, n, k) exactly matches a record in `data`
            // no need to search further
            break;
        }
    }

    if( ir >= 0 ) {
        NB   = (*data)[ir][NB_INDEX_RTC];
    }

    return NB;
}
