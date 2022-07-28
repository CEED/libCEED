#include<stdio.h>
#include<vector>
#include <limits>
#include <sys/time.h>
#include "ceed-magma.h"

#include"./gemm_tuning/indices.h"
#include"./gemm_tuning/a100.h"
#include"./gemm_tuning/v100.h"
#include"./gemm_tuning/mi100.h"
#include"./gemm_tuning/mi250x.h"

////////////////////////////////////////////////////////////////////////////////
static void* gemm_selector_get_data(int gpu_arch, char precision, char transA)
{
  // a default
  void* data = (void*)&sgemm_nn_mi250x;

  #ifdef CEED_MAGMA_USE_HIP
  // TODO: choose between mi250x and mi100
  //if( gpu_arch >= 908 ) {
  data = ( precision == 's' ) ?
         (( transA == 'n') ? (void*)&sgemm_nn_mi250x : (void*)&sgemm_tn_mi250x ):
         (( transA == 'n') ? (void*)&dgemm_nn_mi250x : (void*)&dgemm_tn_mi250x );
  //}
  #else
  // CUDA: A100 GPU or newer
  if( gpu_arch >= 800 ) {
    data = ( precision == 's' ) ?
           (( transA == 'n') ? (void*)&sgemm_nn_a100 : (void*)&sgemm_tn_a100 ):
           (( transA == 'n') ? (void*)&dgemm_nn_a100 : (void*)&dgemm_tn_a100 );
  }
  else { // CUDA: V100 GPU or older
    data = ( precision == 's' ) ?
           (( transA == 'n') ? (void*)&sgemm_nn_v100 : (void*)&sgemm_tn_v100 ):
           (( transA == 'n') ? (void*)&dgemm_nn_v100 : (void*)&dgemm_tn_v100 );
  }
  #endif

  return data;
}

////////////////////////////////////////////////////////////////////////////////
void gemm_selector(
        int gpu_arch,
        char precision, char transA,
        int m, int n, int k,
        int *nbatch, int *use_magma )
{
    // defaults
    *nbatch    = n;
    *use_magma = 0;
    std::vector< std::vector<int> > *data = NULL;
    data =  (std::vector< std::vector<int> >*)
            gemm_selector_get_data(gpu_arch, precision, transA);

    int ir = -1;
    double norm = std::numeric_limits<double>::max();
    for(size_t i = 0; i < data->size(); i++) {
        int im = (*data)[i][M_INDEX];
        int in = (*data)[i][N_INDEX];
        int ik = (*data)[i][K_INDEX];

        double mdiff = (double)(im-m);
        double ndiff = (double)(in-n);
        double kdiff = (double)(ik-k);

        double nrm = sqrt( mdiff*mdiff + ndiff*ndiff + kdiff*kdiff );

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
        #if 0
        printf("matching record {%3d, %3d, %3d, %3d, %3d}\n",
                (*data)[ir][M_INDEX], (*data)[ir][N_INDEX], (*data)[ir][K_INDEX],
                (*data)[ir][N_BATCH_INDEX],
                (*data)[ir][USE_MAGMA_INDEX] );
        #endif
        *use_magma   = (*data)[ir][USE_MAGMA_INDEX];

        // if the closest match indicates that n = nbatch,
        // that means calling the regular non-batch gemm.
        // So nbatch is set to n instead of the 'nbatch'
        // entry of the matching record
        int n_      = (*data)[ir][N_INDEX];
        int nbatch_ = (*data)[ir][N_BATCH_INDEX];
        *nbatch     = (n_ == nbatch_) ? n : nbatch_;
    }
}
