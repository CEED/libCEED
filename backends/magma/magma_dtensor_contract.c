#include <ceed-impl.h>
#include <string.h>
#include "magma.h"

// Contracts on the middle index
// NOTRANSPOSE: V_ajc = T_jb U_abc
// TRANSPOSE:   V_ajc = T_bj U_abc
// If Add != 0, "=" is replaced by "+="
void magma_dtensor_contract(Ceed ceed,
                            CeedInt A, CeedInt B, CeedInt C, CeedInt J,
                            const CeedScalar *t, CeedTransposeMode tmode,
                            const CeedInt Add,
                            const CeedScalar *u, CeedScalar *v) {
  magma_init();
  double *dT, *dU, *dV;
  double **dT_array, **dU_array, **dV_array;
  magma_trans_t transT = (tmode == CEED_TRANSPOSE) ? MagmaTrans : MagmaNoTrans;
  magma_trans_t transU = MagmaNoTrans;
  magma_int_t Tm, Tn, Um, Un, Vm, Vn;
  magma_int_t ldt, ldu, ldv;
  magma_int_t batchCount = A;
  double alpha = 1;
  double beta = (Add) ? 1 : 0;
  if ( transT == MagmaNoTrans ) {
    ldt = Tm = B;
    Tn = J;
  } else {
    ldt = Tm = J;
    Tn = B;
  }
  ldu = Um = C;
  Un = B;

  ldv = Vm = C;
  Vn = J;

  magma_dmalloc( &dT, ldt*Tn            );
  magma_dmalloc( &dU, ldu*Un*batchCount );
  magma_dmalloc( &dV, ldv*Vn*batchCount );
  magma_malloc((void**)&dT_array, batchCount * sizeof(double*));
  magma_malloc((void**)&dU_array, batchCount * sizeof(double*));
  magma_malloc((void**)&dV_array, batchCount * sizeof(double*));

  magma_queue_t queue;
  magma_queue_create( &queue );

  magma_dsetmatrix( Tm, Tn, t, ldt, dT, ldt );
  magma_dsetmatrix( Um, Un*batchCount, u, ldu, dU, ldu );
  magma_dsetmatrix( Vm, Vn*batchCount, v, ldv, dV, ldv );
  magma_dset_pointer( dT_array, dT, ldt, 0, 0,      0, batchCount, queue);
  magma_dset_pointer( dU_array, dU, ldu, 0, 0, ldu*Un, batchCount, queue);
  magma_dset_pointer( dV_array, dV, ldv, 0, 0, ldv*Vn, batchCount, queue);
  magmablas_dgemm_batched(transU, transT, C, J, B,
                          alpha, (double const * const *)dU_array, ldu,
                          (double const * const *)dT_array, ldt,
                          beta,  dV_array, ldv, batchCount, queue);
  magma_queue_sync( queue );
  magma_dgetmatrix( Vm, Vn*batchCount, dV, ldv, v, ldv );
  magma_queue_destroy( queue );
  magma_free(dT_array);
  magma_free(dU_array);
  magma_free(dV_array);
  magma_free(dT);
  magma_free(dU);
  magma_free(dV);
  magma_finalize();
}
