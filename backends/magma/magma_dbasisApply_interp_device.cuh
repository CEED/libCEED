#ifndef MAGMA_BASIS_APPLY_INTERP_DEVICE_CUH
#define MAGMA_BASIS_APPLY_INTERP_DEVICE_CUH

//////////////////////////////////////////////////////////////////////////////////////////
template<int P, int Q>
static __device__ __inline__ void
dbasis_apply_eval_interp_device( 
    const int dim, const int ncomp, const int pre_org, const int post_org, const int tmp_size, 
    const double *dT, magma_trans_t transT,
    const double *dU, double *dV, 
    double* shared_data )
{
#define B    (P)
#define J    (Q)
#define A    (pre)
#define C    (post)

    const int nthreads = blockDim.x;
    int pre      = pre_org;
    int post     = post_org;
    int nslices  = nthreads / C;
    int tx       = threadIdx.x;
    int tx_      = tx % C;
    int slice_id = tx / C;
    int i = 0;

    const magma_int_t add = (transT == MagmaTrans);
    
    double* sT    = (double*)shared_data;
    double* sTmp1 = sT + B * J;
    double* sTmp2 = sTmp1 + tmp_size; 
    double rU[P]  = { MAGMA_D_ZERO };    // each thread has an entire row of U
    double rV[Q]  = { MAGMA_D_ZERO };    // each thread computes an entire row of V
    double *sU, *sV; 

    // read T in shared memory
    dread_T_gm2sm<B, J>(tx, transT, dT, sT );    

    sU = sTmp1; 
    sV = sTmp2;

    // read U in sTmp1 (AC x B)
    sU += slice_id * C * B;
    dU += slice_id * C * B;
    #pragma unroll
    for(i = 0; i < A-nslices; i+=nslices) {
        #pragma unroll
        for(int b = 0; b < B; b++) {
            sU[b * C + tx_] = dU[b * C + tx_];
        }
        dU += nslices * C * B;
        sU += nslices * C * B;
    }
    
    if(slice_id < A-i) {
        #pragma unroll
        for(int b = 0; b < B; b++) {
            //printf("tx = %d, tx_ = %d, accessing b * C + tx_ = %d\n", tx, tx_, b * C + tx_);
            sU[b * C + tx_] = dU[b * C + tx_];
        }
    }
    __syncthreads();

    int d = 0; 
    #pragma unroll
    for(d = 0; d < dim-1; d++) {
        sU = (d % 2 == 0) ? sTmp1 : sTmp2;
        sV = (d % 2 == 0) ? sTmp2 : sTmp1;
        
        sU += slice_id * C * B;
        sV += slice_id * C * J; 
        #pragma unroll
        for(i = 0; i < A-nslices; i+=nslices) {
            dread_U_gsm2reg<B>(C, tx_, sU, rU);   // read U
            dgemm_slice<B, J>(MAGMA_D_ONE, sT, rU, MAGMA_D_ZERO, rV); // multiply
            dwrite_V_reg2gsm<J>(C, tx_, rV, sV ); // write V back
            sU += nslices * C * B;
            sV += nslices * C * J;
        }

        if(slice_id < A-i){
            dread_U_gsm2reg<B>(C, tx_, sU, rU);   // read U
            dgemm_slice<B, J>(MAGMA_D_ONE, sT, rU, MAGMA_D_ZERO, rV); // multiply
            dwrite_V_reg2gsm<J>(C, tx_, rV, sV ); // write V back
        }
        __syncthreads(); 
        
        
        #if 0 
        __syncthreads();
        if(tx == 0) {
            printf("GPU,dim = %d \n", d);
            for(int i = 0; i < pre * post; i++) {
                for(int j = 0; j < Q; j++) {
                    printf("%5.2f ", sV[j * (pre*post) + i]);
                }
                printf("\n");
            }
        }
        __syncthreads();
        #endif

        // adjust dimensions and re-calculate the thread indices 
        pre     /= P;
        post    *= Q;
        nslices  = nthreads / C;
        tx_      = tx % C; 
        slice_id = tx / C;
    }
    
    // handle last iteration (d = dim-1) with dV and beta
    // no need for sV in the last iteration, just use sU and write directly into dV
    sU = (d % 2 == 0) ? sTmp1 : sTmp2;
    //sV = (d % 2 == 0) ? sTmp2 : sTmp1; 
    double beta = (add == 1) ? MAGMA_D_ONE : MAGMA_D_ZERO; 
        
    sU += slice_id * C * B;
    dV += slice_id * C * J;
    #pragma unroll
    for(i = 0; i < A-nslices; i+=nslices) {
        dread_U_gsm2reg<B>(C, tx_, sU, rU);   // read U
        if( add ) {
            dread_V_gsm2reg<J>(C, tx_, dV, rV); 
        }
        dgemm_slice<B, J>(MAGMA_D_ONE, sT, rU, beta, rV); // multiply
        dwrite_V_reg2gsm<J>(C, tx_, rV, dV ); // write V back
        sU += nslices * C * B;
        dV += nslices * C * J;
    }

    if(slice_id < A-i){
        dread_U_gsm2reg<B>(C, tx_, sU, rU);   // read U
        if( add ) {
            dread_V_gsm2reg<J>(C, tx_, dV, rV); 
        }
        dgemm_slice<B, J>(MAGMA_D_ONE, sT, rU, beta, rV); // multiply
        dwrite_V_reg2gsm<J>(C, tx_, rV, dV ); // write V back
    }
    
    
    #if 0
    __syncthreads();
    if(tx == 0) {
        printf("GPU,dim = %d \n", d);
        for(int i = 0; i < pre * post; i++) {
            for(int j = 0; j < Q; j++) {
                printf("%5.2f ", dV[j * (pre*post) + i]);
            }
            printf("\n");
        }
    }
    __syncthreads();
    #endif

    
    pre     /= P;
    post    *= Q;
#undef B
#undef J
#undef A
#undef C
}

#endif    // MAGMA_BASIS_APPLY_INTERP_DEVICE_CUH
