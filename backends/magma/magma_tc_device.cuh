#ifndef MAGMA_TC_DEVICE_CUH
#define MAGMA_TC_DEVICE_CUH

//////////////////////////////////////////////////////////////////////////////////////////
// reads T into shared memory
// must sync after call
template<int B, int J>
__device__ __inline__ void
dread_T_gm2sm(
        const int tx, const magma_trans_t transT, 
        const double* dT, double *sT ) 
{
    if( transT == MagmaNoTrans ) {
        // T is B x J
        if(tx < B) {
            #pragma unroll
            for(int i = 0; i < J; i++) {
                sT[i * B + tx] = dT[i * B + tx];
            }
        }
    }
    else {
        // T is J x B
        if(tx < J) {
            #pragma unroll
            for(int i = 0; i < B; i++) {
                sT[tx * B + i] = dT[i * J + tx];
            }
        }
    }
    // must sync after call
}

//////////////////////////////////////////////////////////////////////////////////////////
// reads a slice of U from shared/global memory into registers
// the correct pointer U must be precomputed
template<int B>
__device__ __inline__ void
dread_U_gsm2reg( 
        const int C, const int tx_, 
        const double* U, double rU[B] ) 
{
    #pragma unroll
    for(int i = 0; i < B; i++){
        rU[i] = U[i * C + tx_];
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
// reads a slice of V from shared/global memory into registers with scaling
// the correct pointer V must be precomputed
template<int J>
__device__ __inline__ void
dread_V_gsm2reg( 
        const int C, const int tx_, const double* V, double rV[J] ) 
{
    #pragma unroll
    for(int i = 0; i < J; i++){
        rV[i] = V[i * C + tx_];
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
// writes a slice of V from reg to shared/global memory
// the correct pointer V must be precomputed
template<int J>
__device__ __noinline__ void
dwrite_V_reg2gsm( 
        const int C, const int tx_, 
        double rV[J], double* V ) 
{
    #pragma unroll
    for(int i = 0; i < J; i++){
        V[i * C + tx_] = rV[i];
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
// multiply a slice of U times T to produce a slice of V
template<int B, int J>
__device__ __inline__ void
dgemm_slice( 
        double alpha, double *sT, 
        double rU[B], double beta, double rV[J] ) 
{
    double rTmp;
    #pragma unroll
    for(int j = 0; j < J; j++) {
        rTmp = MAGMA_D_ZERO;
        #pragma unroll
        for(int b = 0; b < B; b++){
            rTmp += rU[ b ] * sT[ j * B + b ];
        }
        rV[ j ] *= beta; 
        rV[ j ] += alpha * rTmp;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
template<int B, int J>
__device__  __inline__ void
dgemm_ceed_device( const int tx, const int A, const int C, magma_trans_t transT, double *sT, 
                   const double alpha, const double beta,
                   const double *dU,   double *dV, 
                         double rU[B], double rV[J])
{
    const int tx_      = tx % C;
    const int slice_id = tx / C;

    // advance pointers for U and V
    dU += slice_id * C * B;
    dV += slice_id * C * J;

    // read V if beta is non-zero  
    if( beta != MAGMA_D_ZERO ) {
        dread_V_gsm2reg<J>(C, tx_, (const double*)dV, rV); 
    }

    // read U
    dread_U_gsm2reg<B>(C, tx_, dU, rU);

    // multiply
    dgemm_slice<B, J>(alpha, sT, rU, beta, rV);

    // write V back
    dwrite_V_reg2gsm<J>(C, tx_, rV, dV );
} 


#endif // MAGMA_TC_DEVICE_CUH
