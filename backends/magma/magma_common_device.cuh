// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#ifndef MAGMA_COMMON_DEVICE_CUH
#define MAGMA_COMMON_DEVICE_CUH

//////////////////////////////////////////////////////////////////////////////////////////
// init scalar to zero
template<typename T>
__device__ __inline__ T
make_zero()
{
    return 0;
}
//////////////////////////////////////////////////////////////////////////////////////////
// init scalar to zero -- specialization
template<>
__device__ __inline__ magmaFloatComplex
make_zero<magmaFloatComplex>()
{
    return MAGMA_C_ZERO;
}
//////////////////////////////////////////////////////////////////////////////////////////
// init scalar to zero -- specialization
template<>
__device__ __inline__ magmaDoubleComplex
make_zero<magmaDoubleComplex>()
{
    return MAGMA_Z_ZERO;
}

//////////////////////////////////////////////////////////////////////////////////////////
// init scalar to one
template<typename T>
__device__ __inline__ T
make_one()
{
    return 1;
}
//////////////////////////////////////////////////////////////////////////////////////////
// init scalar to zero -- specialization
template<>
__device__ __inline__ magmaFloatComplex
make_one<magmaFloatComplex>()
{
    return MAGMA_C_ONE;
}
//////////////////////////////////////////////////////////////////////////////////////////
// init scalar to zero -- specialization
template<>
__device__ __inline__ magmaDoubleComplex
make_one<magmaDoubleComplex>()
{
    return MAGMA_Z_ONE;
}

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
            for(int i = 0; i < J; i++) {
                sT[i * B + tx] = dT[i * B + tx];
            }
        }
    }
    else {
        // T is J x B
        if(tx < J) {
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
    for(int j = 0; j < J; j++) {
        rTmp = MAGMA_D_ZERO;
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


#endif // MAGMA_COMMON_DEVICE_CUH
