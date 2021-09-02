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

#ifndef CEED_MAGMA_COMMON_DEVICE_H
#define CEED_MAGMA_COMMON_DEVICE_H

#ifdef HAVE_HIP
#define MAGMA_DEVICE_SHARED(type, name) HIP_DYNAMIC_SHARED(type, name)
#else 
#define MAGMA_DEVICE_SHARED(type, name) extern __shared__ type name[];
#endif

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
// read U or V of a 1D element into shared memory sU[][] or sV[][] --  for all components
// the devptr is assumed to point directly to the element
// must sync after call
template<typename T, int LENGTH, int NCOMP>
__device__ __inline__ void 
read_1d(const T* devptr, const int compstride, T* sBuffer[NCOMP], const int tx)
{
    if (tx < LENGTH) {
        for(int icomp = 0; icomp < NCOMP; icomp++) {
            sBuffer[icomp][tx] = devptr[icomp * compstride + tx];
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
// write V of a 1D element into global memory from sV[][] --  for all components
// the devptr is assumed to point directly to the element
template<typename T, int LENGTH, int NCOMP>
__device__ __inline__ void 
write_1d(T* sBuffer[NCOMP], T* devptr, const int compstride, const int tx)
{
    if (tx < LENGTH) {
        for(int icomp = 0; icomp < NCOMP; icomp++) {
            devptr[icomp * compstride + tx] = sBuffer[icomp][tx];
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
// read U of a 2D element into registers rU[][][] --  for all components of a single dim
// dU is assumed to be offset by elem-stride and dim-stride
// register is assumed to be rU[DIMU][NCOMP][rUsize]
// iDIM specifies which dimension is being read into in rU
// rUsize can be different from P (e.g. MAXPQ)
// sTmp is a shared memory workspace of size P^2
template<typename T, int P, int DIMU, int NCOMP, int rUsize, int iDIM>
__device__ __inline__ void 
readU_2d(const T* dU, const int compstride, T rU[DIMU][NCOMP][rUsize], T* sTmp, const int tx)
{
    // read U as a batch P of (1xP) vectors
    // vec 0  : [u0, u1, u2, ... u_(P-1)] -- contiguous in memory
    // vec 1  : [u0, u1, u2, ... u_(P-1)] -- contiguous in memory
    // ... 
    // vec P-1: [u0, u1, u2, ... u_(P-1)] -- contiguous in memory
    // threads collaboratively read vec0 and then vec1 and so on
    // but for the kernel, we want
    // thread 0 to hold all of vec0 in registers, and
    // thread 1 to hold all of vec1 in registers, and and so on
    // so we need to transpose
    for(int icomp = 0; icomp < NCOMP; icomp++) {
        // read from global memory into shared memory
        if (tx < P) {
            for(int i = 0; i < P; i++) {
                sTmp[i*P + tx] = dU[icomp * compstride + i*P + tx];
            }
        }
        __syncthreads();

        if (tx < P) {
            for(int i = 0; i < P; i++) {
                rU[iDIM][icomp][i] = sTmp[tx*P + i];
            }
        }
        __syncthreads();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
// read V of a 2D element into registers rV[][][] --  for all components of a single dim
// dV is assumed to be offset by elem-stride and dim-stride
// register is assumed to be rV[DIMV][NCOMP][rVsize]
// iDIM specifies which dimension is being read into in rV
// rVsize can be different from P (e.g. MAXPQ)
template<typename T, int Q, int DIMV, int NCOMP, int rVsize, int iDIM>
__device__ __inline__ void 
readV_2d(const T* dV, const int compstride, T rV[DIMV][NCOMP][rVsize], const int tx)
{
    if (tx < Q) {
        for(int icomp = 0; icomp < NCOMP; icomp++) {
            for(int j = 0; j < Q; j++) {
                rV[iDIM][icomp][j] = dV[icomp * compstride + j*Q + tx];
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
// write V of a 2D element from registers rV[][][] to global memory --  for all components of a single dim
// dV is assumed to be offset by elem-stride and dim-stride
// register is assumed to be rV[DIMV][NCOMP][rVsize]
// iDIM specifies which dimension is being read from in rV
// idim specifies which dimension is being written to in dV
// rVsize can be different from P (e.g. MAXPQ)
template<typename T, int Q, int DIMV, int NCOMP, int rVsize, int iDIM>
__device__ __inline__ void 
writeV_2d(T* dV, const int compstride, T rV[DIMV][NCOMP][rVsize], const int tx)
{
    if (tx < Q) {
        for(int icomp = 0; icomp < NCOMP; icomp++) {
            for(int j = 0; j < Q; j++) {
                dV[icomp * compstride + j*Q + tx] = rV[iDIM][icomp][j];
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
// read U of a 3D element into registers rU[][][] --  for all components of a single dim
// dU is assumed to be offset by elem-stride and dim-stride
// register is assumed to be rU[DIMU][NCOMP][rUsize]
// iDIM specifies which dimension is being read into in rU
// rUsize can be different from P (e.g. MAXPQ)
// sTmp is a shared memory workspace of size P^3
template<typename T, int P, int DIMU, int NCOMP, int rUsize, int iDIM>
__device__ __inline__ void 
readU_3d(const T* dU, const int compstride, T rU[DIMU][NCOMP][rUsize], T* sTmp, const int tx)
{
    // read U as a batch P^2 of (1xP) vectors
    // vec 0    : [u0, u1, u2, ... u_(P-1)] -- contiguous in memory
    // vec 1    : [u0, u1, u2, ... u_(P-1)] -- contiguous in memory
    // ... 
    // vec P^2-1: [u0, u1, u2, ... u_(P-1)] -- contiguous in memory
    // threads collaboratively read vec0 and then vec1 and so on
    // but for the kernel, we want
    // thread 0 to hold all of vec0 in registers, and
    // thread 1 to hold all of vec1 in registers, and and so on
    // so we need to transpose
    for(int icomp = 0; icomp < NCOMP; icomp++) {
        // read from global memory into shared memory
        if (tx < P*P) {
            for(int i = 0; i < P; i++) {
                sTmp[i*P*P + tx] = dU[icomp * compstride + i*P*P + tx];
            }
        }
        __syncthreads();

        if (tx < P*P) {
            for(int i = 0; i < P; i++) {
                rU[iDIM][icomp][i] = sTmp[tx*P + i];
            }
        }
        __syncthreads();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
// read V of a 3D element into registers rV[][][] --  for all components of a single dim
// dV is assumed to be offset by elem-stride and dim-stride
// register is assumed to be rV[DIMV][NCOMP][rVsize]
// iDIM specifies which dimension is being read into in rV
// rVsize can be different from P (e.g. MAXPQ)
template<typename T, int Q, int DIMV, int NCOMP, int rVsize, int iDIM>
__device__ __inline__ void 
readV_3d(const T* dV, const int compstride, T rV[DIMV][NCOMP][rVsize], const int tx)
{
    if (tx < Q*Q) {
        for(int icomp = 0; icomp < NCOMP; icomp++) {
            for(int j = 0; j < Q; j++) {
                rV[iDIM][icomp][j] = dV[icomp * compstride + j*(Q*Q) + tx];
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
// write V of a 3D element from registers rV[][][] to global memory --  for all components of a single dim
// dV is assumed to point directly to the element (i.e. already offset by elem-stride)
// register is assumed to be rV[DIMV][NCOMP][rVsize]
// iDIM specifies which dimension is being read from in rV
// idim specifies which dimension is being written to in dV
// rVsize can be different from P (e.g. MAXPQ)
template<typename T, int Q, int DIMV, int NCOMP, int rVsize, int iDIM>
__device__ __inline__ void 
writeV_3d(T* dV, const int compstride, T rV[DIMV][NCOMP][rVsize], const int tx)
{
    if (tx < (Q*Q)) {
        for(int icomp = 0; icomp < NCOMP; icomp++) {
            for(int j = 0; j < Q; j++) {
                dV[icomp * compstride + j*(Q*Q) + tx] = rV[iDIM][icomp][j];
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
// reads T into shared memory
// must sync after call
template<int B, int J>
__device__ __inline__ void
dread_T_gm2sm(
        const int tx, const magma_trans_t transT, 
        const CeedScalar* dT, CeedScalar *sT ) 
{
    if ( transT == MagmaNoTrans ) {
        // T is B x J
        if (tx < B) {
            for(int i = 0; i < J; i++) {
                sT[i * B + tx] = dT[i * B + tx];
            }
        }
    }
    else {
        // T is J x B
        if (tx < J) {
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
        const CeedScalar* U, CeedScalar rU[B] ) 
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
        const int C, const int tx_, const CeedScalar* V, CeedScalar rV[J] ) 
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
        CeedScalar rV[J], CeedScalar* V ) 
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
        CeedScalar alpha, CeedScalar *sT, 
        CeedScalar rU[B], CeedScalar beta, CeedScalar rV[J] ) 
{
    CeedScalar rTmp;
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
dgemm_ceed_device( const int tx, const int A, const int C, magma_trans_t transT, CeedScalar *sT, 
                   const CeedScalar alpha, const CeedScalar beta,
                   const CeedScalar *dU,   CeedScalar *dV, 
                         CeedScalar rU[B], CeedScalar rV[J])
{
    const int tx_      = tx % C;
    const int slice_id = tx / C;

    // advance pointers for U and V
    dU += slice_id * C * B;
    dV += slice_id * C * J;

    // read V if beta is non-zero  
    if ( beta != MAGMA_D_ZERO ) {
        dread_V_gsm2reg<J>(C, tx_, (const CeedScalar*)dV, rV); 
    }

    // read U
    dread_U_gsm2reg<B>(C, tx_, dU, rU);

    // multiply
    dgemm_slice<B, J>(alpha, sT, rU, beta, rV);

    // write V back
    dwrite_V_reg2gsm<J>(C, tx_, rV, dV );
} 

#endif // CEED_MAGMA_COMMON_DEVICE_H
