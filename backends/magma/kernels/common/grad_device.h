// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_MAGMA_GRAD_DEVICE_H
#define CEED_MAGMA_GRAD_DEVICE_H

#define maxpq(p,q)    (p > q ? p : q)

// macros to abstract access of shared memory and reg. file
#define sT(i,j)           sT[(j) * P + (i)]
#define sTmp(i,j,ldw)     sTmp[(j)*(ldw) + (i)]
#define sTmp2(i,j,ldw)    sTmp2[(j)*(ldw) + (i)]
#define rU(idim,icomp,i) rU[(idim)*NCOMP*P + (icomp)*P + (i)]
#define rV(idim,icomp,i) rV[(idim)*NCOMP*Q + (icomp)*Q + (i)]

//////////////////////////////////////////////////////////////////////////////////////////
// grad basis action (1D)
template<typename T, int DIM, int NCOMP, int P, int Q>
static __device__ __inline__ void
magma_grad_1d_device( 
    const T *sT, magma_trans_t transT, 
    T* sU[NCOMP], T* sV[NCOMP], const int tx)
{
    // Assumptions
    // 1. 1D threads of size max(P,Q)
    // 2. sU[i] is 1xP: in shared memory
    // 3. sV[i] is 1xQ: in shared memory
    // 4. Product per component is one row (1xP) times T matrix (PxQ) => one row (1xQ)
    // 5. Each thread computes one entry in sV[i]
    // 6. Must sync before and after call
    // 7. Note that the layout for U and V is different from 2D/3D problem

    T rv;
    if (tx < Q) {
        for(int icomp = 0; icomp < NCOMP; icomp++) {
            rv = (transT == MagmaTrans) ? sV[icomp][tx] : make_zero<T>();
            for(int i = 0; i < P; i++) {
                rv += sU[icomp][i] * sT(i,tx);	
            }
            sV[icomp][tx] = rv;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
// grad basis action (2D)
// This function is called two times at a higher level for 2D
// DIMU  -- for the size of rU[DIMU * NCOMP * MAXPQ]
// DIMV  -- for the size of rV[DIMV * NCOMP * MAXPQ]
// iDIM  -- the index of the outermost loop over dimensions in grad 
// iDIMU -- which dim index of rU is accessed (always 0 for notrans, 0 or 1 for trans)
// iDIMV -- which dim index of rV is accessed (0 or 1 for notrans, always 0 for trans)
// the scalar beta is used to specify whether to accumulate to rV, or overwrite it
template<typename T, int DIMU, int DIMV, int NCOMP, int P, int Q, int rUsize, int rVsize, int iDIM, int iDIMU, int iDIMV>
static __device__ __inline__ void
magma_grad_2d_device( 
    const T *sTinterp, const T *sTgrad, 
    T rU[DIMU][NCOMP][rUsize] , T rV[DIMV][NCOMP][rVsize], 
    T beta, const int tx, T rTmp, T* swork)
{
    // Assumptions
    // 0. This device routine applies grad for one dim only (iDIM), so it should be called twice for 2D
    // 1. 1D threads of size max(P,Q)
    // 2. input:  rU[DIMU x NCOMP x P] in registers (per thread)
    // 3. output: rV[DIMV x NCOMP x Q] in registers (per thread)
    // 4. Two products per each (dim,component) pair
    //  4.1 Batch P of (1xP) matrices times (PxQ) matrix => Batch P of (1xQ) matrices
    //  4.2 Batch 1 of (QxP) matrix   times (PxQ) matrix => (QxQ) matrix
    // 6. Each thread computes one row of the output of each product
    // 7. Sync is recommended before and after the call

    for(int icomp = 0; icomp < NCOMP; icomp++){
        // 1st product -- Batch P of (1xP) matrices [reg] x (PxQ) [shmem] => Batch P of (1xQ) matrices
        // the batch output P x (1xQ) is written on the fly to shmem
        if (tx < P) {
            const int batchid = tx;
            const int sld     = 1;
            const T *sT = (iDIM == 0) ? sTgrad : sTinterp;
            T* sTmp = swork + batchid * (1 * Q);
            for(int j = 0; j < Q; j++){
                rTmp = make_zero<T>();
                for(int i = 0; i < P; i++){
                    rTmp += rU[iDIMU][icomp][i] * sT(i,j);
                }
                sTmp(0,j,sld) = rTmp;
            }
        }    // end of: if (tx < P)
        __syncthreads();

        // 2nd product -- Batch 1 of a (QxP) matrix [shmem] x (PxQ) [shmem] => (QxQ) matrix [reg]
        if (tx < Q) {
            const int batchid = 0;
            const int sld     = Q;
            const T *sT = (iDIM == 1) ? sTgrad : sTinterp;
            T* sTmp = swork + batchid * (Q*P);
            for(int j = 0; j < Q; j++){
                rTmp = make_zero<T>();
                for(int i = 0; i < P; i++){
                    rTmp += sTmp(tx,i,sld) * sT(i,j);
                }
                rV[iDIMV][icomp][j] *= beta;
                rV[iDIMV][icomp][j] += rTmp;
            }
        }
        __syncthreads();
    }  // loop over NCOMP
}

//////////////////////////////////////////////////////////////////////////////////////////
// grad basis action (3D)
// This function is called three times at a higher level for 3D
// DIMU  -- for the size of rU[DIMU * NCOMP * MAXPQ]
// DIMV  -- for the size of rV[DIMV * NCOMP * MAXPQ]
// iDIM  -- the index of the outermost loop over dimensions in grad 
// iDIMU -- which dim index of rU is accessed (always 0 for notrans, 0, 1, or 2 for trans)
// iDIMV -- which dim index of rV is accessed (0, 1, or 2 for notrans, always 0 for trans)
// the scalar beta is used to specify whether to accumulate to rV, or overwrite it
template<typename T, int DIMU, int DIMV, int NCOMP, int P, int Q, int rUsize, int rVsize, int iDIM, int iDIMU, int iDIMV>
static __device__ __inline__ void
magma_grad_3d_device( 
    const T *sTinterp, const T *sTgrad, 
    T rU[DIMU][NCOMP][rUsize] , T rV[DIMV][NCOMP][rVsize], 
    T beta, const int tx, T rTmp, T* swork)
{
    // Assumptions
    // 0. This device routine applies grad for one dim only (iDIM), so it should be thrice for 3D
    // 1. 1D threads of size max(P,Q)^2
    // 2. input:  rU[DIMU x NCOMP x rUsize] in registers (per thread)
    // 3. output: rV[DIMV x NCOMP x rVsize] in registers (per thread)
    // 4. Three products per each (dim,component) pair
    //  4.1 Batch P^2 of (1xP) matrices times (PxQ) matrix => Batch P^2 of (1xQ) matrices
    //  4.2 Batch P   of (QxP) matrices times (PxQ) matrix => Batch P   of (QxQ) matrices
    //  4.3 Batch 1   of (Q^2xP) matrix times (PxQ) matrix => (Q^2xQ) matrix
    // 6. Each thread computes one row of the output of each product
    // 7. Sync is recommended before and after the call

    T* sW1 = swork;
    T* sW2 = sW1 + P*P*Q;
    for(int icomp = 0; icomp < NCOMP; icomp++){
        // Batch P^2 of (1xP) matrices [reg] times (PxQ) matrix [shmem] => Batch P^2 of (1xQ) matrices [shmem]
        if (tx < (P*P)) {
            const int batchid = tx;
            const int sld     = 1;
            const T *sT = (iDIM == 0) ? sTgrad : sTinterp;
            T* sTmp = sW1 + batchid * (1*Q);
            for(int j = 0; j < Q; j++){
                rTmp = make_zero<T>();
                for(int i = 0; i < P; i++){
                    //rTmp += rU(iDIMU,icomp,i) * sT(i,j);
                    rTmp += rU[iDIMU][icomp][i] * sT(i,j);
                }
                sTmp(0,j,sld) = rTmp;
            }
        }    // end of: if (tx < P*P)
        __syncthreads();

        // Batch P of (QxP) matrices [shmem] times (PxQ) matrix [shmem] => Batch P of (QxQ) matrices [reg]
        if (tx < (P*Q)) {
            const int batchid = tx / Q;
            const int tx_     = tx % Q;
            const int sld     = Q;
            const T *sT = (iDIM == 1) ? sTgrad : sTinterp;
            T* sTmp  = sW1 + batchid * (Q*P); // sTmp is input
            T* sTmp2 = sW2 + batchid * (Q*Q); // sTmp2 is output
            for(int j = 0; j < Q; j++){
                rTmp = make_zero<T>();
                for(int i = 0; i < P; i++){
                    rTmp += sTmp(tx_,i,sld) * sT(i,j);
                }
                sTmp2(tx_,j,sld) = rTmp;
            }
        }
        __syncthreads();

       // Batch 1 of (Q^2xP) matrices [shmem] times (PxQ) matrix [shmem] => Batch 1 of (Q^2xQ) matrices [reg]
       if (tx < (Q*Q)) {
           // No need to declare batchid = (tx  / Q^2) = always zero
           // No need to declare tx_     = (tx_ % Q^2) = always tx
           const int sld = Q*Q;
           const T *sT   = (iDIM == 2) ? sTgrad : sTinterp;
           T* sTmp = sW2;  // sTmp is input
           for(int j = 0; j < Q; j++) {
               rTmp = make_zero<T>();
               for(int i = 0; i < P; i++) {
                   rTmp += sTmp(tx,i,sld) * sT(i,j);
               }
               //rV(iDIMV,icomp,j) *= beta;
               //rV(iDIMV,icomp,j) += rTmp;
               rV[iDIMV][icomp][j] *= beta;
               rV[iDIMV][icomp][j] += rTmp;
           }
       }
       __syncthreads();
    }  // loop over NCOMP
}

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, int P, int Q>
static __device__ __inline__ void
magma_grad_generic_device( 
    const int p, 
    const int dim, const int ncomp, const int pre_org, const int post_org, const int tmp_size, 
    const T *sTinterp, const T *sTgrad, magma_trans_t transT,
    const T *dU, T *dV, 
    T* shared_data )
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
    
    T* sTmp1 = (T*)shared_data;
    T* sTmp2 = sTmp1 + tmp_size; 
    T rU[P]  = { MAGMA_D_ZERO };    // each thread has an entire row of U
    T rV[Q]  = { MAGMA_D_ZERO };    // each thread computes an entire row of V
    T *sU, *sV, *sT; 

    sU = sTmp1; 
    sV = sTmp2;

    // read U in sTmp1 (AC x B)
    sU += slice_id * C * B;
    dU += slice_id * C * B;
    for(i = 0; i < A-nslices; i+=nslices) {
        #pragma unroll
        for(int b = 0; b < B; b++) {
            sU[b * C + tx_] = dU[b * C + tx_];
        }
        dU += nslices * C * B;
        sU += nslices * C * B;
    }
    
    if (slice_id < A-i) {
        #pragma unroll
        for(int b = 0; b < B; b++) {
            //printf("tx = %d, tx_ = %d, accessing b * C + tx_ = %d\n", tx, tx_, b * C + tx_);
            sU[b * C + tx_] = dU[b * C + tx_];
        }
    }
    __syncthreads();
    
    int d = 0; 
    for(d = 0; d < dim-1; d++) {
        sT = (p == d) ? (T*)sTgrad : (T*)sTinterp;
        sU = (d % 2 == 0) ? sTmp1 : sTmp2;
        sV = (d % 2 == 0) ? sTmp2 : sTmp1;
        
        sU += slice_id * C * B;
        sV += slice_id * C * J; 
        for(i = 0; i < A-nslices; i+=nslices) {
            dread_U_gsm2reg<B>(C, tx_, sU, rU);   // read U
            dgemm_slice<B, J>(MAGMA_D_ONE, sT, rU, MAGMA_D_ZERO, rV); // multiply
            dwrite_V_reg2gsm<J>(C, tx_, rV, sV ); // write V back
            sU += nslices * C * B;
            sV += nslices * C * J;
        }

        if (slice_id < A-i){
            dread_U_gsm2reg<B>(C, tx_, sU, rU);   // read U
            dgemm_slice<B, J>(MAGMA_D_ONE, sT, rU, MAGMA_D_ZERO, rV); // multiply
            dwrite_V_reg2gsm<J>(C, tx_, rV, sV ); // write V back
        }
        __syncthreads(); 
        
        // adjust dimensions and re-calculate the thread indices 
        pre     /= P;
        post    *= Q;
        nslices  = nthreads / C;
        tx_      = tx % C; 
        slice_id = tx / C;
    }
    
    // handle last iteration (d = dim-1) with dV and beta
    // no need for sV in the last iteration, just use sU and write directly into dV
    sT = (p == d) ? (T*)sTgrad : (T*)sTinterp;
    sU = (d % 2 == 0) ? sTmp1 : sTmp2;
    //sV = (d % 2 == 0) ? sTmp2 : sTmp1; 
    T beta = (add == 1) ? MAGMA_D_ONE : MAGMA_D_ZERO; 
        
    sU += slice_id * C * B;
    dV += slice_id * C * J;
    for(i = 0; i < A-nslices; i+=nslices) {
        dread_U_gsm2reg<B>(C, tx_, sU, rU);   // read U
        if ( add ) {
            dread_V_gsm2reg<J>(C, tx_, dV, rV); 
        }
        dgemm_slice<B, J>(MAGMA_D_ONE, sT, rU, beta, rV); // multiply
        dwrite_V_reg2gsm<J>(C, tx_, rV, dV ); // write V back
        sU += nslices * C * B;
        dV += nslices * C * J;
    }

    if (slice_id < A-i){
        dread_U_gsm2reg<B>(C, tx_, sU, rU);   // read U
        if ( add ) {
            dread_V_gsm2reg<J>(C, tx_, dV, rV); 
        }
        dgemm_slice<B, J>(MAGMA_D_ONE, sT, rU, beta, rV); // multiply
        dwrite_V_reg2gsm<J>(C, tx_, rV, dV ); // write V back
    }

    pre     /= P;
    post    *= Q;
#undef B
#undef J
#undef A
#undef C
}

#endif    // CEED_MAGMA_GRAD_DEVICE_H
