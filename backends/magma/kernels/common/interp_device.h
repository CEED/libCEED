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

#ifndef CEED_MAGMA_INTERP_DEVICE_H
#define CEED_MAGMA_INTERP_DEVICE_H

#define maxpq(p,q)    (p > q ? p : q)

// macros to abstract access of shared memory and reg. file
#define sT(i,j)          sT[(j) * P + (i)]
#define sTmp(i,j,ldw)    sTmp[(j)*(ldw) + (i)]
#define rU(idim,icomp,i) rU[(idim)*NCOMP*P + (icomp)*P + (i)]
#define rV(idim,icomp,i) rV[(idim)*NCOMP*Q + (icomp)*Q + (i)]

//////////////////////////////////////////////////////////////////////////////////////////
// interp basis action (1D)
template<typename T, int DIM, int NCOMP, int P, int Q>
static __device__ __inline__ void
magma_interp_1d_device( 
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
                rv += sU[icomp][i] * sT(i,tx); //sT[tx * P + i];	
            }
            sV[icomp][tx] = rv;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
// interp basis action (2D)
template<typename T, int DIMU, int DIMV, int NCOMP, int P, int Q, int rUsize, int rVsize>
static __device__ __inline__ void
magma_interp_2d_device( 
    const T *sT, magma_trans_t transT, 
    T rU[DIMU][NCOMP][rUsize] , T rV[DIMV][NCOMP][rVsize], 
    const int tx, T rTmp, T* swork)
{
    // Assumptions
    // 1. 1D threads of size max(P,Q)
    // 2. input:  rU[DIMU x NCOMP x rUsize] in registers (per thread)
    // 3. output: rV[DIMV x NCOMP x rVsize] in registers (per thread)
    // 4. Two products per component
    //  4.1 Batch P of (1xP) matrices times (PxQ) matrix => Batch P of (1xQ) matrices
    //  4.2 Batch 1 of (QxP) matrix   times (PxQ) matrix => (QxQ) matrix
    // 5. Each thread computes one row of the output of each product
    // 6. Sync is recommended before and after the call

    for(int icomp = 0; icomp < NCOMP; icomp++){
        // 1st product -- Batch P of (1xP) matrices [reg] x (PxQ) [shmem] => Batch P of (1xQ) matrices
        // the batch output P x (1xQ) is written on the fly to shmem
        if (tx < P) {
            const int batchid = tx;
            const int sld     = 1;
            T* sTmp = swork + batchid * (1 * Q);
            for(int j = 0; j < Q; j++){
                rTmp = make_zero<T>();
                for(int i = 0; i < P; i++){
                    rTmp += rU[0][icomp][i] * sT(i,j);
                }
                sTmp(0,j,sld) = rTmp;
            }
        }    // end of: if (tx < P)
        __syncthreads();

        // 2nd product -- Batch 1 of a (QxP) matrix [shmem] x (PxQ) [shmem] => (QxQ) matrix [reg]
        if (tx < Q) {
            const int batchid = 0;
            const int sld     = Q;
            T* sTmp = swork + batchid * (Q*P);
            for(int j = 0; j < Q; j++){
                rTmp = make_zero<T>();
                for(int i = 0; i < P; i++){
                    rTmp += sTmp(tx,i,sld) * sT(i,j);
                }
                rV[0][icomp][j] += rTmp;
            }
        }
        __syncthreads();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
// interp basis action (3D)
template<typename T, int DIMU, int DIMV, int NCOMP, int P, int Q, int rUsize, int rVsize>
static __device__ __inline__ void
magma_interp_3d_device( 
    const T *sT, magma_trans_t transT, 
    T rU[DIMU][NCOMP][rUsize] , T rV[DIMV][NCOMP][rVsize], 
    const int tx, T rTmp[Q], T* swork)
{
    // Assumptions
    // 1. 1D threads of size max(P,Q)^2
    // 2. input:  rU[DIMU x NCOMP x rUsize] in registers (per thread)
    // 3. output: rV[DIMV x NCOMP x rVsize] in registers (per thread)
    // 4. Three products per component
    //  4.1 Batch P^2 of (1xP) matrices times (PxQ) matrix => Batch P^2 of (1xQ) matrices
    //  4.2 Batch P   of (QxP) matrices times (PxQ) matrix => Batch P   of (QxQ) matrices
    //  4.3 Batch 1   of (Q^2xP) matrix times (PxQ) matrix => (Q^2xQ) matrix
    // 5. Each thread computes one row of the output of each product
    // 6. Sync is recommended before and after the call

    for(int icomp = 0; icomp < NCOMP; icomp++){
        // Batch P^2 of (1xP) matrices [reg] times (PxQ) matrix [shmem] => Batch P^2 of (1xQ) matrices [shmem]
        if (tx < (P*P)) {
            const int batchid = tx;
            const int sld     = 1;
            T* sTmp = swork + batchid * (1*Q);
            for(int j = 0; j < Q; j++){
                rTmp[0] = make_zero<T>();
                for(int i = 0; i < P; i++){
                    rTmp[0] += rU[0][icomp][i] * sT(i,j);
                }
                sTmp(0,j,sld) = rTmp[0];
            }
        }    // end of: if (tx < P*P)
        __syncthreads();

        // Batch P of (QxP) matrices [shmem] times (PxQ) matrix [shmem] => Batch P of (QxQ) matrices [reg]
        if (tx < (P*Q)) {
            const int batchid = tx / Q;
            const int tx_     = tx % Q;
            const int sld     = Q;
            T* sTmp = swork + batchid * (Q*P); // sTmp is input
            for(int j = 0; j < Q; j++){
                rTmp[j] = make_zero<T>();
                for(int i = 0; i < P; i++){
                    rTmp[j] += sTmp(tx_,i,sld) * sT(i,j);
                }
            }
        }
        __syncthreads();

        // write rTmp[] into shmem as batch P of QxQ matrices
        if (tx < (P*Q)){
            const int batchid = tx / Q;
            const int tx_     = tx % Q;
            const int sld     = Q;
            T* sTmp = swork + batchid * (Q*Q);
            for(int j = 0; j < Q; j++){
                sTmp(tx_, j, sld) = rTmp[j];
            }
        }
        __syncthreads();

       // Batch 1 of (Q^2xP) matrices [shmem] times (PxQ) matrix [shmem] => Batch 1 of (Q^2xQ) matrices [reg]
       if (tx < (Q*Q)) {
           // No need to declare batchid = (tx  / Q^2) = always zero
           // No need to declare tx_     = (tx_ % Q^2) = always tx
           const int sld     = Q*Q;
           T* sTmp = swork;
           for(int j = 0; j < Q; j++) {
               rTmp[0] = make_zero<T>();
               for(int i = 0; i < P; i++) {
                   rTmp[0] += sTmp(tx,i,sld) * sT(i,j);
               }
               rV[0][icomp][j] += rTmp[0];
           }
       }
       __syncthreads();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
// interp basis action -- dim and ncomp are run-time variables
template<int P, int Q>
static __device__ __inline__ void
magma_interp_generic_device( 
    const int dim, const int ncomp, const int pre_org, const int post_org, const int tmp_size, 
    const CeedScalar *dT, magma_trans_t transT,
    const CeedScalar *dU, CeedScalar *dV, 
    CeedScalar* shared_data )
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
    
    CeedScalar* sT    = (CeedScalar*)shared_data;
    CeedScalar* sTmp1 = sT + B * J;
    CeedScalar* sTmp2 = sTmp1 + tmp_size; 
    CeedScalar rU[P]  = { MAGMA_D_ZERO };    // each thread has an entire row of U
    CeedScalar rV[Q]  = { MAGMA_D_ZERO };    // each thread computes an entire row of V
    CeedScalar *sU, *sV; 

    // read T in shared memory
    dread_T_gm2sm<B, J>(tx, transT, dT, sT );    

    sU = sTmp1; 
    sV = sTmp2;

    // read U in sTmp1 (AC x B)
    sU += slice_id * C * B;
    dU += slice_id * C * B;
    for(i = 0; i < A-nslices; i+=nslices) {
        for(int b = 0; b < B; b++) {
            sU[b * C + tx_] = dU[b * C + tx_];
        }
        dU += nslices * C * B;
        sU += nslices * C * B;
    }
    
    if (slice_id < A-i) {
        for(int b = 0; b < B; b++) {
            //printf("tx = %d, tx_ = %d, accessing b * C + tx_ = %d\n", tx, tx_, b * C + tx_);
            sU[b * C + tx_] = dU[b * C + tx_];
        }
    }
    __syncthreads();

    int d = 0; 
    for(d = 0; d < dim-1; d++) {
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
        
        
        #if 0 
        __syncthreads();
        if (tx == 0) {
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
    CeedScalar beta = (add == 1) ? MAGMA_D_ONE : MAGMA_D_ZERO; 
        
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
    
    
    #if 0
    __syncthreads();
    if (tx == 0) {
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

#endif    // CEED_MAGMA_INTERP_DEVICE_H
