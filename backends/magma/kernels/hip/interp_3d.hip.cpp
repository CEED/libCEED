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

#include "hip/hip_runtime.h"
#include "../common/interp.h"

//////////////////////////////////////////////////////////////////////////////////////////
template<typename T, typename TC, int NCOMP, int P, int Q>
static magma_int_t 
magma_interp_3d_kernel_driver(  
                const TC *dT, magma_trans_t transT,
                const T *dU, magma_int_t estrdU, magma_int_t cstrdU, 
                      T *dV, magma_int_t estrdV, magma_int_t cstrdV, 
                magma_int_t nelem, magma_queue_t queue)
{
    magma_device_t device;
    magma_getdevice( &device );
    magma_int_t shmem_max, nthreads_max;
    const int MAXPQ = maxpq(P,Q);

    magma_int_t nthreads = MAXPQ*MAXPQ;
    magma_int_t ntcol = MAGMA_BASIS_NTCOL(nthreads, MAGMA_MAXTHREADS_3D);
    magma_int_t shmem  = 0;
    shmem += sizeof(TC)* (P*Q);  // for sT
    shmem += sizeof(TC)* ntcol * (max(P*P*MAXPQ, P*Q*Q));  // rU needs P^2xP, the intermediate output needs max(P^2xQ,PQ^2)    

    hipDeviceGetAttribute (&nthreads_max, hipDeviceAttributeMaxThreadsPerBlock, device);
    hipDeviceGetAttribute (&shmem_max, hipDeviceAttributeMaxSharedMemoryPerBlock, device);

    if ( (nthreads*ntcol) > nthreads_max || shmem > shmem_max ) {
        return 1;    // launch failed
    }
    else { 
        magma_int_t nblocks = (nelem + ntcol-1) / ntcol;
        dim3 threads(nthreads, ntcol, 1);
        dim3 grid(nblocks, 1, 1);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(magma_interp_3d_kernel<T,TC,NCOMP,P,Q,MAXPQ>), dim3(grid), dim3(threads), shmem, magma_queue_get_hip_stream(queue), dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem);
        return (hipPeekAtLastError() == hipSuccess) ? 0 : 1;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
template<int P, int Q>
static magma_int_t 
magma_interp_3d_ncomp(
                magma_int_t ncomp,
                const double *dT, magma_trans_t transT,
                const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU, 
                      CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV, 
                magma_int_t nelem, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch (ncomp) {
        case 1: 
          launch_failed = magma_interp_3d_kernel_driver<CeedScalar,double,1,P,Q>
          (dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case 2: 
          launch_failed = magma_interp_3d_kernel_driver<CeedScalar,double,2,P,Q>
          (dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case 3: 
          launch_failed = magma_interp_3d_kernel_driver<CeedScalar,double,3,P,Q>
          (dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        default: launch_failed = 1;
    }
    return launch_failed;
}

//////////////////////////////////////////////////////////////////////////////////////////
template<int P>
static magma_int_t 
magma_interp_1d_ncomp_q(
                magma_int_t Q, magma_int_t ncomp,
                const double *dT, magma_trans_t transT,
                const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU, 
                      CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV, 
                magma_int_t nelem, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch (Q) {
        case  1: 
          launch_failed = magma_interp_3d_ncomp<P, 1>
          (ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  2: 
          launch_failed = magma_interp_3d_ncomp<P, 2>
          (ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  3: 
          launch_failed = magma_interp_3d_ncomp<P, 3>
          (ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  4: 
          launch_failed = magma_interp_3d_ncomp<P, 4>
          (ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  5: 
          launch_failed = magma_interp_3d_ncomp<P, 5>
          (ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  6: 
          launch_failed = magma_interp_3d_ncomp<P, 6>
          (ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  7: 
          launch_failed = magma_interp_3d_ncomp<P, 7>
          (ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  8: 
          launch_failed = magma_interp_3d_ncomp<P, 8>
          (ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  9: 
          launch_failed = magma_interp_3d_ncomp<P, 9>
          (ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case 10: 
          launch_failed = magma_interp_3d_ncomp<P,10>
          (ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        default: launch_failed = 1;
    }
    return launch_failed;
}

//////////////////////////////////////////////////////////////////////////////////////////
static magma_int_t 
magma_interp_3d_ncomp_q_p(
                magma_int_t P, magma_int_t Q, magma_int_t ncomp,
                const double *dT, magma_trans_t transT,
                const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU, 
                      CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV, 
                magma_int_t nelem, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;
    switch (P) {
        case  1: 
          launch_failed = magma_interp_1d_ncomp_q< 1>
          (Q, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  2: 
          launch_failed = magma_interp_1d_ncomp_q< 2>
          (Q, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  3: 
          launch_failed = magma_interp_1d_ncomp_q< 3>
          (Q, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  4: 
          launch_failed = magma_interp_1d_ncomp_q< 4>
          (Q, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  5: 
          launch_failed = magma_interp_1d_ncomp_q< 5>
          (Q, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  6: 
          launch_failed = magma_interp_1d_ncomp_q< 6>
          (Q, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  7: 
          launch_failed = magma_interp_1d_ncomp_q< 7>
          (Q, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  8: 
          launch_failed = magma_interp_1d_ncomp_q< 8>
          (Q, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case  9: 
          launch_failed = magma_interp_1d_ncomp_q< 9>
          (Q, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        case 10: 
          launch_failed = magma_interp_1d_ncomp_q<10>
          (Q, ncomp, dT, transT, dU, estrdU, cstrdU, dV, estrdV, cstrdV, nelem, queue); 
          break;
        default: launch_failed = 1;
    }
    return launch_failed;
}


//////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t 
magma_interp_3d( 
    magma_int_t P, magma_int_t Q, magma_int_t ncomp,  
    const double *dT, CeedTransposeMode tmode,
    const CeedScalar *dU, magma_int_t estrdU, magma_int_t cstrdU, 
          CeedScalar *dV, magma_int_t estrdV, magma_int_t cstrdV, 
    magma_int_t nelem, magma_queue_t queue)
{    
    magma_int_t launch_failed = 0;
    magma_trans_t transT = (tmode == CEED_NOTRANSPOSE) ? MagmaNoTrans : MagmaTrans;
    launch_failed = magma_interp_3d_ncomp_q_p(
                        P, Q, ncomp, 
                        dT, transT, 
                        dU, estrdU, cstrdU, 
                        dV, estrdV, cstrdV, 
                        nelem, queue);

    return launch_failed;
}
