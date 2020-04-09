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

#include "ceed-magma.h"

//////////////////////////////////////////////////////////////////////////////////////////
#ifdef __cplusplus
CEED_INTERN "C"
#endif
magma_int_t 
magma_grad( 
    magma_int_t P, magma_int_t Q, magma_int_t dim, magma_int_t ncomp,  
    const CeedScalar *dinterp1d, const CeedScalar *dgrad1d, CeedTransposeMode tmode, 
    const CeedScalar *dU, magma_int_t u_elstride, magma_int_t u_compstride, magma_int_t u_dimstride, 
          CeedScalar *dV, magma_int_t v_elstride, magma_int_t v_compstride, magma_int_t v_dimstride, 
    magma_int_t nelem, magma_kernel_mode_t kernel_mode, magma_queue_t queue)
{
    magma_int_t launch_failed = 0;

    if(false && (kernel_mode == MAGMA_KERNEL_DIM_SPECIFIC)) {
        /*switch(dim) {
            case 1: magma_interp_1d(P, Q, ncomp, dT, tmode, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
            case 2: magma_interp_2d(P, Q, ncomp, dT, tmode, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
            case 3: magma_interp_3d(P, Q, ncomp, dT, tmode, dU, u_elstride, u_compstride, dV, v_elstride, v_compstride, nelem, queue); break;
            default: printf("Launch failed at %s\n", __func__);
        }*/
    }
    else {
        launch_failed = magma_grad_generic(
                P, Q, dim, ncomp, 
                dinterp1d, dgrad1d, tmode, 
                dU, u_elstride, u_compstride, u_dimstride, 
                dV, v_elstride, v_compstride, v_dimstride, 
                nelem, queue ); 
    }

    return launch_failed;
}
