// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <arm_sve.h>
#include <arm_sme.h>
#include <stdint.h>

#include "ceed-sme.h"

#ifdef CEED_SCALAR_IS_FP64
#define rtype svfloat64_t
#define vlength() ((CeedSize)svcntd())
#define whilelt(i, n) svwhilelt_b64((int64_t)(i), (int64_t)(n))
#define ptrue() svptrue_b64()
#define cntp(g, pg) svcntp_b64(g, pg)
#define load_vec(pg, src) svld1_f64(pg, src)
#define load_za_row(tile, row, pg_row, src) svld1_hor_za64(tile, row, pg_row, src)
#define store_za_row(tile, row, pg_row, dst) svst1_hor_za64(tile, row, pg_row, dst)
#define fmopa(tile, pg_col, pg_row, src_col, src_row) svmopa_za64_f64_m(tile, pg_col, pg_row, src_col, src_row)
#else

#endif


//------------------------------------------------------------------------------
// Tensor Contract Slice
//------------------------------------------------------------------------------
// v[j,c] (+)= sum_b t[j,b] u[b,c] for one a; vectorized over c, tiled over j.
__arm_new("za") static inline int CeedTensorContract_Sme_Slice(CeedInt B, CeedInt C, CeedInt J, const CeedScalar *restrict t, CeedTransposeMode t_mode,
                                               const CeedInt add, const CeedScalar *restrict u, CeedScalar *restrict v) __arm_streaming {
  
  CeedInt s0 = B, s1 = 1;

  if(t_mode == CEED_TRANSPOSE){
    s0 = 1;
    s1 = J;
  }

  svbool_t pg_col;
  for(CeedSize j = 0;
      svptest_first(ptrue(), pg_col = whilelt(j, J));
      j += vlength())
  {

    svbool_t pg_row;
    for(CeedSize c = 0;
        svptest_first(ptrue(), pg_row = whilelt(c, C));
        c += vlength())
    {
      svzero_za();
      const CeedInt n = svcntp_b64(ptrue(), pg_col);

      if(add)
        for(CeedInt i = 0; i < n; i++)
        {          
          load_za_row(0, i, pg_row, v + ((CeedSize)j + i) * C + c);
        }

      for(CeedInt b = 0; b < B; b++)
      {
        CeedScalar tmp[vlength()];
  
        for(CeedInt i = 0; i < n; i++)
          tmp[i] = t[((CeedSize)j + i) * s0 + (CeedSize)b * s1];
        rtype tt = load_vec(pg_col, tmp);

        rtype uu = load_vec(pg_row, u + (CeedSize)b * C + c);

        fmopa(0, pg_col, pg_row, tt, uu);
      }

      for(CeedInt i = 0; i < n; i++)
      {          
        store_za_row(0, i, pg_row, v + ((CeedSize)j + i) * C + c);
      }
    }
  }
  return CEED_ERROR_SUCCESS;
}


//------------------------------------------------------------------------------
// Tensor Contract Apply
//------------------------------------------------------------------------------
static int CeedTensorContractApply_Sme(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const CeedScalar *restrict t,
                                       CeedTransposeMode t_mode, const CeedInt add, const CeedScalar *restrict u, CeedScalar *restrict v) {
  for (CeedInt a = 0; a < A; a++) {
    CeedCallBackend(CeedTensorContract_Sme_Slice(B, C, J, t, t_mode, add, &u[(CeedSize)a * B * C], &v[(CeedSize)a * J * C]));
  }
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
// Tensor Contract Create
//------------------------------------------------------------------------------
int CeedTensorContractCreate_Sme(CeedTensorContract contract) {
  CeedCallBackend(CeedSetBackendFunction(CeedTensorContractReturnCeed(contract), "TensorContract", contract, "Apply", CeedTensorContractApply_Sme));
  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
