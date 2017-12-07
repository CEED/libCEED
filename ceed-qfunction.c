// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
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

#include <ceed-impl.h>

int CeedQFunctionCreateInterior(Ceed ceed, CeedInt vlength, CeedInt nfields,
                                size_t qdatasize, CeedEvalMode inmode, CeedEvalMode outmode,
                                int (*f)(void*, void*, CeedInt, const CeedScalar *const*, CeedScalar *const*),
                                const char *focca, CeedQFunction *qf)
{
   int ierr;

   if (!ceed->QFunctionCreate) { return CeedError(ceed, 1, "Backend does not support QFunctionCreate"); }
   ierr = CeedCalloc(1,qf); CeedChk(ierr);
   (*qf)->ceed = ceed;
   (*qf)->vlength = vlength;
   (*qf)->nfields = nfields;
   (*qf)->qdatasize = qdatasize;
   (*qf)->inmode = inmode;
   (*qf)->outmode = outmode;
   (*qf)->function = f;
   (*qf)->focca = focca;
   ierr = ceed->QFunctionCreate(*qf); CeedChk(ierr);
   return 0;
}

int CeedQFunctionSetContext(CeedQFunction qf, void *ctx, size_t ctxsize)
{
   qf->ctx = ctx;
   qf->ctxsize = ctxsize;
   return 0;
}

int CeedQFunctionDestroy(CeedQFunction *qf)
{
   int ierr;

   if (!*qf) { return 0; }
   if ((*qf)->Destroy)
   {
      ierr = (*qf)->Destroy(*qf); CeedChk(ierr);
   }
   ierr = CeedFree(qf); CeedChk(ierr);
   return 0;
}
