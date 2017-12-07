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

int CeedVectorCreate(Ceed ceed, CeedInt length, CeedVector *vec)
{
   int ierr;

   if (!ceed->VecCreate) { return CeedError(ceed, 1, "Backend does not support VecCreate"); }
   ierr = CeedCalloc(1,vec); CeedChk(ierr);
   (*vec)->ceed = ceed;
   (*vec)->length = length;
   ierr = ceed->VecCreate(ceed, length, *vec); CeedChk(ierr);
   return 0;
}

int CeedVectorSetArray(CeedVector x, CeedMemType mtype, CeedCopyMode cmode,
                       CeedScalar *array)
{
   int ierr;

   if (!x || !x->SetArray) { return CeedError(x ? x->ceed : NULL, 1, "Not supported"); }
   ierr = x->SetArray(x, mtype, cmode, array); CeedChk(ierr);
   return 0;
}

int CeedVectorGetArray(CeedVector x, CeedMemType mtype, CeedScalar **array)
{
   int ierr;

   if (!x || !x->GetArray) { return CeedError(x ? x->ceed : NULL, 1, "Not supported"); }
   ierr = x->GetArray(x, mtype, array); CeedChk(ierr);
   return 0;
}

int CeedVectorGetArrayRead(CeedVector x, CeedMemType mtype,
                           const CeedScalar **array)
{
   int ierr;

   if (!x || !x->GetArrayRead) { return CeedError(x ? x->ceed : NULL, 1, "Not supported"); }
   ierr = x->GetArrayRead(x, mtype, array); CeedChk(ierr);
   return 0;
}

int CeedVectorRestoreArray(CeedVector x, CeedScalar **array)
{
   int ierr;

   if (!x || !x->RestoreArray) { return CeedError(x ? x->ceed : NULL, 1, "Not supported"); }
   ierr = x->RestoreArray(x, array); CeedChk(ierr);
   return 0;
}

int CeedVectorRestoreArrayRead(CeedVector x, const CeedScalar **array)
{
   int ierr;

   if (!x || !x->RestoreArrayRead) { return CeedError(x ? x->ceed : NULL, 1, "Not supported"); }
   ierr = x->RestoreArrayRead(x, array); CeedChk(ierr);
   return 0;
}

int CeedVectorDestroy(CeedVector *x)
{
   int ierr;

   if (!*x) { return 0; }
   if ((*x)->Destroy)
   {
      ierr = (*x)->Destroy(*x); CeedChk(ierr);
   }
   ierr = CeedFree(x); CeedChk(ierr);
   return 0;
}
