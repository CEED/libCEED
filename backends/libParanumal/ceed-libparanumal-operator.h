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

#include <ceed-impl.h>

#if 0
typedef struct {
  CeedScalar *h_array;
  CeedScalar *h_array_allocated;
  CeedScalar *d_array;
  CeedScalar *d_array_allocated;
} CeedVector_libparanumal;

typedef struct {
  CUmodule module;
  CUfunction noTrNoTr;
  CUfunction noTrTr;
  CUfunction trNoTr;
  CUfunction trTr;
  CeedVector indices;
} CeedElemRestriction_libparanumal;

typedef struct {
  CUmodule module;
  CUfunction callback;
  const CeedScalar *const *d_u;
  CeedScalar *const *d_v;
  void *d_c;
} CeedQFunction_libparanumal;

typedef struct {
  CUmodule module;
  CUfunction interp;
  CUfunction grad;
  CUfunction weight;
  CeedScalar *d_interp1d;
  CeedScalar *d_grad1d;
  CeedScalar *d_qweight1d;
} CeedBasis_libparanumal;

typedef struct {
  int optblocksize;
  Ceed ceedref;
} Ceed_libparanumal;

CEED_INTERN int compile(Ceed ceed, const char *source, CUmodule *module, const CeedInt numopts, ...);

CEED_INTERN int get_kernel(Ceed ceed, CUmodule module, const char *name, CUfunction* kernel);

CEED_INTERN int run_kernel(Ceed ceed, CUfunction kernel, const int gridSize, const int blockSize, void **args);

CEED_INTERN int CeedVectorCreate_libparanumal(Ceed ceed, CeedInt n, CeedVector vec);

CEED_INTERN int CeedElemRestrictionCreate_libparanumal(CeedElemRestriction r,
    CeedMemType mtype,
    CeedCopyMode cmode, const CeedInt *indices);

CEED_INTERN int CeedBasisApplyElems_libparanumal(CeedBasis basis, const CeedInt nelem, CeedTransposeMode tmode, CeedEvalMode emode, const CeedVector u, CeedVector v);

CEED_INTERN int CeedQFunctionApplyElems_libparanumal(CeedQFunction qf, const CeedInt Q, const CeedVector *const u, const CeedVector* v);

CEED_INTERN int CeedBasisCreateTensorH1_libparanumal(Ceed ceed, CeedInt dim, CeedInt P1d,
    CeedInt Q1d, const CeedScalar *interp1d,
    const CeedScalar *grad1d,
    const CeedScalar *qref1d,
    const CeedScalar *qweight1d,
    CeedBasis basis);

CEED_INTERN int CeedQFunctionCreate_libparanumal(CeedQFunction qf);
#endif

#define BUFSIZ 16

typedef struct {
  //libCeed stuff
  CeedVector  *evecs;   /// E-vectors needed to apply operator (input followed by outputs)
  CeedScalar **edata;
  CeedScalar **qdata; /// Inputs followed by outputs
  CeedScalar **qdata_alloc; /// Allocated quadrature data arrays (to be freed by us)
  CeedScalar **indata;
  CeedScalar **outdata;
  CeedInt    numein;
  CeedInt    numeout;
  CeedInt    numqin;
  CeedInt    numqout;
  //libParanumal stuff
  mesh_t mesh;
  char[BUFSIZ] fileName;
  char[BUFSIZ] kernelName;
  occaProperties kernelInfo;
  occaKernel kernel;//Might need more than one
} CeedOperator_libparanumal;
