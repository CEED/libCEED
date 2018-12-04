#include "ceed-libparanumal.h"

//int CeedQFunctionCreate_libparanumal(CeedQFunction qf) {
//  int ierr;
//  Ceed ceed;
//  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);
//
//  CeedQFunction_Occa *impl;
//  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
//  ierr = CeedCalloc(16, &impl->inputs); CeedChk(ierr);
//  ierr = CeedCalloc(16, &impl->outputs); CeedChk(ierr);
//  ierr = CeedQFunctionSetData(qf, (void*)&impl); CeedChk(ierr);
  
//ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Apply",
//                              CeedQFunctionApply_Ref); CeedChk(ierr);
//ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Destroy",
//                              CeedQFunctionDestroy_Ref); CeedChk(ierr);
//
//  return 0;
//}
