#include <ceed-impl.h>
#include <ceed-backend.h>
#include <string.h>

/**
  @brief Create a CeedQFunction for evaluating interior (volumetric) terms.

  @param ceed       A Ceed object where the CeedQFunction will be created
  @param vlength    Vector length.  Caller must ensure that number of quadrature
                    points is a multiple of vlength.
  @param spec       The specification of the desired qFunction
  @param focca      OCCA identifier "file.c:function_name" for definition of `f`
  @param[out] qf    Address of the variable where the newly created
                     CeedQFunction will be stored

  @return An error code: 0 - success, otherwise - failure
**/
int CeedQFunctionCreateInteriorFromGallery(Ceed ceed, CeedInt vlength, char* spec, CeedQFunction *qf) {
  int ierr;
  if (!strcmp(spec,"elliptic"))
  {
    ierr = CeedCalloc(1,qf); CeedChk(ierr);
    (*qf)->ceed = ceed;
    ceed->refcount++;
    (*qf)->refcount = 1;
    (*qf)->vlength = vlength;
    (*qf)->function = NULL;//TODO give a default implementation
    //(*qf)->focca = focca_copy; //TODO
    (*qf)->spec = spec;
    ierr = ceed->QFunctionCreate(*qf); CeedChk(ierr);
    CeedQFunctionAddInput (*qf, "ggeo", 1, CEED_EVAL_NONE);
    CeedQFunctionAddInput (*qf, "D", 1, CEED_EVAL_NONE);
    CeedQFunctionAddInput (*qf, "S", 1, CEED_EVAL_NONE);
    CeedQFunctionAddInput (*qf, "MM", 1, CEED_EVAL_NONE);
    CeedQFunctionAddInput (*qf, "lambda", 1, CEED_EVAL_NONE);
    CeedQFunctionAddInput (*qf, "q", 1, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(*qf, "Aq", 1, CEED_EVAL_NONE);
    return 0;
  }
  return 1;
}

