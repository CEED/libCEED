#include <feme-impl.h>
#include <string.h>

typedef struct {
  FemeScalar *array;
  FemeScalar *array_allocated;
} FemeVec_Ref;

typedef struct {
  const FemeInt *indices;
  FemeInt *indices_allocated;
} FemeElemRestriction_Ref;

static int FemeVecSetArray_Ref(FemeVec vec, FemeMemType mtype, FemeCopyMode cmode, FemeScalar *array) {
  FemeVec_Ref *impl = vec->data;
  int ierr;

  if (mtype != FEME_MEM_HOST) FemeError(vec->feme, 1, "Only MemType = HOST supported");
  switch (cmode) {
  case FEME_COPY_VALUES:
    ierr = FemeMalloc(vec->n, &impl->array_allocated);FemeChk(ierr);
    impl->array = impl->array_allocated;
    if (array) memcpy(impl->array, array, vec->n * sizeof(array[0]));
    break;
  case FEME_OWN_POINTER:
    impl->array_allocated = array;
    impl->array = array;
    break;
  case FEME_USE_POINTER:
    impl->array = array;
  }
  return 0;
}

static int FemeVecGetArray_Ref(FemeVec vec, FemeMemType mtype, FemeScalar **array) {
  FemeVec_Ref *impl = vec->data;

  if (mtype != FEME_MEM_HOST) FemeError(vec->feme, 1, "Can only provide to HOST memory");
  *array = impl->array;
  return 0;
}

static int FemeVecGetArrayRead_Ref(FemeVec vec, FemeMemType mtype, const FemeScalar **array) {
  FemeVec_Ref *impl = vec->data;

  if (mtype != FEME_MEM_HOST) FemeError(vec->feme, 1, "Can only provide to HOST memory");
  *array = impl->array;
  return 0;
}

static int FemeVecRestoreArray_Ref(FemeVec vec, FemeScalar **array) {
  *array = NULL;
  return 0;
}

static int FemeVecRestoreArrayRead_Ref(FemeVec vec, const FemeScalar **array) {
  *array = NULL;
  return 0;
}

static int FemeVecDestroy_Ref(FemeVec vec) {
  FemeVec_Ref *impl = vec->data;
  int ierr;

  ierr = FemeFree(&impl->array_allocated);FemeChk(ierr);
  ierr = FemeFree(&vec->data);FemeChk(ierr);
  return 0;
}

static int FemeVecCreate_Ref(Feme feme, FemeInt n, FemeVec vec) {
  FemeVec_Ref *impl;
  int ierr;

  vec->SetArray = FemeVecSetArray_Ref;
  vec->GetArray = FemeVecGetArray_Ref;
  vec->GetArrayRead = FemeVecGetArrayRead_Ref;
  vec->RestoreArray = FemeVecRestoreArray_Ref;
  vec->RestoreArrayRead = FemeVecRestoreArrayRead_Ref;
  vec->Destroy = FemeVecDestroy_Ref;
  ierr = FemeCalloc(1,&impl);FemeChk(ierr);
  vec->data = impl;
  return 0;
}

static int FemeElemRestrictionApply_Ref(FemeElemRestriction r, FemeTransposeMode tmode, FemeVec u, FemeVec v, FemeRequest *request) {
  FemeElemRestriction_Ref *impl = r->data;
  int ierr;
  const FemeScalar *uu;
  FemeScalar *vv;

  ierr = FemeVecGetArrayRead(u, FEME_MEM_HOST, &uu);FemeChk(ierr);
  ierr = FemeVecGetArray(v, FEME_MEM_HOST, &vv);FemeChk(ierr);
  if (tmode == FEME_NOTRANSPOSE) {
    for (FemeInt i=0; i<r->nelem*r->elemsize; i++) vv[i] = uu[impl->indices[i]];
  } else {
    for (FemeInt i=0; i<r->nelem*r->elemsize; i++) vv[impl->indices[i]] += uu[i];
  }
  ierr = FemeVecRestoreArrayRead(u, &uu);FemeChk(ierr);
  ierr = FemeVecRestoreArray(v, &vv);FemeChk(ierr);
  if (request != FEME_REQUEST_IMMEDIATE) *request = NULL;
  return 0;
}

static int FemeElemRestrictionDestroy_Ref(FemeElemRestriction r) {
  FemeElemRestriction_Ref *impl = r->data;
  int ierr;

  ierr = FemeFree(&impl->indices_allocated);FemeChk(ierr);
  ierr = FemeFree(&r->data);FemeChk(ierr);
  return 0;
}

static int FemeElemRestrictionCreate_Ref(FemeElemRestriction r, FemeMemType mtype, FemeCopyMode cmode, const FemeInt *indices) {
  int ierr;
  FemeElemRestriction_Ref *impl;

  if (mtype != FEME_MEM_HOST) FemeError(r->feme, 1, "Only MemType = HOST supported");
  ierr = FemeCalloc(1,&impl);FemeChk(ierr);
  switch (cmode) {
  case FEME_COPY_VALUES:
    ierr = FemeMalloc(r->nelem*r->elemsize, &impl->indices_allocated);FemeChk(ierr);
    memcpy(impl->indices_allocated, indices, r->nelem * r->elemsize * sizeof(indices[0]));
    impl->indices = impl->indices_allocated;
    break;
  case FEME_OWN_POINTER:
    impl->indices_allocated = (FemeInt*)indices;
    impl->indices = impl->indices_allocated;
    break;
  case FEME_USE_POINTER:
    impl->indices = indices;
  }
  r->data = impl;
  r->Apply = FemeElemRestrictionApply_Ref;
  r->Destroy = FemeElemRestrictionDestroy_Ref;
  return 0;
}

// Contracts on the middle index
// NOTRANSPOSE: V_ajc = T_jb U_abc
// TRANSPOSE:   V_ajc = T_bj U_abc
static int FemeTensorContract_Ref(Feme feme, FemeInt A, FemeInt B, FemeInt C, FemeInt J, const FemeScalar *t, FemeTransposeMode tmode, const FemeScalar *u, FemeScalar *v) {
  FemeInt tstride0 = B, tstride1 = 1;
  if (tmode == FEME_TRANSPOSE) {
    tstride0 = 1; tstride1 = B;
  }

  for (FemeInt a=0; a<A; a++) {
    for (FemeInt j=0; j<J; j++) {
      for (FemeInt c=0; c<C; c++)
        v[(a*J+j)*C+c] = 0;
      for (FemeInt b=0; b<B; b++) {
        for (FemeInt c=0; c<C; c++) {
          v[(a*J+j)*C+c] += t[j*tstride0 + b*tstride1] * u[(a*B+b)*C+c];
        }
      }
    }
  }
  return 0;
}

static int FemeBasisApply_Ref(FemeBasis basis, FemeTransposeMode tmode, FemeEvalMode emode, const FemeScalar *u, FemeScalar *v) {
  int ierr;
  const FemeInt dim = basis->dim;

  switch (emode) {
  case FEME_EVAL_INTERP: {
    FemeInt P = basis->P1d, Q = basis->Q1d;
    if (tmode == FEME_TRANSPOSE) {
      P = basis->Q1d; Q = basis->P1d;
    }
    FemeInt pre = FemePowInt(P, dim-1), post = 1;
    FemeScalar tmp[2][Q*FemePowInt(P>Q?P:Q, dim-1)];
    for (FemeInt d=0; d<dim; d++) {
      ierr = FemeTensorContract_Ref(basis->feme, pre, P, post, Q, basis->interp1d, tmode, d==0?u:tmp[d%2], d==dim-1?v:tmp[(d+1)%2]);FemeChk(ierr);
      pre /= P;
      post *= Q;
    }
  } break;
  default:
    return FemeError(basis->feme, 1, "EvalMode %d not supported", emode);
  }
  return 0;
}

static int FemeBasisDestroy_Ref(FemeBasis basis) {
  return 0;
}

static int FemeBasisCreateTensorH1_Ref(Feme feme, FemeInt dim, FemeInt P1d, FemeInt Q1d, const FemeScalar *interp1d, const FemeScalar *grad1d, const FemeScalar *qref1d, const FemeScalar *qweight1d, FemeBasis basis) {
  basis->Apply = FemeBasisApply_Ref;
  basis->Destroy = FemeBasisDestroy_Ref;
  return 0;
}

static int FemeInit_Ref(const char *resource, Feme feme) {
  if (strcmp(resource, "/cpu/self") && strcmp(resource, "/cpu/self/ref")) return FemeError(feme, 1, "Ref backend cannot use resource: %s", resource);
  feme->VecCreate = FemeVecCreate_Ref;
  feme->BasisCreateTensorH1 = FemeBasisCreateTensorH1_Ref;
  feme->ElemRestrictionCreate = FemeElemRestrictionCreate_Ref;
  return 0;
}

__attribute__((constructor))
static void Register(void) {
  FemeRegister("/cpu/self/ref", FemeInit_Ref);
}
