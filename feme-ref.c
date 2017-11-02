#include <feme-impl.h>
#include <string.h>

typedef struct {
  FemeScalar *array;
  FemeScalar *array_allocated;
} FemeVec_Ref;

static int FemeVecSetArray_Ref(FemeVec vec, FemeMemType mtype, FemeCopyMode cmode, FemeScalar *array) {
  FemeVec_Ref *impl = vec->data;
  int ierr;

  if (mtype != FEME_MEM_HOST) FemeError(vec->feme, 1, "Only MemType = HOST supported");
  switch (cmode) {
  case FEME_COPY_VALUES:
    ierr = FemeMalloc(vec->n, &impl->array_allocated);FemeChk(ierr);
    impl->array = impl->array_allocated;
    memcpy(impl->array, array, vec->n * sizeof(array[0]));
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

static int FemeInit_Ref(const char *resource, Feme feme) {
  if (strcmp(resource, "/cpu/self") && strcmp(resource, "/cpu/self/ref")) return FemeError(feme, 1, "Ref backend cannot use resource: %s", resource);
  feme->VecCreate = FemeVecCreate_Ref;
  return 0;
}

__attribute__((constructor))
static void Register(void) {
  FemeRegister("/cpu/self/ref", FemeInit_Ref);
}
