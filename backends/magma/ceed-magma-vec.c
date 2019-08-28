// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.

// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.

// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.


// *****************************************************************************
// * Initialize vector vec (after free mem) with values from array based on cmode
// *   CEED_COPY_VALUES: memory is allocated in vec->array_allocated, made equal
// *                     to array, and data is copied (not store passed pointer)
// *   CEED_OWN_POINTER: vec->data->array_allocated and vec->data->array = array
// *   CEED_USE_POINTER: vec->data->array = array (can modify; no ownership)
// * mtype: CEED_MEM_HOST or CEED_MEM_DEVICE
// *****************************************************************************
static int CeedVectorSetArray_Magma(CeedVector vec, CeedMemType mtype,
                                    CeedCopyMode cmode, CeedScalar *array) {
  CeedVector_Magma *impl = vec->data;
  int ierr;

  // If own data, free the "old" data, e.g., as it may be of different size
  if (impl->own_) {
    magma_free( impl->darray );
    magma_free_pinned( impl->array );
    impl->darray = NULL;
    impl->array  = NULL;
    impl->own_ = 0;
    impl->down_= 0;
  }

  if (mtype == CEED_MEM_HOST) {
    // memory is on the host; own_ = 0
    switch (cmode) {
    case CEED_COPY_VALUES:
      ierr = magma_malloc( (void**)&impl->darray,
                           vec->length * sizeof(CeedScalar)); CeedChk(ierr);
      ierr = magma_malloc_pinned( (void**)&impl->array,
                                  vec->length * sizeof(CeedScalar)); CeedChk(ierr);
      impl->own_ = 1;

      if (array != NULL)
        magma_setvector(vec->length, sizeof(array[0]),
                        array, 1, impl->darray, 1);
      break;
    case CEED_OWN_POINTER:
      ierr = magma_malloc( (void**)&impl->darray,
                           vec->length * sizeof(CeedScalar)); CeedChk(ierr);
      // TODO: possible problem here is if we are passed non-pinned memory;
      //       (as we own it, lter in destroy, we use free for pinned memory).
      impl->array = array;
      impl->own_ = 1;

      if (array != NULL)
        magma_setvector(vec->length, sizeof(array[0]),
                        array, 1, impl->darray, 1);
      break;
    case CEED_USE_POINTER:
      ierr = magma_malloc( (void**)&impl->darray,
                           vec->length * sizeof(CeedScalar)); CeedChk(ierr);
      magma_setvector(vec->length, sizeof(array[0]),
                      array, 1, impl->darray, 1);

      impl->down_  = 1;
      impl->array  = array;
    }
  } else if (mtype == CEED_MEM_DEVICE) {
    // memory is on the device; own = 0
    switch (cmode) {
    case CEED_COPY_VALUES:
      ierr = magma_malloc( (void**)&impl->darray,
                           vec->length * sizeof(CeedScalar)); CeedChk(ierr);
      ierr = magma_malloc_pinned( (void**)&impl->array,
                                  vec->length * sizeof(CeedScalar)); CeedChk(ierr);
      impl->own_ = 1;

      if (array)
        magma_copyvector(vec->length, sizeof(array[0]),
                         array, 1, impl->darray, 1);
      else
        // t30 assumes allocation initializes with 0s
        magma_setvector(vec->length, sizeof(array[0]),
                        impl->array, 1, impl->darray, 1);
      break;
    case CEED_OWN_POINTER:
      impl->darray = array;
      ierr = magma_malloc_pinned( (void**)&impl->array,
                                  vec->length * sizeof(CeedScalar)); CeedChk(ierr);
      impl->own_ = 1;

      break;
    case CEED_USE_POINTER:
      impl->darray = array;
      impl->array  = NULL;
    }

  } else
    return CeedError(vec->ceed, 1, "Only MemType = HOST or DEVICE supported");

  return 0;
}

// *****************************************************************************
// * Give data pointer from vector vec to array (on HOST or DEVICE)
// *****************************************************************************
static int CeedVectorGetArray_Magma(CeedVector vec, CeedMemType mtype,
                                    CeedScalar **array) {
  CeedVector_Magma *impl = vec->data;
  int ierr;

  if (mtype == CEED_MEM_HOST) {
    if (impl->own_) {
      // data is owned so GPU had the most up-to-date version; copy it
      // TTT - apparantly it doesn't have most up to date data
      magma_getvector(vec->length, sizeof(*array[0]),
                      impl->darray, 1, impl->array, 1);
      CeedDebug("\033[31m[CeedVectorGetArray_Magma]");
      //fprintf(stderr,"rrrrrrrrrrrrrrr\n");
    } else if (impl->array == NULL) {
      // Vector doesn't own the data and was set on GPU
      if (impl->darray == NULL) {
        // call was made just to allocate memory
        ierr = CeedVectorSetArray(vec, mtype, CEED_COPY_VALUES, NULL);
        CeedChk(ierr);
      } else
        return CeedError(vec->ceed, 1, "Can not access DEVICE vector on HOST");
    }
    *array = impl->array;
  } else if (mtype == CEED_MEM_DEVICE) {
    if (impl->darray == NULL) {
      // Vector doesn't own the data and was set on the CPU
      if (impl->array == NULL) {
        // call was made just to allocate memory
        ierr = CeedVectorSetArray(vec, mtype, CEED_COPY_VALUES, NULL);
        CeedChk(ierr);
      } else
        return CeedError(vec->ceed, 1, "Can not access HOST vector on DEVICE");
    }
    *array = impl->darray;
  } else
    return CeedError(vec->ceed, 1, "Can only provide to HOST or DEVICE memory");

  return 0;
}

// *****************************************************************************
// * Give data pointer from vector vec to array (on HOST or DEVICE) to read it
// *****************************************************************************
static int CeedVectorGetArrayRead_Magma(CeedVector vec, CeedMemType mtype,
                                        const CeedScalar **array) {
  CeedVector_Magma *impl = vec->data;
  int ierr;

  if (mtype == CEED_MEM_HOST) {
    if (impl->own_) {
      // data is owned so GPU had the most up-to-date version; copy it
      magma_getvector(vec->length, sizeof(*array[0]),
                      impl->darray, 1, impl->array, 1);
    } else if (impl->array == NULL) {
      // Vector doesn't own the data and was set on GPU
      if (impl->darray == NULL) {
        // call was made just to allocate memory
        ierr = CeedVectorSetArray(vec, mtype, CEED_COPY_VALUES, NULL);
        CeedChk(ierr);
      } else
        return CeedError(vec->ceed, 1, "Can not access DEVICE vector on HOST");
    }
    *array = impl->array;
  } else if (mtype == CEED_MEM_DEVICE) {
    if (impl->darray == NULL) {
      // Vector doesn't own the data and was set on the CPU
      if (impl->array == NULL) {
        // call was made just to allocate memory
        ierr = CeedVectorSetArray(vec, mtype, CEED_COPY_VALUES, NULL);
        CeedChk(ierr);
      } else
        return CeedError(vec->ceed, 1, "Can not access HOST vector on DEVICE");
    }
    *array = impl->darray;
  } else
    return CeedError(vec->ceed, 1, "Can only provide to HOST or DEVICE memory");

  return 0;
}

// *****************************************************************************
// * There is no mtype here for array so it is not clear if we restore from HOST
// * memory or from DEVICE memory. We assume that it is CPU memory because if
// * it was GPU memory we would not call this routine at all.
// * Restore vector vec with values from array, where array received its values
// * from vec and possibly modified them.
// *****************************************************************************
static int CeedVectorRestoreArray_Magma(CeedVector vec, CeedScalar **array) {
  CeedVector_Magma *impl = vec->data;

  // Check if the array is a CPU pointer
  if (*array == impl->array) {
    // Update device, if the device pointer is not NULL
    if (impl->darray != NULL) {
      magma_setvector(vec->length, sizeof(*array[0]),
                      *array, 1, impl->darray, 1);
    } else {
      // nothing to do (case of CPU use pointer)
    }

  } else if (impl->down_) {
    // nothing to do if array is on GPU, except if down_=1(case CPU use pointer)
    magma_getvector(vec->length, sizeof(*array[0]),
                    impl->darray, 1, impl->array, 1);
  }

  *array = NULL;
  return 0;
}

// *****************************************************************************
// * There is no mtype here for array so it is not clear if we restore from HOST
// * memory or from DEVICE memory. We assume that it is CPU memory because if
// * it was GPU memory we would not call this routine at all.
// * Restore vector vec with values from array, where array received its values
// * from vec to only read them; in this case vec may have been modified meanwhile
// * and needs to be restored here.
// *****************************************************************************
static int CeedVectorRestoreArrayRead_Magma(CeedVector vec,
    const CeedScalar **array) {
  CeedVector_Magma *impl = vec->data;

  // Check if the array is a CPU pointer
  if (*array == impl->array) {
    // Update device, if the device pointer is not NULL
    if (impl->darray != NULL) {
      magma_setvector(vec->length, sizeof(*array[0]),
                      *array, 1, impl->darray, 1);
    } else {
      // nothing to do (case of CPU use pointer)
    }

  } else if (impl->down_) {
    // nothing to do if array is on GPU, except if down_=1(case CPU use pointer)
    magma_getvector(vec->length, sizeof(*array[0]),
                    impl->darray, 1, impl->array, 1);
  }

  *array = NULL;
  return 0;
}

static int CeedVectorDestroy_Magma(CeedVector vec) {
  CeedVector_Magma *impl = vec->data;
  int ierr;

  // Free if we own the data
  if (impl->own_) {
    ierr = magma_free_pinned(impl->array); CeedChk(ierr);
    ierr = magma_free(impl->darray);       CeedChk(ierr);
  } else if (impl->down_) {
    ierr = magma_free(impl->darray);       CeedChk(ierr);
  }
  ierr = CeedFree(&vec->data); CeedChk(ierr);
  return 0;
}

int CeedVectorCreate_Magma(CeedInt n, CeedVector vec) {
  int ierr;
  CeedVector_Magma *impl;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);

  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "SetArray",
                                CeedVectorSetArray_Magma); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "GetArray",
                                CeedVectorGetArray_Magma); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayRead",
                                CeedVectorGetArrayRead_Magma); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "RestoreArray",
                                CeedVectorRestoreArray_Magma); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "RestoreArrayRead",
                                CeedVectorRestoreArrayRead_Magma); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "Destroy",
                                CeedVectorDestroy_Magma); CeedChk(ierr);
  ierr = CeedCalloc(1,&impl); CeedChk(ierr);
  ierr = CeedVectorSetData(vec, (void *)&impl); CeedChk(ierr);
  return 0;

}
