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

#include <ceed/ceed.h>
#include <ceed/backend.h>
#include <ceed-impl.h>
#include <stdbool.h>
#include <stdio.h>

/// @file
/// Implementation of CeedElemRestriction interfaces

/// ----------------------------------------------------------------------------
/// CeedElemRestriction Library Internal Functions
/// ----------------------------------------------------------------------------
/// @addtogroup CeedElemRestrictionDeveloper
/// @{

/**
  @brief Permute and pad offsets for a blocked restriction

  @param offsets     Array of shape [@a num_elem, @a elem_size]. Row i holds the
                       ordered list of the offsets (into the input CeedVector)
                       for the unknowns corresponding to element i, where
                       0 <= i < @a num_elem. All offsets must be in the range
                       [0, @a l_size - 1].
  @param blk_offsets Array of permuted and padded offsets of
                       shape [@a num_blk, @a elem_size, @a blk_size].
  @param num_blk     Number of blocks
  @param num_elem    Number of elements
  @param blk_size    Number of elements in a block
  @param elem_size   Size of each element

  @return An error code: 0 - success, otherwise - failure

  @ref Utility
**/
int CeedPermutePadOffsets(const CeedInt *offsets, CeedInt *blk_offsets,
                          CeedInt num_blk, CeedInt num_elem, CeedInt blk_size,
                          CeedInt elem_size) {
  for (CeedInt e=0; e<num_blk*blk_size; e+=blk_size)
    for (int j=0; j<blk_size; j++)
      for (int k=0; k<elem_size; k++)
        blk_offsets[e*elem_size + k*blk_size + j]
          = offsets[CeedIntMin(e+j,num_elem-1)*elem_size + k];
  return CEED_ERROR_SUCCESS;
}

/// @}

/// ----------------------------------------------------------------------------
/// CeedElemRestriction Backend API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedElemRestrictionBackend
/// @{

/**

  @brief Get the strides of a strided CeedElemRestriction

  @param rstr          CeedElemRestriction
  @param[out] strides  Variable to store strides array

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionGetStrides(CeedElemRestriction rstr,
                                  CeedInt (*strides)[3]) {
  if (!rstr->strides)
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, CEED_ERROR_MINOR,
                     "ElemRestriction has no stride data");
  // LCOV_EXCL_STOP

  for (int i=0; i<3; i++)
    (*strides)[i] = rstr->strides[i];
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get read-only access to a CeedElemRestriction offsets array by memtype

  @param rstr          CeedElemRestriction to retrieve offsets
  @param mem_type      Memory type on which to access the array.  If the backend
                         uses a different memory type, this will perform a copy
                         (possibly cached).
  @param[out] offsets  Array on memory type mem_type

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionGetOffsets(CeedElemRestriction rstr,
                                  CeedMemType mem_type,
                                  const CeedInt **offsets) {
  int ierr;

  if (!rstr->GetOffsets)
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, CEED_ERROR_UNSUPPORTED,
                     "Backend does not support GetOffsets");
  // LCOV_EXCL_STOP

  ierr = rstr->GetOffsets(rstr, mem_type, offsets); CeedChk(ierr);
  rstr->num_readers++;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restore an offsets array obtained using CeedElemRestrictionGetOffsets()

  @param rstr     CeedElemRestriction to restore
  @param offsets  Array of offset data

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionRestoreOffsets(CeedElemRestriction rstr,
                                      const CeedInt **offsets) {
  *offsets = NULL;
  rstr->num_readers--;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the strided status of a CeedElemRestriction

  @param rstr             CeedElemRestriction
  @param[out] is_strided  Variable to store strided status, 1 if strided else 0

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionIsStrided(CeedElemRestriction rstr, bool *is_strided) {
  *is_strided = rstr->strides ? true : false;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the backend stride status of a CeedElemRestriction

  @param rstr                      CeedElemRestriction
  @param[out] has_backend_strides  Variable to store stride status

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionHasBackendStrides(CeedElemRestriction rstr,
    bool *has_backend_strides) {
  if (!rstr->strides)
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, CEED_ERROR_MINOR,
                     "ElemRestriction has no stride data");
  // LCOV_EXCL_STOP

  *has_backend_strides = ((rstr->strides[0] == CEED_STRIDES_BACKEND[0]) &&
                          (rstr->strides[1] == CEED_STRIDES_BACKEND[1]) &&
                          (rstr->strides[2] == CEED_STRIDES_BACKEND[2]));
  return CEED_ERROR_SUCCESS;
}

/**

  @brief Get the E-vector layout of a CeedElemRestriction

  @param rstr         CeedElemRestriction
  @param[out] layout  Variable to store layout array,
                        stored as [nodes, components, elements].
                        The data for node i, component j, element k in the
                        E-vector is given by
                        i*layout[0] + j*layout[1] + k*layout[2]

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionGetELayout(CeedElemRestriction rstr,
                                  CeedInt (*layout)[3]) {
  if (!rstr->layout[0])
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, CEED_ERROR_MINOR,
                     "ElemRestriction has no layout data");
  // LCOV_EXCL_STOP

  for (int i=0; i<3; i++)
    (*layout)[i] = rstr->layout[i];
  return CEED_ERROR_SUCCESS;
}

/**

  @brief Set the E-vector layout of a CeedElemRestriction

  @param rstr    CeedElemRestriction
  @param layout  Variable to containing layout array,
                   stored as [nodes, components, elements].
                   The data for node i, component j, element k in the
                   E-vector is given by
                   i*layout[0] + j*layout[1] + k*layout[2]

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionSetELayout(CeedElemRestriction rstr,
                                  CeedInt layout[3]) {
  for (int i = 0; i<3; i++)
    rstr->layout[i] = layout[i];
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the backend data of a CeedElemRestriction

  @param rstr       CeedElemRestriction
  @param[out] data  Variable to store data

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionGetData(CeedElemRestriction rstr, void *data) {
  *(void **)data = rstr->data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Set the backend data of a CeedElemRestriction

  @param[out] rstr  CeedElemRestriction
  @param data       Data to set

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionSetData(CeedElemRestriction rstr, void *data) {
  rstr->data = data;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Increment the reference counter for a CeedElemRestriction

  @param rstr  ElemRestriction to increment the reference counter

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionReference(CeedElemRestriction rstr) {
  rstr->ref_count++;
  return CEED_ERROR_SUCCESS;
}

/// @}

/// @cond DOXYGEN_SKIP
static struct CeedElemRestriction_private ceed_elemrestriction_none;
/// @endcond

/// ----------------------------------------------------------------------------
/// CeedElemRestriction Public API
/// ----------------------------------------------------------------------------
/// @addtogroup CeedElemRestrictionUser
/// @{

/// Indicate that the stride is determined by the backend
const CeedInt CEED_STRIDES_BACKEND[3] = {0};

/// Indicate that no CeedElemRestriction is provided by the user
const CeedElemRestriction CEED_ELEMRESTRICTION_NONE =
  &ceed_elemrestriction_none;

/**
  @brief Create a CeedElemRestriction

  @param ceed         A Ceed object where the CeedElemRestriction will be created
  @param num_elem     Number of elements described in the @a offsets array
  @param elem_size    Size (number of "nodes") per element
  @param num_comp     Number of field components per interpolation node
                        (1 for scalar fields)
  @param comp_stride  Stride between components for the same L-vector "node".
                        Data for node i, component j, element k can be found in
                        the L-vector at index
                        offsets[i + k*elem_size] + j*comp_stride.
  @param l_size       The size of the L-vector. This vector may be larger than
                        the elements and fields given by this restriction.
  @param mem_type     Memory type of the @a offsets array, see CeedMemType
  @param copy_mode    Copy mode for the @a offsets array, see CeedCopyMode
  @param offsets      Array of shape [@a num_elem, @a elem_size]. Row i holds the
                        ordered list of the offsets (into the input CeedVector)
                        for the unknowns corresponding to element i, where
                        0 <= i < @a num_elem. All offsets must be in the range
                        [0, @a l_size - 1].
  @param[out] rstr    Address of the variable where the newly created
                        CeedElemRestriction will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionCreate(Ceed ceed, CeedInt num_elem, CeedInt elem_size,
                              CeedInt num_comp, CeedInt comp_stride,
                              CeedInt l_size, CeedMemType mem_type,
                              CeedCopyMode copy_mode, const CeedInt *offsets,
                              CeedElemRestriction *rstr) {
  int ierr;

  if (!ceed->ElemRestrictionCreate) {
    Ceed delegate;
    ierr = CeedGetObjectDelegate(ceed, &delegate, "ElemRestriction");
    CeedChk(ierr);

    if (!delegate)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
                       "Backend does not support ElemRestrictionCreate");
    // LCOV_EXCL_STOP

    ierr = CeedElemRestrictionCreate(delegate, num_elem, elem_size, num_comp,
                                     comp_stride, l_size, mem_type, copy_mode,
                                     offsets, rstr); CeedChk(ierr);
    return CEED_ERROR_SUCCESS;
  }

  ierr = CeedCalloc(1, rstr); CeedChk(ierr);
  (*rstr)->ceed = ceed;
  ierr = CeedReference(ceed); CeedChk(ierr);
  (*rstr)->ref_count = 1;
  (*rstr)->num_elem = num_elem;
  (*rstr)->elem_size = elem_size;
  (*rstr)->num_comp = num_comp;
  (*rstr)->comp_stride = comp_stride;
  (*rstr)->l_size = l_size;
  (*rstr)->num_blk = num_elem;
  (*rstr)->blk_size = 1;
  ierr = ceed->ElemRestrictionCreate(mem_type, copy_mode, offsets, *rstr);
  CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a CeedElemRestriction with orientation sign

  @param ceed         A Ceed object where the CeedElemRestriction will be created
  @param num_elem     Number of elements described in the @a offsets array
  @param elem_size    Size (number of "nodes") per element
  @param num_comp     Number of field components per interpolation node
                        (1 for scalar fields)
  @param comp_stride  Stride between components for the same L-vector "node".
                        Data for node i, component j, element k can be found in
                        the L-vector at index
                        offsets[i + k*elem_size] + j*comp_stride.
  @param l_size       The size of the L-vector. This vector may be larger than
                        the elements and fields given by this restriction.
  @param mem_type     Memory type of the @a offsets array, see CeedMemType
  @param copy_mode    Copy mode for the @a offsets array, see CeedCopyMode
  @param offsets      Array of shape [@a num_elem, @a elem_size]. Row i holds the
                        ordered list of the offsets (into the input CeedVector)
                        for the unknowns corresponding to element i, where
                        0 <= i < @a num_elem. All offsets must be in the range
                        [0, @a l_size - 1].
  @param orient       Array of shape [@a num_elem, @a elem_size] with bool false
                        for positively oriented and true to flip the orientation.
  @param scale        An scalar value that scales the dofs in assembly.
  @param[out] rstr    Address of the variable where the newly created
                        CeedElemRestriction will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionCreateOriented(Ceed ceed, CeedInt num_elem,
                                      CeedInt elem_size, CeedInt num_comp,
                                      CeedInt comp_stride, CeedInt l_size,
                                      CeedMemType mem_type, CeedCopyMode copy_mode,
                                      const CeedInt *offsets, const bool *orient,
                                      CeedElemRestriction *rstr) {
  int ierr;

  if (!ceed->ElemRestrictionCreate) {
    Ceed delegate;
    ierr = CeedGetObjectDelegate(ceed, &delegate, "ElemRestriction");
    CeedChk(ierr);

    if (!delegate)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
                       "Backend does not support ElemRestrictionCreateOriented");
    // LCOV_EXCL_STOP

    ierr = CeedElemRestrictionCreateOriented(delegate, num_elem, elem_size,
           num_comp,
           comp_stride, l_size, mem_type, copy_mode,
           offsets, orient, rstr); CeedChk(ierr);
    return CEED_ERROR_SUCCESS;
  }

  ierr = CeedCalloc(1, rstr); CeedChk(ierr);
  (*rstr)->ceed = ceed;
  ierr = CeedReference(ceed); CeedChk(ierr);
  (*rstr)->ref_count = 1;
  (*rstr)->num_elem = num_elem;
  (*rstr)->elem_size = elem_size;
  (*rstr)->num_comp = num_comp;
  (*rstr)->comp_stride = comp_stride;
  (*rstr)->l_size = l_size;
  (*rstr)->num_blk = num_elem;
  (*rstr)->blk_size = 1;
  ierr = ceed->ElemRestrictionCreateOriented(mem_type, copy_mode, offsets, orient,
         *rstr);
  CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a strided CeedElemRestriction

  @param ceed       A Ceed object where the CeedElemRestriction will be created
  @param num_elem   Number of elements described by the restriction
  @param elem_size  Size (number of "nodes") per element
  @param num_comp   Number of field components per interpolation "node"
                      (1 for scalar fields)
  @param l_size     The size of the L-vector. This vector may be larger than
                      the elements and fields given by this restriction.
  @param strides    Array for strides between [nodes, components, elements].
                      Data for node i, component j, element k can be found in
                      the L-vector at index
                      i*strides[0] + j*strides[1] + k*strides[2].
                      @a CEED_STRIDES_BACKEND may be used with vectors created
                      by a Ceed backend.
  @param rstr       Address of the variable where the newly created
                      CeedElemRestriction will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionCreateStrided(Ceed ceed, CeedInt num_elem,
                                     CeedInt elem_size,
                                     CeedInt num_comp, CeedInt l_size,
                                     const CeedInt strides[3],
                                     CeedElemRestriction *rstr) {
  int ierr;

  if (!ceed->ElemRestrictionCreate) {
    Ceed delegate;
    ierr = CeedGetObjectDelegate(ceed, &delegate, "ElemRestriction");
    CeedChk(ierr);

    if (!delegate)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_UNSUPPORTED,
                       "Backend does not support ElemRestrictionCreate");
    // LCOV_EXCL_STOP

    ierr = CeedElemRestrictionCreateStrided(delegate, num_elem, elem_size, num_comp,
                                            l_size, strides, rstr);
    CeedChk(ierr);
    return CEED_ERROR_SUCCESS;
  }

  ierr = CeedCalloc(1, rstr); CeedChk(ierr);
  (*rstr)->ceed = ceed;
  ierr = CeedReference(ceed); CeedChk(ierr);
  (*rstr)->ref_count = 1;
  (*rstr)->num_elem = num_elem;
  (*rstr)->elem_size = elem_size;
  (*rstr)->num_comp = num_comp;
  (*rstr)->l_size = l_size;
  (*rstr)->num_blk = num_elem;
  (*rstr)->blk_size = 1;
  ierr = CeedMalloc(3, &(*rstr)->strides); CeedChk(ierr);
  for (int i=0; i<3; i++)
    (*rstr)->strides[i] = strides[i];
  ierr = ceed->ElemRestrictionCreate(CEED_MEM_HOST, CEED_OWN_POINTER, NULL,
                                     *rstr);
  CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a blocked CeedElemRestriction, typically only called by backends

  @param ceed         A Ceed object where the CeedElemRestriction will be created.
  @param num_elem     Number of elements described in the @a offsets array.
  @param elem_size    Size (number of unknowns) per element
  @param blk_size     Number of elements in a block
  @param num_comp     Number of field components per interpolation node
                        (1 for scalar fields)
  @param comp_stride  Stride between components for the same L-vector "node".
                        Data for node i, component j, element k can be found in
                        the L-vector at index
                        offsets[i + k*elem_size] + j*comp_stride.
  @param l_size       The size of the L-vector. This vector may be larger than
                        the elements and fields given by this restriction.
  @param mem_type     Memory type of the @a offsets array, see CeedMemType
  @param copy_mode    Copy mode for the @a offsets array, see CeedCopyMode
  @param offsets      Array of shape [@a num_elem, @a elem_size]. Row i holds the
                        ordered list of the offsets (into the input CeedVector)
                        for the unknowns corresponding to element i, where
                        0 <= i < @a num_elem. All offsets must be in the range
                        [0, @a l_size - 1]. The backend will permute and pad this
                        array to the desired ordering for the blocksize, which is
                        typically given by the backend. The default reordering is
                        to interlace elements.
  @param rstr         Address of the variable where the newly created
                        CeedElemRestriction will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
 **/
int CeedElemRestrictionCreateBlocked(Ceed ceed, CeedInt num_elem,
                                     CeedInt elem_size,
                                     CeedInt blk_size, CeedInt num_comp,
                                     CeedInt comp_stride, CeedInt l_size,
                                     CeedMemType mem_type, CeedCopyMode copy_mode,
                                     const CeedInt *offsets,
                                     CeedElemRestriction *rstr) {
  int ierr;
  CeedInt *blk_offsets;
  CeedInt num_blk = (num_elem / blk_size) + !!(num_elem % blk_size);

  if (!ceed->ElemRestrictionCreateBlocked) {
    Ceed delegate;
    ierr = CeedGetObjectDelegate(ceed, &delegate, "ElemRestriction");
    CeedChk(ierr);

    if (!delegate)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support "
                       "ElemRestrictionCreateBlocked");
    // LCOV_EXCL_STOP

    ierr = CeedElemRestrictionCreateBlocked(delegate, num_elem, elem_size, blk_size,
                                            num_comp, comp_stride, l_size, mem_type,
                                            copy_mode, offsets, rstr);
    CeedChk(ierr);
    return CEED_ERROR_SUCCESS;
  }

  ierr = CeedCalloc(1, rstr); CeedChk(ierr);

  ierr = CeedCalloc(num_blk*blk_size*elem_size, &blk_offsets); CeedChk(ierr);
  ierr = CeedPermutePadOffsets(offsets, blk_offsets, num_blk, num_elem, blk_size,
                               elem_size); CeedChk(ierr);

  (*rstr)->ceed = ceed;
  ierr = CeedReference(ceed); CeedChk(ierr);
  (*rstr)->ref_count = 1;
  (*rstr)->num_elem = num_elem;
  (*rstr)->elem_size = elem_size;
  (*rstr)->num_comp = num_comp;
  (*rstr)->comp_stride = comp_stride;
  (*rstr)->l_size = l_size;
  (*rstr)->num_blk = num_blk;
  (*rstr)->blk_size = blk_size;
  ierr = ceed->ElemRestrictionCreateBlocked(CEED_MEM_HOST, CEED_OWN_POINTER,
         (const CeedInt *) blk_offsets, *rstr); CeedChk(ierr);
  if (copy_mode == CEED_OWN_POINTER) {
    ierr = CeedFree(&offsets); CeedChk(ierr);
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create a blocked strided CeedElemRestriction

  @param ceed       A Ceed object where the CeedElemRestriction will be created
  @param num_elem   Number of elements described by the restriction
  @param elem_size  Size (number of "nodes") per element
  @param blk_size   Number of elements in a block
  @param num_comp   Number of field components per interpolation node
                      (1 for scalar fields)
  @param l_size     The size of the L-vector. This vector may be larger than
                      the elements and fields given by this restriction.
  @param strides    Array for strides between [nodes, components, elements].
                      Data for node i, component j, element k can be found in
                      the L-vector at index
                      i*strides[0] + j*strides[1] + k*strides[2].
                      @a CEED_STRIDES_BACKEND may be used with vectors created
                      by a Ceed backend.
  @param rstr       Address of the variable where the newly created
                      CeedElemRestriction will be stored

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionCreateBlockedStrided(Ceed ceed, CeedInt num_elem,
    CeedInt elem_size, CeedInt blk_size, CeedInt num_comp, CeedInt l_size,
    const CeedInt strides[3], CeedElemRestriction *rstr) {
  int ierr;
  CeedInt num_blk = (num_elem / blk_size) + !!(num_elem % blk_size);

  if (!ceed->ElemRestrictionCreateBlocked) {
    Ceed delegate;
    ierr = CeedGetObjectDelegate(ceed, &delegate, "ElemRestriction");
    CeedChk(ierr);

    if (!delegate)
      // LCOV_EXCL_START
      return CeedError(ceed, CEED_ERROR_UNSUPPORTED, "Backend does not support "
                       "ElemRestrictionCreateBlocked");
    // LCOV_EXCL_STOP

    ierr = CeedElemRestrictionCreateBlockedStrided(delegate, num_elem, elem_size,
           blk_size, num_comp, l_size, strides, rstr); CeedChk(ierr);
    return CEED_ERROR_SUCCESS;
  }

  ierr = CeedCalloc(1, rstr); CeedChk(ierr);

  (*rstr)->ceed = ceed;
  ierr = CeedReference(ceed); CeedChk(ierr);
  (*rstr)->ref_count = 1;
  (*rstr)->num_elem = num_elem;
  (*rstr)->elem_size = elem_size;
  (*rstr)->num_comp = num_comp;
  (*rstr)->l_size = l_size;
  (*rstr)->num_blk = num_blk;
  (*rstr)->blk_size = blk_size;
  ierr = CeedMalloc(3, &(*rstr)->strides); CeedChk(ierr);
  for (int i=0; i<3; i++)
    (*rstr)->strides[i] = strides[i];
  ierr = ceed->ElemRestrictionCreateBlocked(CEED_MEM_HOST, CEED_OWN_POINTER,
         NULL, *rstr); CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Copy the pointer to a CeedElemRestriction. Both pointers should
           be destroyed with `CeedElemRestrictionDestroy()`;
           Note: If `*rstr_copy` is non-NULL, then it is assumed that
           `*rstr_copy` is a pointer to a CeedElemRestriction. This
           CeedElemRestriction will be destroyed if `*rstr_copy` is the
           only reference to this CeedElemRestriction.

  @param rstr            CeedElemRestriction to copy reference to
  @param[out] rstr_copy  Variable to store copied reference

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionReferenceCopy(CeedElemRestriction rstr,
                                     CeedElemRestriction *rstr_copy) {
  int ierr;

  ierr = CeedElemRestrictionReference(rstr); CeedChk(ierr);
  ierr = CeedElemRestrictionDestroy(rstr_copy); CeedChk(ierr);
  *rstr_copy = rstr;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Create CeedVectors associated with a CeedElemRestriction

  @param rstr   CeedElemRestriction
  @param l_vec  The address of the L-vector to be created, or NULL
  @param e_vec  The address of the E-vector to be created, or NULL

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionCreateVector(CeedElemRestriction rstr, CeedVector *l_vec,
                                    CeedVector *e_vec) {
  int ierr;
  CeedInt e_size, l_size;
  l_size = rstr->l_size;
  e_size = rstr->num_blk * rstr->blk_size * rstr->elem_size * rstr->num_comp;
  if (l_vec) {
    ierr = CeedVectorCreate(rstr->ceed, l_size, l_vec); CeedChk(ierr);
  }
  if (e_vec) {
    ierr = CeedVectorCreate(rstr->ceed, e_size, e_vec); CeedChk(ierr);
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restrict an L-vector to an E-vector or apply its transpose

  @param rstr    CeedElemRestriction
  @param t_mode  Apply restriction or transpose
  @param u       Input vector (of size @a l_size when t_mode=@ref CEED_NOTRANSPOSE)
  @param ru      Output vector (of shape [@a num_elem * @a elem_size] when
                   t_mode=@ref CEED_NOTRANSPOSE). Ordering of the e-vector is decided
                   by the backend.
  @param request Request or @ref CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionApply(CeedElemRestriction rstr, CeedTransposeMode t_mode,
                             CeedVector u, CeedVector ru,
                             CeedRequest *request) {
  CeedInt m, n;
  int ierr;

  if (t_mode == CEED_NOTRANSPOSE) {
    m = rstr->num_blk * rstr->blk_size * rstr->elem_size * rstr->num_comp;
    n = rstr->l_size;
  } else {
    m = rstr->l_size;
    n = rstr->num_blk * rstr->blk_size * rstr->elem_size * rstr->num_comp;
  }
  if (n != u->length)
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, CEED_ERROR_DIMENSION,
                     "Input vector size %d not compatible with "
                     "element restriction (%d, %d)", u->length, m, n);
  // LCOV_EXCL_STOP
  if (m != ru->length)
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, CEED_ERROR_DIMENSION,
                     "Output vector size %d not compatible with "
                     "element restriction (%d, %d)", ru->length, m, n);
  // LCOV_EXCL_STOP
  ierr = rstr->Apply(rstr, t_mode, u, ru, request); CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Restrict an L-vector to a block of an E-vector or apply its transpose

  @param rstr    CeedElemRestriction
  @param block   Block number to restrict to/from, i.e. block=0 will handle
                   elements [0 : blk_size] and block=3 will handle elements
                   [3*blk_size : 4*blk_size]
  @param t_mode  Apply restriction or transpose
  @param u       Input vector (of size @a l_size when t_mode=@ref CEED_NOTRANSPOSE)
  @param ru      Output vector (of shape [@a blk_size * @a elem_size] when
                   t_mode=@ref CEED_NOTRANSPOSE). Ordering of the e-vector is decided
                   by the backend.
  @param request Request or @ref CEED_REQUEST_IMMEDIATE

  @return An error code: 0 - success, otherwise - failure

  @ref Backend
**/
int CeedElemRestrictionApplyBlock(CeedElemRestriction rstr, CeedInt block,
                                  CeedTransposeMode t_mode, CeedVector u,
                                  CeedVector ru, CeedRequest *request) {
  CeedInt m, n;
  int ierr;

  if (t_mode == CEED_NOTRANSPOSE) {
    m = rstr->blk_size * rstr->elem_size * rstr->num_comp;
    n = rstr->l_size;
  } else {
    m = rstr->l_size;
    n = rstr->blk_size * rstr->elem_size * rstr->num_comp;
  }
  if (n != u->length)
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, CEED_ERROR_DIMENSION,
                     "Input vector size %d not compatible with "
                     "element restriction (%d, %d)", u->length, m, n);
  // LCOV_EXCL_STOP
  if (m != ru->length)
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, CEED_ERROR_DIMENSION,
                     "Output vector size %d not compatible with "
                     "element restriction (%d, %d)", ru->length, m, n);
  // LCOV_EXCL_STOP
  if (rstr->blk_size*block > rstr->num_elem)
    // LCOV_EXCL_START
    return CeedError(rstr->ceed, CEED_ERROR_DIMENSION,
                     "Cannot retrieve block %d, element %d > "
                     "total elements %d", block, rstr->blk_size*block,
                     rstr->num_elem);
  // LCOV_EXCL_STOP
  ierr = rstr->ApplyBlock(rstr, block, t_mode, u, ru, request);
  CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the Ceed associated with a CeedElemRestriction

  @param rstr       CeedElemRestriction
  @param[out] ceed  Variable to store Ceed

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedElemRestrictionGetCeed(CeedElemRestriction rstr, Ceed *ceed) {
  *ceed = rstr->ceed;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the L-vector component stride

  @param rstr              CeedElemRestriction
  @param[out] comp_stride  Variable to store component stride

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedElemRestrictionGetCompStride(CeedElemRestriction rstr,
                                     CeedInt *comp_stride) {
  *comp_stride = rstr->comp_stride;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the total number of elements in the range of a CeedElemRestriction

  @param rstr           CeedElemRestriction
  @param[out] num_elem  Variable to store number of elements

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedElemRestrictionGetNumElements(CeedElemRestriction rstr,
                                      CeedInt *num_elem) {
  *num_elem = rstr->num_elem;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the size of elements in the CeedElemRestriction

  @param rstr            CeedElemRestriction
  @param[out] elem_size  Variable to store size of elements

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedElemRestrictionGetElementSize(CeedElemRestriction rstr,
                                      CeedInt *elem_size) {
  *elem_size = rstr->elem_size;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the size of the l-vector for a CeedElemRestriction

  @param rstr         CeedElemRestriction
  @param[out] l_size  Variable to store number of nodes

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedElemRestrictionGetLVectorSize(CeedElemRestriction rstr,
                                      CeedInt *l_size) {
  *l_size = rstr->l_size;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the number of components in the elements of a
         CeedElemRestriction

  @param rstr           CeedElemRestriction
  @param[out] num_comp  Variable to store number of components

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedElemRestrictionGetNumComponents(CeedElemRestriction rstr,
                                        CeedInt *num_comp) {
  *num_comp = rstr->num_comp;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the number of blocks in a CeedElemRestriction

  @param rstr            CeedElemRestriction
  @param[out] num_block  Variable to store number of blocks

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedElemRestrictionGetNumBlocks(CeedElemRestriction rstr,
                                    CeedInt *num_block) {
  *num_block = rstr->num_blk;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the size of blocks in the CeedElemRestriction

  @param rstr           CeedElemRestriction
  @param[out] blk_size  Variable to store size of blocks

  @return An error code: 0 - success, otherwise - failure

  @ref Advanced
**/
int CeedElemRestrictionGetBlockSize(CeedElemRestriction rstr,
                                    CeedInt *blk_size) {
  *blk_size = rstr->blk_size;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the multiplicity of nodes in a CeedElemRestriction

  @param rstr       CeedElemRestriction
  @param[out] mult  Vector to store multiplicity (of size l_size)

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionGetMultiplicity(CeedElemRestriction rstr,
                                       CeedVector mult) {
  int ierr;
  CeedVector e_vec;

  // Create e_vec to hold intermediate computation in E^T (E 1)
  ierr = CeedElemRestrictionCreateVector(rstr, NULL, &e_vec); CeedChk(ierr);

  // Compute e_vec = E * 1
  ierr = CeedVectorSetValue(mult, 1.0); CeedChk(ierr);
  ierr = CeedElemRestrictionApply(rstr, CEED_NOTRANSPOSE, mult, e_vec,
                                  CEED_REQUEST_IMMEDIATE); CeedChk(ierr);
  // Compute multiplicity, mult = E^T * e_vec = E^T (E 1)
  ierr = CeedVectorSetValue(mult, 0.0); CeedChk(ierr);
  ierr = CeedElemRestrictionApply(rstr, CEED_TRANSPOSE, e_vec, mult,
                                  CEED_REQUEST_IMMEDIATE); CeedChk(ierr);
  // Cleanup
  ierr = CeedVectorDestroy(&e_vec); CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief View a CeedElemRestriction

  @param[in] rstr    CeedElemRestriction to view
  @param[in] stream  Stream to write; typically stdout/stderr or a file

  @return Error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionView(CeedElemRestriction rstr, FILE *stream) {
  char stridesstr[500];
  if (rstr->strides)
    sprintf(stridesstr, "[%d, %d, %d]", rstr->strides[0], rstr->strides[1],
            rstr->strides[2]);
  else
    sprintf(stridesstr, "%d", rstr->comp_stride);

  fprintf(stream, "%sCeedElemRestriction from (%d, %d) to %d elements with %d "
          "nodes each and %s %s\n", rstr->blk_size > 1 ? "Blocked " : "",
          rstr->l_size, rstr->num_comp, rstr->num_elem, rstr->elem_size,
          rstr->strides ? "strides" : "component stride", stridesstr);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Destroy a CeedElemRestriction

  @param rstr  CeedElemRestriction to destroy

  @return An error code: 0 - success, otherwise - failure

  @ref User
**/
int CeedElemRestrictionDestroy(CeedElemRestriction *rstr) {
  int ierr;

  if (!*rstr || --(*rstr)->ref_count > 0) return CEED_ERROR_SUCCESS;
  if ((*rstr)->num_readers)
    // LCOV_EXCL_START
    return CeedError((*rstr)->ceed, CEED_ERROR_ACCESS,
                     "Cannot destroy CeedElemRestriction, "
                     "a process has read access to the offset data");
  // LCOV_EXCL_STOP
  if ((*rstr)->Destroy) {
    ierr = (*rstr)->Destroy(*rstr); CeedChk(ierr);
  }
  ierr = CeedFree(&(*rstr)->strides); CeedChk(ierr);
  ierr = CeedDestroy(&(*rstr)->ceed); CeedChk(ierr);
  ierr = CeedFree(rstr); CeedChk(ierr);
  return CEED_ERROR_SUCCESS;
}

/// @}
