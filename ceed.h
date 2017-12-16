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

#ifndef _ceed_h
#define _ceed_h

#ifdef __cplusplus
#  define CEED_EXTERN extern "C"
#else
#  define CEED_EXTERN extern
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdarg.h>
#include <stdio.h>

// We can discuss ways to avoid forcing these to be compile-time decisions, but let's leave that for later.
typedef int32_t CeedInt;
typedef double CeedScalar;

typedef struct Ceed_private *Ceed;
typedef struct CeedRequest_private *CeedRequest;
typedef struct CeedVector_private *CeedVector;
typedef struct CeedElemRestriction_private *CeedElemRestriction;
/** @brief A type representing a set of discrete basis functions that can be
    evaluated at a predefined set of quadrature points. */
typedef struct CeedBasis_private *CeedBasis;
typedef struct CeedQFunction_private *CeedQFunction;
typedef struct CeedOperator_private *CeedOperator;

CEED_EXTERN int CeedRegister(const char *prefix, int (*init)(const char *,
                             Ceed));

CEED_EXTERN int CeedInit(const char *resource, Ceed *ceed);
CEED_EXTERN int CeedErrorReturn(Ceed, const char *, int, const char *, int,
                                const char *, va_list);
CEED_EXTERN int CeedErrorAbort(Ceed, const char *, int, const char *, int,
                               const char *, va_list);
CEED_EXTERN int CeedSetErrorHandler(Ceed,
                                    int (*)(Ceed, int, const char *, va_list));
CEED_EXTERN int CeedErrorImpl(Ceed, const char *, int, const char *, int,
                              const char *, ...);
#define CeedError(ceed, ecode, ...)                                     \
  CeedErrorImpl((ceed), __FILE__, __LINE__, __func__, (ecode), __VA_ARGS__)
CEED_EXTERN int CeedDestroy(Ceed *ceed);
CEED_EXTERN int CeedCompose(int n, const Ceed *ceeds, Ceed *composed);

typedef enum {CEED_MEM_HOST, CEED_MEM_DEVICE} CeedMemType;
/* When ownership of dynamically alocated CEED_MEM_HOST pointers is transferred
   to the library (CEED_OWN_POINTER mode), they will be deallocated by calling
   the standard C library function, free(). In particular, pointers allocated
   with the C++ operator new should not be used with CEED_OWN_POINTER mode. */
typedef enum {CEED_COPY_VALUES, CEED_USE_POINTER, CEED_OWN_POINTER} CeedCopyMode;

/* The CeedVectorGet* and CeedVectorRestore* functions provide access to array
   pointers in the desired memory space. Pairing get/restore allows the Vector
   to track access, thus knowing if norms or other operations may need to be
   recomputed. */
CEED_EXTERN int CeedVectorCreate(Ceed ceed, CeedInt len, CeedVector *vec);
CEED_EXTERN int CeedVectorSetArray(CeedVector vec, CeedMemType mtype,
                                   CeedCopyMode cmode, CeedScalar *array);
CEED_EXTERN int CeedVectorGetArray(CeedVector vec, CeedMemType mtype,
                                   CeedScalar **array);
CEED_EXTERN int CeedVectorGetArrayRead(CeedVector vec, CeedMemType mtype,
                                       const CeedScalar **array);
CEED_EXTERN int CeedVectorRestoreArray(CeedVector vec, CeedScalar **array);
CEED_EXTERN int CeedVectorRestoreArrayRead(CeedVector vec,
    const CeedScalar **array);
CEED_EXTERN int CeedVectorDestroy(CeedVector *vec);

/* When CEED_REQUEST_IMMEDIATE is passed as the CeedRequest pointer to a call,
   the called function must ensure that all output is immediately available
   after it returns. In other words, the operation does not need to be executed
   asynchronously, and if it is, the called function will wait for the
   asynchronous execution to complete before returning. */
CEED_EXTERN CeedRequest *CEED_REQUEST_IMMEDIATE;
/* When CEED_REQUEST_NULL (or simply NULL) is given as the CeedRequest pointer
   to a function call, the caller is indicating that he/she will not need to
   call CeedRequestWait to wait for the completion of the operation. In general,
   the operation is expected to be executed asyncronously and its result to be
   available before the execution of next asynchronous operation using the same
   Ceed. */
#define CEED_REQUEST_NULL ((CeedRequest *)NULL)
CEED_EXTERN int CeedRequestWait(CeedRequest *req);

typedef enum {CEED_NOTRANSPOSE, CEED_TRANSPOSE} CeedTransposeMode;

/**
  @brief Create a CeedElemRestriction

  @param ceed       A Ceed object where the CeedElemRestriction will be created.
  @param nelements  Number of elements described in the @a indices array.
  @param esize      Size (number of unknowns) per element.
  @param ndof       The total size of the input CeedVector to which the
                    restriction will be applied. This size may include data
                    used by other CeedElemRestriction objects describing
                    different types of elements.
  @param mtype      Memory type of the @a indices array, see CeedMemType.
  @param cmode      Copy mode for the @a indices array, see CeedCopyMode.
  @param indices    A 2D array of dimensions (@a esize x @a nelements) using
                    column-major storage layout. Column i holds the ordered list
                    of the indices (into the input CeedVector) for the unknowns
                    corresponding to element i, where 0 <= i < @a nelements.
                    All indices must be in the range [0, @a ndof).
  @param r          The address of the variable where the newly created
                    CeedElemRestriction will be stored.

  @return An error code: 0 - success, otherwise - failure.
 */
CEED_EXTERN int CeedElemRestrictionCreate(Ceed ceed, CeedInt nelements,
    CeedInt esize, CeedInt ndof, CeedMemType mtype, CeedCopyMode cmode,
    const CeedInt *indices, CeedElemRestriction *r);

/**
  @brief Create a blocked CeedElemRestriction

  @param ceed        A Ceed object where the CeedElemRestriction will be created.
  @param nelements   Number of elements described ...
  @param esize       Size (number of unknowns) per element.
  @param blocksize   ...
  @param mtype       Memory type of the @a blkindices array, see CeedMemType.
  @param cmode       Copy mode for the @a blkindices array, see CeedCopyMode.
  @param blkindices  ...
  @param r           The address of the variable where the newly created
                     CeedElemRestriction will be stored.

  @return An error code: 0 - success, otherwise - failure.
 */
CEED_EXTERN int CeedElemRestrictionCreateBlocked(Ceed ceed, CeedInt nelements,
    CeedInt esize, CeedInt blocksize, CeedMemType mtype, CeedCopyMode cmode,
    CeedInt *blkindices, CeedElemRestriction *r);
CEED_EXTERN int CeedElemRestrictionApply(CeedElemRestriction r,
    CeedTransposeMode tmode, CeedVector u, CeedVector ru, CeedRequest *request);
CEED_EXTERN int CeedElemRestrictionDestroy(CeedElemRestriction *r);

// The formalism here is that we have the structure
//   \int_\Omega v^T f_0(u, \nabla u, qdata) + (\nabla v)^T f_1(u, \nabla u, qdata)
// where gradients are with respect to the reference element.

typedef enum {CEED_EVAL_NONE   = 0,
              CEED_EVAL_INTERP = 1, // values at quadrature points
              CEED_EVAL_GRAD   = 2, // gradients
              CEED_EVAL_DIV    = 4, // divergence
              CEED_EVAL_CURL   = 8, // curl
              CEED_EVAL_WEIGHT = 16, // quadrature weights for reference element
             } CeedEvalMode;

/** @brief Enumeration for quadrature point sets. Used also for specifying node
    locations for nodal finite element bases. */
typedef enum {
  CEED_CUSTOM_QMODE = 0,  /**< Unknown/custom quadrature. */
  CEED_GAUSS,             /**< Gauss quadrature. */
  CEED_GAUSS_LOBATTO,     /**< Gauss-Lobatto quadrature. */
} CeedQuadMode;

/** @brief Enumeration for reference geometric shapes for mesh elements. */
typedef enum {
  CEED_CUSTOM_TOPOLOGY = 0,  /**< Custom/unknown reference element type */
  CEED_POINT,       /**< Point */
  CEED_LINE,        /**< Unit line segment: [0,1] */
  CEED_TRIANGLE,    /**< Triangle with vertices: (0,0), (1,0), and (0,1) */
  CEED_QUAD,        /**< Unit square: [0,1]^2 */
  CEED_TET,         /**< Tetrahedron with vertices: (0,0,0), (1,0,0), (0,1,0),
                         and (0,0,1) */
  CEED_HEX,         /**< Unit cube: [0,1]^3 */
  CEED_NUM_TOPO     /**< This is the number of reference element types: insert
                         new types in front of it. */
} CeedTopology;

/** @brief A function that returns the dimension, @a dim, of a given
    CeedTopology, @a topo. */
CEED_EXTERN int CeedTopologyGetDimension(CeedTopology topo, CeedInt *dim);

/** @brief Enumeration for finite element basis types. Node locations, if
    applicable, are specified separately using CeedQuadMode. */
typedef enum {
  CEED_BASIS_CUSTOM = 0,  /**< User-specified basis type. */
  CEED_BASIS_LAGRANGE     /**< Nodal scalar Lagrange basis. */
} CeedBasisType;

/** @brief Allocate and zero-initialize a CeedBasis in the variable pointed to
    by @a basis_ptr, associating it with the given Ceed object, @a ceed. */
CEED_EXTERN int CeedBasisCreate(Ceed ceed, CeedBasis *basis_ptr);

/** @brief Set the reference element type, @a topo, of @a basis. */
CEED_EXTERN int CeedBasisSetElement(CeedBasis basis, CeedTopology topo);

/** @brief Set the basis type, @a btype, @a degree, and node location, @a nloc,
    (when applicable) for the given @a basis. */
CEED_EXTERN int CeedBasisSetType(CeedBasis basis, CeedBasisType btype,
                                 CeedInt degree, CeedQuadMode nloc);

/** @brief Set the quadrature rule, where the @a basis will be evaluated.

    Note that this function uses the quadrature @a order (i.e. the polynonial
    degree for which the quadrature is exact) as input: the number of quadrature
    points on the element will be determined based on @a order. */
CEED_EXTERN int CeedBasisSetQuadrature(CeedBasis basis, CeedInt qorder,
                                       CeedQuadMode qmode);

/** @brief Complete the construction of the CeedBasis object, @a basis.

    This function uses the currently set parameters (element type, basis
    parameters, quadrature parameters) to complete the construction of the
    object on both the host and the device (from the associated Ceed object).
    Generally, it will invoke the backend (from the associated Ceed object) to
    perform the object setup on the device. */
CEED_EXTERN int CeedBasisComplete(CeedBasis basis);

/** @brief Construct a CeedBasis for a scalar tensor-product basis in 1D
    (CEED_LINE), 2D (CEED_QUAD), or 3D (CEED_HEX).

    @param[in]  ceed    Ceed object to associate with the new CeedBasis
    @param[in]  dim     Reference element dimension
    @param[in]  ncomp   Number of components; TODO: remove this
    @param[in]  degree  Polynomial degree of the basis
    @param[in]  Q       Number of quadrature points in 1D, used for all spatial
                        dimensions
    @param[in]  qmode   Quadrature mode; CEED_CUSTOM_QMODE can not be used here
    @param[out] basis   The address of the output CeedBasis variable

    @return An error code: 0 - success, otherwise - failure.
*/
CEED_EXTERN int CeedBasisCreateTensorH1Lagrange(Ceed ceed, CeedInt dim,
    CeedInt ncomp, CeedInt degree, CeedInt Q, CeedQuadMode qmode, CeedBasis *basis);

/** @brief Construst a CeedBasis for a scalar tensor-product basis in 1D
    (CEED_LINE), 2D (CEED_QUAD), or 3D (CEED_HEX).

    @note This constructor will copy the content of the input arrays internally.

    @param[in]  ceed       Ceed object to associate with the new CeedBasis
    @param[in]  dim        Reference element dimension
    @param[in]  ncomp      Number of components; TODO: remove this
    @param[in]  P1d        Number of degrees of freedom, i.e. number of basis
                           functions, in 1D
    @param[in]  Q1d        Number of quadrature points in 1D, used for all
                           spatial dimensions
    @param[in]  interp1d   Interpolation matrix with dimensions @a Q1d x @a P1d,
                           using column-major layout. It represents the values
                           of the 1D basis functions at the 1D quadrature
                           points.
    @param[in]  grad1d     Interpolation matrix with dimensions @a Q1d x @a P1d,
                           using column-major layout. It represents the
                           derivatives of the 1D basis functions at the 1D
                           quadrature points.
    @param[in]  qref1d     Coordinates of the 1D quadrature points: array of
                           size @a Q1d
    @param[in]  qweight1d  1D quadrature point weights: array of size @a Q1d
    @param[out] basis      The address of the output CeedBasis variable

    @return An error code: 0 - success, otherwise - failure.
*/
CEED_EXTERN int CeedBasisCreateTensorH1(Ceed ceed, CeedInt dim, CeedInt ncomp,
                                        CeedInt P1d, CeedInt Q1d, const CeedScalar *interp1d, const CeedScalar *grad1d,
                                        const CeedScalar *qref1d, const CeedScalar *qweight1d, CeedBasis *basis);

/** @brief Construct a CeedBasis for a scalar, generic basis.

    The constructed object represents a basis of scalar functions that do not
    have any special structure, or the associated a qudrature rule does not
    allow any special structure of the basis to be utilized for fast evaluation.

    @param[in]  ceed     Ceed object to associate with the new CeedBasis
    @param[in]  dim      Reference element dimension
    @param[in]  ndof     Number of degrees of freedom, i.e. number of basis
                         functions
    @param[in]  nqpt     Number of quadrature points
    @param[in]  interp   Interpolation matrix with dimensions @a nqpt x @a ndof,
                         using column-major layout. It represents the values of
                         the basis functions at the quadrature points.
    @param[in]  grad     Interpolation rank 3 tensor with dimensions @a nqpt x
                         @a dim x @a ndof, using column-major layout. It
                         represents the partial derivatives of the basis
                         functions at the quadrature points.
    @param[in]  qweights Quadrature point weights: array of size @a nqpt
    @param[out] basis    The address of the output CeedBasis variable

    @return An error code: 0 - success, otherwise - failure.
*/
CEED_EXTERN int CeedBasisCreateScalarGeneric(Ceed ceed, CeedInt dim,
    CeedInt ndof, CeedInt nqpt, const CeedScalar *interp, const CeedScalar *grad,
    const CeedScalar *qweights, CeedBasis *basis);

/** @brief Write text information about @a basis to the given @a stream.

    @note Generally, the written information will not be sufficient to
    reconstruct a copy of @a basis, by reading it back. */
CEED_EXTERN int CeedBasisView(CeedBasis basis, FILE *stream);

/** @brief Apply a basis transformation between degrees of freedom and
    quadrature point data.

    @param[in]  basis  The CeedBasis to use.

    @param[in]  tmode  In CEED_NOTRANSPOSE mode, transform degrees of freedom to
                       quadrature point data; in CEED_TRANSPOSE mode, perform
                       the transpose operation from quadrature point data to
                       degrees of freedom.

    @param[in]  emode  Evaluation mode: a bitwise-or of constants from the
                       enumeration CeedEvalMode, describing what quadrature data
                       is given as input, or expected as output; input or output
                       depends on the @a tmode argument.

    @param[in]  u      When the input represents degrees of freedom, the array
                       @a u must contain the coefficients multiplying each
                       basis function.

                       When the input represents quadrature point data, the data
                       layout is a concatenation of arrays corresponding to the
                       specified CeedEvalMode, @a emode, ordered as defined in
                       the enumeration.

                       The data for CEED_EVAL_INTERP is an array of size nqpt
                       (number of quadrature point). If the basis is a vector
                       basis, the data is an array with dimensions nqpt x vdim,
                       using column-major layout, where vdim is the number of
                       vector components of all basis functions.

                       The data for CEED_EVAL_GRAD is an array with dimensions
                       nqpt x dim, using column-major layout, where dim is the
                       dimension of the reference element space. For vector
                       bases, a third array dimension is added of size vdim: the
                       number of vector components of all basis functions.

                       The data for CEED_EVAL_DIV is an array of size nqpt.

                       The data for CEED_EVAL_CURL is an array with dimensions
                       nqpt x 3 in 3D, and nqpt x 1 in 2D.

                       The data for CEED_EVAL_WEIGHT is an array of size nqpt.

                       TODO: add a parameter that allows this function to
                       perform the transformations on multiple elements, or
                       multiple vector field component at the same time. Such a
                       parameter can be used for vectorization of the
                       implementation.

    @param[out] v      Output data; uses the same data layouts as the input
                       data, @a u.

    @return An error code: 0 - success, otherwise - failure.
*/
CEED_EXTERN int CeedBasisApply(CeedBasis basis, CeedTransposeMode tmode,
                               CeedEvalMode emode, const CeedScalar *u, CeedScalar *v);

/** Destroy a CeedBasis object. */
CEED_EXTERN int CeedBasisDestroy(CeedBasis *basis);

/** @brief Helper function for constructing 1D Gauss quadrature. */
CEED_EXTERN int CeedGaussQuadrature(CeedInt Q, CeedScalar *qref1d,
                                    CeedScalar *qweight1d);
/** @brief Helper function for constructing 1D Gauss-Lobatto quadrature. */
CEED_EXTERN int CeedLobattoQuadrature(CeedInt Q, CeedScalar *qref1d,
                                      CeedScalar *qweight1d);

CEED_EXTERN int CeedQFunctionCreateInterior(Ceed ceed, CeedInt vlength,
    CeedInt nfields, size_t qdatasize, CeedEvalMode inmode, CeedEvalMode outmode,
    int (*f)(void *ctx, void *qdata, CeedInt nq, const CeedScalar *const *u,
             CeedScalar *const *v), const char *focca, CeedQFunction *qf);
CEED_EXTERN int CeedQFunctionSetContext(CeedQFunction qf, void *ctx,
                                        size_t ctxsize);
CEED_EXTERN int CeedQFunctionApply(CeedQFunction qf, void *qdata, CeedInt Q,
                                   const CeedScalar *const *u,
                                   CeedScalar *const *v);
CEED_EXTERN int CeedQFunctionDestroy(CeedQFunction *qf);

CEED_EXTERN int CeedOperatorCreate(Ceed ceed, CeedElemRestriction r,
                                   CeedBasis b, CeedQFunction qf, CeedQFunction dqf, CeedQFunction dqfT,
                                   CeedOperator *op);
CEED_EXTERN int CeedOperatorGetQData(CeedOperator op, CeedVector *qdata);
CEED_EXTERN int CeedOperatorApply(CeedOperator op, CeedVector qdata,
                                  CeedVector ustate, CeedVector residual, CeedRequest *request);
CEED_EXTERN int CeedOperatorApplyJacobian(CeedOperator op, CeedVector qdata,
    CeedVector ustate, CeedVector dustate, CeedVector dresidual,
    CeedRequest *request);
CEED_EXTERN int CeedOperatorDestroy(CeedOperator *op);

static inline CeedInt CeedPowInt(CeedInt base, CeedInt power) {
  CeedInt result = 1;
  while (power) {
    if (power & 1) result *= base;
    power >>= 1;
    base *= base;
  }
  return result;
}

#endif
