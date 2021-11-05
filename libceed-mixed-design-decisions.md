# libCEED mixed precision design working document

## New and proposed types

### CeedScalarType

`CeedScalarType` was added when single precision support was added to the library.
It is an enum currently defined as:

```
typedef enum {
  /// Single precision
  CEED_SCALAR_FP32,
  /// Double precision
  CEED_SCALAR_FP64
} CeedScalarType;
```

The expectation is that we can expand to other precision types in the future. 
`CEED_SCALAR_TYPE` is defined to be a member of `CeedScalarType` matching 
`CeedScalar` based on which precision-specific header is included in `ceed.h`.

### CeedSalarArray (proposed)

A new data struct for holding the same data in potentially multiple precisions
at once, as well as possibly being a general container for data of different precisions
in function prototypes, is proposed:

```
typedef struct CeedScalarArrayObj {
  size_t length;
  double *double_data;
  float *float_data;
} CeedScalarArray;
```

***Discussion point: should a `CeedScalarArray` contain its own length data, if
it will be used by `CeedVector` as well, which has length as an interface-level
(not backend-dependent) variable?***  This would likely depend on whether
or not we ever intend `CeedScalarArray` to have wider use than just within
backends that have enabled mixed precision, and also:
***Discussion point: do we want users to be able to interact with `CeedScalarArray`
objects when getting/setting data from `CeedVector`s -- would there be a use case
for this?***

For more about potentially using this data struct in `CeedVector`, see the next section.


##  Design of CeedVector

###  Background

First, let's recall the functions for interacting with a `CeedVector`'s data
directly to/from a `CeedScalar` array, used both by libCEED backends and user
code:

```
Set the array used by a CeedVector, freeing any previously allocated
           array if applicable. The backend may copy values to a different
           memtype, such as during CeedOperatorApply.
CEED_EXTERN int CeedVectorSetArray(CeedVector vec, CeedMemType mem_type,
                                   CeedCopyMode copy_mode, CeedScalar *array);

Take ownership of the CeedVector array and remove the array from the
           CeedVector. The caller is responsible for managing and freeing
           the array.
CEED_EXTERN int CeedVectorTakeArray(CeedVector vec, CeedMemType mem_type,
                                    CeedScalar **array);

Get read/write access to a CeedVector via the specified memory type.
CEED_EXTERN int CeedVectorGetArray(CeedVector vec, CeedMemType mem_type,
                                   CeedScalar **array);

Get read-only access to a CeedVector via the specified memory type.
CEED_EXTERN int CeedVectorGetArrayRead(CeedVector vec, CeedMemType mem_type,
                                       const CeedScalar **array);

Restore an array obtained using CeedVectorGetArray. (Note that 
*array is set to NULL, and the vector state is updated; this happens
in the interface-level code, and the backends do not do anything but 
return CEED_ERROR_SUCCESS.)
CEED_EXTERN int CeedVectorRestoreArray(CeedVector vec, CeedScalar **array);


Restore an array obtained using CeedVectorGetArrayRead. (Note that 
*array is set to NULL, and the number of readers is decremented; this 
happens in the interface-level code, and the backends do not do anything 
but return CEED_ERROR_SUCCESS.)
CEED_EXTERN int CeedVectorRestoreArrayRead(CeedVector vec,
                                           const CeedScalar **array);
```

### Proposed rules for the vector access in mixed precision

We will introduce a new `CeedVector` member, the *preferred precision*, which can be get/set
at any point in a vector's lifetime.  E.g., the preferred precision can be set
before ever filling the vector with any data, or it can be set later, at which point
it would trigger a conversion to the new precision.  ***Discussion point: should this
be considered read or read/write access as far as whether or not the original data should
be deleted/cleared?***  This could be an interface-level parameter or a backend-specific
parameter (part of a private data struct in a backend).

We have the following proposed rules for allowing multiple precision access to 
`CeedVector`s:

 - `Read` access (`GetArrayRead`) will convert if necessary to a new precision, but will not 
    delete/clear the data from the original precision. If more than one precision is currently
    active (in the future, if we expand to more than two precision options) and a third precision
    is requested, the *preferred precision* will be used as the basis for copying/converting to the new precision.

 - `Read/write` access (`GetArray`) will convert if necessary, and *will* delete/clear the 
    original data. The *preferred precision* will be updated to the new precision.

 - `Take` access will convert if necessary and delete/clear all data from the `CeedVector`.
    *Preferred precision* will remain unchanged, since the `CeedVector` no longer contains any data. 

    Following this access pattern, if two (or more) precisions in a `CeedVector`
    are active at the same time, we can conclude that they are the same data, but in different
    precisions, and both are still "valid" representations of the `CeedVector` data. 

Note that we are assuming all backends will keep the distinction between non-owned data and data
allocated/owned by the vector.  Thus, any data conversion will result in allocating the corresponding
memory in the `allocated` object, even if the data being converted was not owned by the vector. 

***Discussion point: SetArray behavior***

We must first define exactly what we mean by `SetArray` in this context. One way to interpret it is
"use exactly this data," when coupled with the `CEED_USE_POINTER` or `CEED_OWN_POINTER` 
options.  A possible consequence of such an interpretation would be to omit any "requested precision" parameter
from the multiprecision version of `SetArray`, indicating that the `CeedVector` should update itself to 
use the given data, in whichever precision it is -- or the multiple precisions it is in, if the general
interface uses `CeedScalarArray *` to pass the data (see discussion on options for general interface below).
One could argue this is the direct correlation to the `CeedScalar` (current) case.  However, we could 
also add a preferred precision argument to this function, as with the other multiprecision versions, 
and have the `CeedVector` convert the input data as necessary.  This could, however, blur the lines 
between the `modes` of the function, as a `CEED_USE_POINTER` mode might still end up allocating and 
copying/casting data as in the `CEED_COPY_VALUES` case.

Another possible difficulty in not being able to set the data directly through pointers could
be in the common use case of passive input data which will be used in the QFunction, resulting in 
unnecessary repeated data conversions; see further 
discussion in options for general interface surrounding the use of various pointer options for data.

Currently, `CeedVectorSetArray` will allocate new memory if `array` is NULL and the mode is 
`CEED_COPY_VALUES`.  In a multiprecision `CeedVector`, this would correspond to either allocating
memory in the vector's current preferred precision (in the version where `SetArray` doesn't have 
a requested/preferred precision parameter) or the requested precision.

A final note about the behavior of `SetArray` is that of the preferred precision.  If we can 
set data with more than one precision currently active to the `CeedVector` at the same time (taking
a `CeedScalarArray` parameter as input, but without a preferred precision parameter), how would 
the `CeedVector` know which precision it should consider its preferred precision moving foward? (Note 
that if `SetArray` is called through a general interface with `void *`, we would of course
have to provide information about the type of the data in the `void *`, but this could be used as the 
preferred precision or in addition to a preferred precision parameter, if we want the routine to do
automatic conversion.)


***Discussion point: multiple versions of `GetArray`/read/write access?***
Would we want to add another option for the case where we are getting read/write access *only* to overwrite
the data, and do not actually care about the current values ("pure write" access)? 

In the current `CeedScalar` setup, the distinction doesn't matter, since we are only getting a pointer; 
in the multiprecision context, we may end up allocating data and converting the current data to the new requested precision, even 
though we only want a pointer to data in the new precision for purposes of writing.

However, perhaps we would not often be changing the precision of a vector (requesting in a new precision after
previous data set/data read in another precision) unless we *do* care about the current values?


### Other CeedVector routines with CeedScalar arguments

We may eventually want versions of some or all of the following routines in all precisions
supported by libCEED (in the prototype, not just in whatever the backend implementations
do based on the current preferred precision of the vector):

```
CeedVectorSetValue(CeedVector vec, CeedScalar value)

CeedVectorNorm(CeedVector vec, CeedNormType norm_type, CeedScalar *norm)

CeedVectorScale(CeedVector x, CeedScalar alpha)

CeedVectorAXPY(CeedVector y, CeedScalar alpha, CeedVector x)
```
 

### Specific implementation modifications to CeedVector

There are several options for how to incorporate multiple precisions in `CeedVector` objects
in backends supporting mixed precisions.  Some ideas are: 

 - A. The private backend data for the `CeedVector` will contain `CeedScalarArray` objects
   in the place of the current `CeedScalar *` objects for holding data.  The `array`/`array_allocated`
   distinction will remain for managing which data is owned by the `CeedVector`.  GPU backends
   will maintain `h_array`/`d_array` for host and device data. 

 - B. The private backend data for the `CeedVector` will contain `CeedScalarArray *` pointers
   in the place of the current `CeedScalar *` objects for holding the data. Everything else is
   the same as A.

 - C. The private backend data for the `CeedVector` will store all arrays "flat" -- as in, adding
   `h_array_{precision}` / `h_array_allocated_{precision}` and `d_array_{precision}` / 
   `d_array_allocated_{precision}` pointers for every precision currently allowed by libCEED. 

- D. Some other option?

The benefit of (A) and (B) over (C) is that the data is nicely encapsulated in the `CeedScalarArray` struct,
making it clear that any data currently active in a particular `CeedScalarArray` should be the same data
in different precisions (though host/device arrays may currently have different data, depending on sync 
status). It may also facilitate expanding to other precisions in the future with less code duplication and 
fewer different data pointers in the private backend struct. "The data" will still be in something named `{h/d_}array` /
 `{h/d_}array_allocated`.

A benefit of (A) over (B) is that the individual pointers in the `CeedScalarArray` objects will be initialized/set to 
NULL by the standard call `ierr = CeedCalloc(1, &data);` in vector creation, and `Get/Set` functions
that check for the existence/validity of the pointers in the `CeedScalarArray` objects will always be able
to check without memory errors. With (B), we would need to `calloc` each `array` member `{h/d_}array{_allocated}`
 before we could check for any valid data, which could be prone to memory leaks/errors (indeed, when I first
tried this implementation, I had trouble tracking down all the segfaults from double free errors). 

A benefit of (B) over (A) is that we could use `SetArray` in the multiprecision case to still *actually*
directly set the vector to use the provided pointer, in the case of a general interface with `CeedScalarArray`
rather than `void *`. In (A), `SetArray` + `USE_POINTER` would have to result in simply setting the pointers
inside the `CeedVector`'s `CeedScalarArray` objects to those in the parameter `CeedScalarArray, rather than 
setting a pointer to the entire `CeedScalarArray`
object itself, meaning that any future conversion/copying done in the `CeedVector` will not also carry through
to the source of the data.  E.g.:

>
> **Example use case: qdata**
>
> Imagine we have an input to our operator containing qdata.  This operator will 
> perform its QFunction in single precision, but the qdata is in double precision
> after being calculated as the output from a setup operator.  
> In the current setup for operators, if the E-vectors/restrictions for the operator
> are also in double (since only QFunction will be in single), the qdata will first
> be accessed by a `read` call in `FP64`, either as input for an element
> restriction, or to directly set the `edata` for this input.  When `SetArray` is called
> to set the `edata` to the `qvec` -- in `FP32`, the data conversion will only happen
> in the operator's `qvec`, not in the original `qdata` vector.  Thus, the next time we
> apply the operator, we will convert again, though if we had stored the coverted data
> in the `qdata` vector, this would be unnecessary.
>
> This could potentially be avoided by modifying the operator code to directly set
> the data from an input L-vector to the input Q-vector (without an `edata` intermediate
> step) using the Q-vector's precision, in a `CEED_STRIDES_BACKEND`/`NONE` basis scenario. 
> This would further increase the importance of using `CEED_STRIDES_BACKEND` when possible,
> or making sure that backends check to see if the strides match those of the backend, even
> if `CEED_STRIDES_BACKEND` was not set explicitly by the user.

A benefit of (C) is the ease of memory management of (A).  (C) would also work well with a "non-`CeedScalarArray`"
general interface (using `void *`) instead. A drawback is the "messiness" of needing 8 pointers for GPU backends,
growing by 4 every time a new precision is added in the future.  The same issues with possible unnecessary
reconversion of passive input data could occur here, depending on how the `edata` inside an operator
is configured (e.g., if using `CeedScalarArray **` in order to allow E-Vectors in different precisions). 
 
***Discusion point: sync behavior***

For the GPU backends, we must determine how `CeedVectorSyncArray` should handle the possibility of multiple
precisions being active.  For example, the sync function could sync/copy all currently active 
precisions on the host or device, or it could sync only the vector's current preferred precision, 
or we could have multiple versions of sync function.  

Syncing all current data is likely the best way to avoid pitfalls, though it could result
in unnecessary data movement/copying. 

###  Options for adding precision to get/set functions

***Discussion point: Public API vs internal use***

It seems likely that we would want to have a general set of functions which can be called 
in places (e.g. inside other backend functions) where the other top-level code could
be agnostic to the datatype (at least when using some parameter from the `CeedScalarType` enum), 
namely in operator-level functions, though perhaps for other objects as well.  For example, 
a common use pattern across backend objects is to get read or read/write access to a vector's 
data in order to use it directly in an object's action or 
pass it as a parameter to a kernel:

>
> **CODE EXAMPLE: preparing an element restriction call in the cuda-ref backend**
>
```
// Get vectors
const CeedScalar *d_u;
CeedScalar *d_v;
ierr = CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, &d_u); CeedChkBackend(ierr);
ierr = CeedVectorGetArray(v, CEED_MEM_DEVICE, &d_v); CeedChkBackend(ierr); 
(&d_u and &d_v passed as kernel args)
```

But a more complicated case is the use inside of operators, e.g. when getting/setting E-Vector
data as part of a `CeedOperator` application. 

The following backends use a data type `CeedScalar **`
inside their private backend operator data structs for E-Vector data: `ref`, `blocked`, `opt`, `cuda-ref` (and thus by 
delegation, `cuda-shared` and `magma`), `hip-ref` (and `hip-shared`, `magma`). 

What if not all E-vectors should have same precision, e.g. a passive input with a basis none 
that will be used in single precision in the QFunction?  Or even if we do want all E-vectors
to be in the same precision, we can't just use `CeedScalar **` as the type in the backend
data structs.  Using `CeedScalarArray **` instead would allow the get/set interactions
of internal E-vectors and Q-vectors with the "edata" to proceed as before.

Whether or not we also
want these generic multiprecision functions to be available as part of the API, in case users may also wish to 
have a polymorphic-ish interface to `CeedVector`s, is an open question that could determine
the other choices. 

A backend would likely want to have internal implementations for different precisions, to which the
general version of the function would dispatch.  However, we could potentially leave the 
general interface set of functions as the only ones that get added as new backend functions for the `CeedVector`, 
so that expanding to more precisions in the future wouldn't mean continually adding new 
function pointers to the `CeedVector` object and adding more "not implemented/error" functions for
every backend that doesn't support mixed precisions (unless the plan is for all backends to eventually 
support it?). We could/almost surely will also add specific interface-level functions with different names for each
precision in the public API -- but should these also be backend functions, or should they call the general
versions with specific parameters set?  Performance considerations?

####  General interface with CeedScalarArray

One idea would be for any backend implementing mixed-precision functionality to add
an additional set of backend functions for `CeedVector` objects:

```
CEED_EXTERN int CeedVectorSetArrayObj(CeedVector vec, CeedMemType mem_type,
                                      CeedCopyMode copy_mode,
                                      CeedScalarType prec,
                                      CeedScalarArray *array);
CEED_EXTERN int CeedVectorTakeArrayObj(CeedVector vec, CeedMemType mem_type,
                                       CeedScalarType prec,
                                       CeedScalarArray **array);
CEED_EXTERN int CeedVectorGetArrayObj(CeedVector vec, CeedMemType mem_type,
                          CeedScalarType prec, CeedScalarArray **array);
CEED_EXTERN int CeedVectorGetArrayObjRead(CeedVector vec, CeedMemType mem_type,
                           CeedScalarType prec, const CeedScalarArray **array);
CEED_EXTERN int CeedVectorRestoreArrayObj(CeedVector vec, CeedScalarArray **array);
CEED_EXTERN int CeedVectorRestoreArrayObjRead(CeedVector vec,
                                              const CeedScalarArray **array);
```

which are the same prototypes as for the current/`CeedScalar`-based functions, 
but with the `array` datatype changed to `CeedScalarArray`.  

 - pro: could be used at API level if desired (no `void *` for potential issues with 
Rust/Julia etc. interfaces).  Very similar to the `CeedScalar` version.  Could easily 
set an array with multiple precisions currently active, if such a use case would arise?

 - con: maybe too complicated? Annoying way to interact with it in setup for `Apply` of objects -- 
 E.g., consider the previous code example for the element restriction in the cuda-ref backend,
but needing to initialize a `CeedScalarArray` object before passing it to the function because
the function expects to be able to check which pointers inside the `CeedScalarArray` are currently
valid:

> 
> **CODE EXAMPLE: double pointers and CeedScalarArray**
> 
```
// Get vectors
CeedScalarArray d_u_i;
const CeedScalarArray *d_u = &d_u_i;
CeedScalarArray d_v_i;
CeedScalarArray *d_v = &d_v_i;
ierr = CeedVectorGetArrayObjRead(u, CEED_MEM_DEVICE, CEED_SCALAR_FP64, &d_u); CeedChkBackend(ierr);
ierr = CeedVectorGetArrayObj(v, CEED_MEM_DEVICE, CEED_SCALAR_FP64, &d_v); CeedChkBackend(ierr);

(&d_u->double_data and &d_v->double_data are passed as kernel args)
```

However, if we use a single pointer in the prototype instead, we cannot have the `array` be `const`
in `GetArrayObjRead`. 


#### General interface with void pointers


Another possibility would be to have a general interface with `void` pointers and requested precision type. 

```
CEED_EXTERN int CeedVectorSetArrayUntyped(CeedVector vec, CeedMemType mem_type,
                                      CeedCopyMode copy_mode,
                                      CeedScalarType prec,
                                      void *array);
CEED_EXTERN int CeedVectorTakeArrayUntyped(CeedVector vec, CeedMemType mem_type,
                                       CeedScalarType prec,
                                       void **array);
CEED_EXTERN int CeedVectorGetArrayUntyped(CeedVector vec, CeedMemType mem_type,
                          CeedScalarType prec, void **array);
CEED_EXTERN int CeedVectorGetArrayReadUntyped(CeedVector vec, CeedMemType mem_type,
                           CeedScalarType prec, const void **array);
CEED_EXTERN int CeedVectorRestoreArrayUntyped(CeedVector vec, void **array);
CEED_EXTERN int CeedVectorRestoreArrayReadUntyped(CeedVector vec,
                                                  const void **array);
```

- pro: No need for intializing `CeedScalarArray` for use 

- con: cannot be used in public API (if we want a general version to be available).  May be harder 
to use in a type-agnostic way if casts are required?  (Need to think more about this)


## Other potential uses for CeedScalarArray

As mentioned previously, it could be used in `edata` for storing E-vector data inside private backend
data for operators.  It could also potentially be used by `Basis` objects for storing basis data
in the backend data, as well. 

## Proposed behavior for other libCEED objects

The computational components of a libCEED operator -- Basis and QFunction objects -- 
could have their working/compute precision set via a `SetPrecision` function prior to 
adding them to an operator. This would allow very fine-tuned control.  

A question is whether the user should be able to change the precision of an object during
its lifetime, or create two separate objects.  A strong argument could be made for being 
able to change the precision during the lifetime of the object, allowing operators to 
change computation precision inside a solve. 

Regardless of the level of control over individual operator components exposed to users, 
we will also want to provide some operator-level options which would be very simple to use.

***Discussion point: what should the operator-level controls look like? Should they be tied
to the operator itself (properties that can be set and reset), or tied to Apply calls?***

A final question is whether there should be operator-level controls for internal storage precisions,
e.g. the case where we may wish to store E- and/or Q-Vectors in a lower precision to save on memory
movement, but perform basis or QFunction computations in a higher precision.  (It seems that
were this feature added in the future, it would make more sense to tie it to the operator itself,
as that is where the E- and Q-Vectors exist if they are formed, rather than the individual Basis 
or QFunction objects.)

Depending on the level of fine-tuned control allowed to users (vs precision being set only
internally by the operators), we may have to add checking routines to make sure the user
is trying to create a compatible combination of basis, QFunction, and vector precisions.

## Limitations on precision combinations

***Discussion point: We should determine what limitations will be set on possible combinations of
vector and computation precisions.***

These rules can be either short term (we will place this limitation now, to help manage the scope involved
in rolling out the first official mixed-precision capability) or long term (we plan to keep this limit
for the foreseeable future, until a good case for its use arises).

Proposed rules:

 - All inputs/outputs to a QFunction will be of the same precision, which may or may not be the same as `CeedScalar`.
 
 - Should all internal E-Vectors need to be the same precision? (Expected use case: input with basis action `NONE`, in which
   case the E-Vector is also the Q-Vector and may need to be in the precision of the QFunction.  This would require
   changes to the way operators store `edata`, as mentioned above.)

 - Should all L-Vector inputs/outputs be required to be the same precision? (Preferred precision)
 


