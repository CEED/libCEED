# Developer Notes

## Style Guide

Please check your code for style issues by running

`make format`

In addition to those automatically enforced style rules, libCEED tends to follow the following code style conventions:

- Variable names: `snake_case`
- Strut members: `snake_case`
- Function and method names: `PascalCase` or language specific style
- Type names: `PascalCase` or language specific style
- Constant names: `CAPS_SNAKE_CASE` or language specific style

Also, documentation files should have one sentence per line to help make git diffs clearer and less disruptive.

## Clang-tidy

Please check your code for common issues by running

`make tidy`

which uses the `clang-tidy` utility included in recent releases of Clang.
This tool is much slower than actual compilation (`make -j8` parallelism helps).
To run on a single file, use

`make interface/ceed.c.tidy`

for example.
All issues reported by `make tidy` should be fixed.

## Include-What-You-Use

Header inclusion for source files should follow the principal of 'include what you use' rather than relying upon transitive `#include` to define all symbols.

Every symbol that is used in the source file `foo.c` should be defined in `foo.c`, `foo.h`, or in a header file `#include`d in one of these two locations.
Please check your code by running the tool [`include-what-you-use`](https://include-what-you-use.org/) to see recommendations for changes to your source.
Most issues reported by `include-what-you-use` should be fixed; however this rule is flexible to account for differences in header file organization in external libraries.
If you have `include-what-you-use` installed in a sibling directory to libCEED or set the environment variable `IWYU_CC`, then you can use the makefile target `make iwyu`.

Header files should be listed in alphabetical order, with installed headers preceding local headers and `ceed` headers being listed first.
The `ceed-f64.h` and `ceed-f32.h` headers should only be included in `ceed.h`.

```c
#include <ceed.h>
#include <ceed/backend.h>
#include <stdbool.h>
#include <string.h>
#include "ceed-avx.h"
```

## Shape

Backends often manipulate tensors of dimension greater than 2.
It is awkward to pass fully-specified multi-dimensional arrays using C99 and certain operations will flatten/reshape the tensors for computational convenience.
We frequently use comments to document shapes using a lexicographic ordering.
For example, the comment

```c
// u has shape [dim, num_comp, Q, num_elem]
```

means that it can be traversed as

```c
for (d=0; d<dim; d++)
  for (c=0; c<num_comp; c++)
    for (q=0; q<Q; q++)
      for (e=0; e<num_elem; e++)
        u[((d*num_comp + c)*Q + q)*num_elem + e] = ...
```

This ordering is sometimes referred to as row-major or C-style.
Note that flattening such as

```c
// u has shape [dim, num_comp, Q*num_elem]
```

and

```c
// u has shape [dim*num_comp, Q, num_elem]
```

are purely implicit -- one just indexes the same array using the appropriate convention.

## `restrict` Semantics

QFunction arguments can be assumed to have `restrict` semantics.
That is, each input and output array must reside in distinct memory without overlap.

## CeedVector Array Access Semantics

Backend implementations are expected to separately track 'owned' and 'borrowed' memory locations.
Backends are responsible for freeing 'owned' memory; 'borrowed' memory is set by the user and backends only have read/write access to 'borrowed' memory.
For any given precision and memory type, a backend should only have 'owned' or 'borrowed' memory, not both.

Backends are responsible for tracking which memory locations contain valid data.
If the user calls {c:func}`CeedVectorTakeArray` on the only memory location that contains valid data, then the {ref}`CeedVector` is left in an *invalid state*.
To repair an *invalid state*, the user must set valid data by calling {c:func}`CeedVectorSetValue`, {c:func}`CeedVectorSetArray`, or {c:func}`CeedVectorGetArrayWrite`.

Some checks for consistency and data validity with {ref}`CeedVector` array access are performed at the interface level.
All backends may assume that array access will conform to these guidelines:

- Borrowed memory

  - {ref}`CeedVector` access to borrowed memory is set with {c:func}`CeedVectorSetArray` with `copy_mode = CEED_USE_POINTER` and revoked with {c:func}`CeedVectorTakeArray`.
    The user must first call {c:func}`CeedVectorSetArray` with `copy_mode = CEED_USE_POINTER` for the appropriate precision and memory type before calling {c:func}`CeedVectorTakeArray`.
  - {c:func}`CeedVectorTakeArray` cannot be called on a vector in a *invalid state*.

- Owned memory

  - Owned memory can be allocated by calling {c:func}`CeedVectorSetValue` or by calling {c:func}`CeedVectorSetArray` with `copy_mode = CEED_COPY_VALUES`.
  - Owned memory can be set by calling {c:func}`CeedVectorSetArray` with `copy_mode = CEED_OWN_POINTER`.
  - Owned memory can also be allocated by calling {c:func}`CeedVectorGetArrayWrite`.
    The user is responsible for manually setting the contents of the array in this case.

- Data validity

  - Internal syncronization and user calls to {c:func}`CeedVectorSync` cannot be made on a vector in an *invalid state*.
  - Calls to {c:func}`CeedVectorGetArray` and {c:func}`CeedVectorGetArrayRead` cannot be made on a vector in an *invalid state*.
  - Calls to {c:func}`CeedVectorSetArray` and {c:func}`CeedVectorSetValue` can be made on a vector in an *invalid state*.
  - Calls to {c:func}`CeedVectorGetArrayWrite` can be made on a vector in an *invalid* state.
    Data syncronization is not required for the memory location returned by {c:func}`CeedVectorGetArrayWrite`.
    The caller should assume that all data at the memory location returned by {c:func}`CeedVectorGetArrayWrite` is *invalid*.

## Internal Layouts

Ceed backends are free to use any **E-vector** and **Q-vector** data layout, to include never fully forming these vectors, so long as the backend passes the `t5**` series tests and all examples.
There are several common layouts for **L-vectors**, **E-vectors**, and **Q-vectors**, detailed below:

- **L-vector** layouts

  - **L-vectors** described by a {ref}`CeedElemRestriction` have a layout described by the `offsets` array and `comp_stride` parameter.
    Data for node `i`, component `j`, element `k` can be found in the **L-vector** at index `offsets[i + k*elem_size] + j*comp_stride`.
  - **L-vectors** described by a strided {ref}`CeedElemRestriction` have a layout described by the `strides` array.
    Data for node `i`, component `j`, element `k` can be found in the **L-vector** at index `i*strides[0] + j*strides[1] + k*strides[2]`.

- **E-vector** layouts

  - If possible, backends should use {c:func}`CeedElemRestrictionSetELayout()` to use the `t2**` tests.
    If the backend uses a strided **E-vector** layout, then the data for node `i`, component `j`, element `k` in the **E-vector** is given by `i*layout[0] + j*layout[1] + k*layout[2]`.
  - Backends may choose to use a non-strided **E-vector** layout; however, the `t2**` tests will not function correctly in this case and the tests will need to be whitelisted for the backend to pass the test suite.

- **Q-vector** layouts

  - When the size of a {ref}`CeedQFunction` field is greater than `1`, data for quadrature point `i` component `j` can be found in the **Q-vector** at index `i + Q*j`.
    Backends are free to provide the quadrature points in any order.
  - When the {ref}`CeedQFunction` field has `emode` `CEED_EVAL_GRAD`, data for quadrature point `i`, component `j`, derivative `k` can be found in the **Q-vector** at index `i + Q*j + Q*size*k`.
  - Note that backend developers must take special care to ensure that the data in the **Q-vectors** for a field with `emode` `CEED_EVAL_NONE` is properly ordered when the backend uses different layouts for **E-vectors** and **Q-vectors**.

## Backend Inheritance

There are three mechanisms by which a Ceed backend can inherit implementation from another Ceed backend.
These options are set in the backend initialization routine.

1. Delegation - Developers may use {c:func}`CeedSetDelegate()` to set a backend that will provide the implementation of any unimplemented Ceed objects.
2. Object delegation  - Developers may use {c:func}`CeedSetObjectDelegate()` to set a backend that will provide the implementation of a specific unimplemented Ceed object.
   Object delegation has higher precedence than delegation.
3. Operator fallback - Developers may use {c:func}`CeedSetOperatorFallbackResource()` to set a {ref}`Ceed` resource that will provide the implementation of unimplemented {ref}`CeedOperator` methods.
   A fallback {ref}`Ceed` with this resource will only be instantiated if a method is called that is not implemented by the parent {ref}`Ceed`.
   In order to use the fallback mechanism, the parent {ref}`Ceed` and fallback resource must use compatible **E-vector** and **Q-vector** layouts.
