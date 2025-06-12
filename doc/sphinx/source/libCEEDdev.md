# Developer Notes

## Library Design

LibCEED has a single user facing API for creating and using the libCEED objects ({ref}`CeedVector`, {ref}`CeedBasis`, etc).
Different Ceed backends are selected by instantiating a different {ref}`Ceed` object to create the other libCEED objects, in a [bridge pattern](https://en.wikipedia.org/wiki/Bridge_pattern).
At runtime, the user can select the different backend implementations to target different hardware, such as CPUs or GPUs.

When designing new features, developers should place the function definitions for the user facing API in the header `/include/ceed/ceed.h`.
The basic implementation of these functions should typically be placed in `/interface/*.c` files.
The interface should pass any computationally expensive or hardware specific operations to a backend implementation.
A new method for the associated libCEED object can be added in `/include/ceed-impl.h`, with a corresponding `CEED_FTABLE_ENTRY` in `/interface/ceed.c` to allow backends to set their own implementations of this method.
Then in the creation of the backend specific implementation of the object, typically found in `/backends/[impl]/ceed-[impl]-[object].c`, the developer creates the backend implementation of the specific method and calls {c:func}`CeedSetBackendFunction` to set this implementation of the method for the backend.
Any supplemental functions intended to be used in the interface or by the backends may be added to the backend API in the header `/include/ceed/backend.h`.
The basic implementation of these functions should also be placed in `/interface/*.c` files.

LibCEED generally follows a "CPU first" implementation strategy when adding new functionality to the user facing API.
If there are no performance specific considerations, it is generally recommended to include a basic CPU default implementation in `/interface/*.c`.
Any new functions must be well documented and tested.
Once the user facing API and the default implementation are in place and verified correct via tests, then the developer can focus on hardware specific implementations (AVX, CUDA, HIP, etc.) as necessary.

## Backend Inheritance

A Ceed backend is not required to implement all libCeed objects or {ref}`CeedOperator` methods.
There are three mechanisms by which a Ceed backend can inherit implementations from another Ceed backend.

1. Delegation - Developers may use {c:func}`CeedSetDelegate` to set a general delegate {ref}`Ceed` object.
   This delegate {ref}`Ceed` will provide the implementation of any libCeed objects that parent backend does not implement.
   For example, the `/cpu/self/xsmm/serial` backend implements the `CeedTensorContract` object itself but delegates all other functionality to the `/cpu/self/opt/serial` backend.

2. Object delegation  - Developers may use {c:func}`CeedSetObjectDelegate` to set a delegate {ref}`Ceed` object for a specific libCEED object.
   This delegate {ref}`Ceed` will only provide the implementation of that specific libCeed object for the parent backend.
   Object delegation has higher precedence than delegation.

3. Operator fallback - Developers may use {c:func}`CeedSetOperatorFallbackResource` to set a string identifying which {ref}`Ceed` backend will be instantiated to provide any unimplemented {ref}`CeedOperator` methods.
   This fallback {ref}`Ceed` object will only be created if a method is called that is not implemented by the parent backend.
   In order to use the fallback mechanism, the parent backend and fallback backend must use compatible E-vector and Q-vector layouts.
   For example, the `/gpu/cuda/gen` falls back to `/gpu/cuda/ref` for missing {ref}`CeedOperator` methods.
   If an unimplemented method is called, then the parent `/gpu/cuda/gen` {ref}`Ceed` object creates a fallback `/gpu/cuda/ref` {ref}`Ceed` object and creates a clone of the {ref}`CeedOperator` with this fallback {ref}`Ceed` object.
   This clone {ref}`CeedOperator` is then used for the missing methods.

## Backend Families

There are 4 general 'families' of backend implementations.
As internal data layouts are specific to backend families, it is generally not possible to delegate between backend families.

### CPU Backends

The basic CPU with the simplest implementation is `/cpu/self/ref/serial`.
This backend contains the basic implementations of most objects that other backends rely upon.
Most of the other CPU backends only update the {ref}`CeedOperator` and `CeedTensorContract` objects.

The `/cpu/self/ref/blockend` and `/cpu/self/opt/*` backends delegate to the `/cpu/self/ref/serial` backend.
The `/cpu/self/ref/blocked` backend updates the {ref}`CeedOperator` to use an E-vector and Q-vector ordering when data for 8 elements are interlaced to provide better vectorization.
The `/cpu/self/opt/*` backends update the {ref}`CeedOperator` to apply the action of the operator in 1 or 8 element batches, depending upon if the blocking strategy is used.
This reduced the memory required to utilize this backend significantly.

The `/cpu/self/avx/*` and `/cpu/self/xsmm/*` backends delegate to the corresponding `/cpu/self/opt/*` backends.
These backends update the `CeedTensorContract` objects using AVX intrinsics and libXSMM functions, respectively.

The `/cpu/self/memcheck/*` backends delegate to the `/cpu/self/ref/*` backends.
These backends replace many of the implementations with methods that include more verification checks and a memory management model that more closely matches the memory management for GPU backends.
These backends rely upon the [Valgrind](https://valgrind.org/) Memcheck tool and Valgrind headers.

### GPU Backends

The CUDA, HIP, and SYCL backend families all follow the same basic design.

The `/gpu/*/ref` backends provide basic functionality.
In these backends, the operator is applied in multiple separate kernel launches, following the libCEED operator decomposition, where first {ref}`CeedElemRestriction` kernels map from the L-vectors to E-vectors, then {ref}`CeedBasis` kernels map from the E-vectors to Q-vectors, then the {ref}`CeedQFunction` kernel provides the action of the user quadrature point function, and the transpose {ref}`CeedBasis` and {ref}`CeedElemRestriction` kernels are applied to go back to the E-vectors and finally the L-vectors.
These kernels apply to all points across all elements in order to maximize the amount of work each kernel launch has.

The `/gpu/*/shared` backends delegate to the corresponding `/gpu/*/ref` backends.
These backends use shared memory to improve performance for the {ref}`CeedBasis` kernels.
All other libCEED objects are delegated to `/gpu/*/ref`.

The `/gpu/*/gen` backends delegate to the corresponding `/gpu/*/shared` backends.
These backends write a single comprehensive kernel to apply the action of the {ref}`CeedOperator`, significantly improving performance by eliminating intermediate data structures and reducing the total number of kernel launches required.

The `/gpu/*/magma` backends delegate to the corresponding `/gpu/*/ref` backends.
These backends provide better performance for {ref}`CeedBasis` kernels but do not have the improvements from the `/gpu/*/gen` backends for {ref}`CeedOperator`.

The `/*/*/occa` backends are an experimental feature and not part of any family.

## Internal Layouts

Ceed backends are free to use any E-vector and Q-vector data layout (including never fully forming these vectors) so long as the backend passes the `t5**` series tests and all examples.
There are several common layouts for L-vectors, E-vectors, and Q-vectors, detailed below:

- **L-vector** layouts

  - L-vectors described by a standard {ref}`CeedElemRestriction` have a layout described by the `offsets` array and `comp_stride` parameter.
    Data for node `i`, component `j`, element `k` can be found in the L-vector at index `offsets[i + k*elem_size] + j*comp_stride`.
  - L-vectors described by a strided {ref}`CeedElemRestriction` have a layout described by the `strides` array.
    Data for node `i`, component `j`, element `k` can be found in the L-vector at index `i*strides[0] + j*strides[1] + k*strides[2]`.

- **E-vector** layouts

  - If possible, backends should use {c:func}`CeedElemRestrictionSetELayout()` to use the `t2**` tests.
    If the backend uses a strided E-vector layout, then the data for node `i`, component `j`, element `k` in the E-vector is given by `i*layout[0] + j*layout[1] + k*layout[2]`.
  - Backends may choose to use a non-strided E-vector layout; however, the `t2**` tests will not function correctly in this case and these tests will need to be marked as allowable failures for this backend in the test suite.

- **Q-vector** layouts

  - When the size of a {ref}`CeedQFunction` field is greater than `1`, data for quadrature point `i` component `j` can be found in the Q-vector at index `i + Q*j`, where `Q` is the total number of quadrature points in the Q-vector.
    Backends are free to provide the quadrature points in any order.
  - When the {ref}`CeedQFunction` field has `emode` `CEED_EVAL_GRAD`, data for quadrature point `i`, component `j`, derivative `k` can be found in the Q-vector at index `i + Q*j + Q*num_comp*k`.
  - Backend developers must take special care to ensure that the data in the Q-vectors for a field with `emode` `CEED_EVAL_NONE` is properly ordered when the backend uses different layouts for E-vectors and Q-vectors.

## CeedVector Array Access

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

  - Internal synchronization and user calls to {c:func}`CeedVectorSync` cannot be made on a vector in an *invalid state*.
  - Calls to {c:func}`CeedVectorGetArray` and {c:func}`CeedVectorGetArrayRead` cannot be made on a vector in an *invalid state*.
  - Calls to {c:func}`CeedVectorSetArray` and {c:func}`CeedVectorSetValue` can be made on a vector in an *invalid state*.
  - Calls to {c:func}`CeedVectorGetArrayWrite` can be made on a vector in an *invalid* state.
    Data synchronization is not required for the memory location returned by {c:func}`CeedVectorGetArrayWrite`.
    The caller should assume that all data at the memory location returned by {c:func}`CeedVectorGetArrayWrite` is *invalid*.

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
for (d = 0; d < dim; d++) {
  for (c = 0; c < num_comp; c++) {
    for (q = 0; q < Q; q++) {
      for (e = 0; e < num_elem; e++) {
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
