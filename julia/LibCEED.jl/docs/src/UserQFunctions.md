# Defining User Q-Functions

An important feature of LibCEED.jl is the ability to define [user
Q-functions](https://libceed.org/en/latest/libCEEDapi/#gallery-of-qfunctions)
natively in Julia. These user Q-functions work with both the CPU and CUDA
backends.

User Q-functions describe the action of the $D$ operator at quadrature points
(see [libCEED's theoretical
framework](https://libceed.org/en/latest/libCEEDapi/#theoretical-framework)).
Since the Q-functions are invoked at every quadrature point, efficiency is
very important.

## Apply mass Q-function in C

Before describing how to define user Q-functions in Julia, we will briefly given
an example of a user Q-function defined in C. This is the "apply mass"
Q-function from `ex1-volume.c`, which computes the action of the mass operator.
The mass operator on each element can be written as $B^\intercal D B$, where $B$
is the basis operator, and $D$ represents multiplication by quadrature weights
and geometric factors (i.e. the determinant of the mesh transformation Jacobian
at each qudarture point). It is the action of $D$ that the Q-function must
implement. The C source of the Q-function is:

```c
/// libCEED Q-function for applying a mass operator
CEED_QFUNCTION(f_apply_mass)(void *ctx, const CeedInt Q,
                             const CeedScalar *const *in,
                             CeedScalar *const *out) {
  const CeedScalar *u = in[0], *qdata = in[1];
  CeedScalar *v = out[0];
  // Quadrature Point Loop
  CeedPragmaSIMD
  for (CeedInt i=0; i<Q; i++) {
    v[i] = qdata[i] * u[i];
  } // End of Quadrature Point Loop
  return 0;
}
```

From this example, we see that a user Q-function is a C callback that takes a
"data context" pointer, a number of quadrature points, and two arrays of arrays,
one for inputs, and one for outputs.

In this example, the first input array is `u`, which is the value of the trial
function evaluated at each quadrature point. The second input array is `qdata`,
which contains the precomputed geometric factors. There is only one output
array, `v`, which will store the pointwise product of `u` and `data`. Given the
definition of this Q-function, the `CeedQFunction` object is created by
```c
CeedQFunctionCreateInterior(ceed, 1, f_apply_mass, f_apply_mass_loc, &apply_qfunc);
CeedQFunctionAddInput(apply_qfunc, "u", 1, CEED_EVAL_INTERP);
CeedQFunctionAddInput(apply_qfunc, "qdata", 1, CEED_EVAL_NONE);
CeedQFunctionAddOutput(apply_qfunc, "v", 1, CEED_EVAL_INTERP);
```
When adding the inputs and outputs, `CEED_EVAL_INTERP` indicates that the $B$
basis operator should be used to interpolate the trial and test functions from
nodal points to quadrature points, and `CEED_EVAL_NONE` indicates that the
`qdata` is already precomputed at quadrature points, and no interpolation is
requried.

## Apply mass Q-function in Julia

We now replicate this Q-function in Julia. The main way of defining user
Q-functions in Julia is using the [`@interior_qf`](@ref) macro. The above C code
(both the definition of the Q-function, its creation, and adding the inputs and
outputs) is analogous to the following Julia code:

```julia
@interior_qf apply_qfunc = (
    ceed, Q,
    (u, :in, EVAL_INTERP, Q), (qdata, :in, EVAL_NONE, Q),
    (v, :out, EVAL_INTERP, Q),
    @inbounds @simd for i=1:Q
        v[i] = qdata[i]*u[i]
    end
)
```

This creates a [`QFunction`](@ref) object named `apply_qfunc`. The Q-function is
defined by the tuple on the right-hand side. `ceed` is the name of the
[`Ceed`](@ref) object where the Q-function will be created, and the second
argument, `Q`, is the name of that variable that will contain the number of
quadrature points. The next three arguments are specifications of the input and
output fields:
```julia
    (u, :in, EVAL_INTERP, Q),
    (qdata, :in, EVAL_NONE, Q),
    (v, :out, EVAL_INTERP, Q),
```
Each input or output field specification is a tuple, where the first entry is
the name of the array, and the second entry is either `:in` or `:out`, according
to whether the array is an input or output array. The third entry is the
[`EvalMode`](@ref) of the field. The remaining entries are the dimensions of the
array. The first dimension is always equal to the number of quadrature points.
In this case, all the arrays are simply vectors whose size is equal to the
number of quadrature points, but in more sophisticated examples (e.g. the [apply
diffusion Q-function](@ref applydiff)) these arrays could consists of vectors or
matrices at each quadrature point. After providing all of the array
specifications, the body of the Q-function is provided.

## [Apply diffusion Q-function in Julia](@id applydiff)

For a more sophisticated example of a Q-function, we consider the "apply
diffusion" Q-function, used in `ex2-surface`. This Q-function computes the
action of the diffusion operator. When written in the form $B^\intercal D B$, in
this case $B$ represents the basis gradient matrix, and $D$ represents
multiplication by $w \det(J) J^{-\intercal} J^{-1}$, where $J$ is the mesh
transformation Jacobian, and $w$ is the quadrature weight.

This Q-function is implemented in Julia as follows:
```julia
@interior_qf apply_qfunc = (
    ceed, Q, dim=dim,
    (du, :in, EVAL_GRAD, Q, dim),
    (qdata, :in, EVAL_NONE, Q, dim*(dim+1)÷2),
    (dv, :out, EVAL_GRAD, Q, dim),
    @inbounds @simd for i=1:Q
        dXdxdXdxT = getvoigt(@view(qdata[i,:]), CeedDim(dim))
        dui = SVector{dim}(@view(du[i,:]))
        dv[i,:] .= dXdxdXdxT*dui
    end
)
```
In contrast to the previous example, before the field specifications, this
Q-function includes a _constant definition_ `dim=dim`. The
[`@interior_qf`](@ref) macro allows for any number of constant definitions,
which make the specified values available within the body of the Q-function as
compile-time constants.

In this example, `dim` is either 1, 2, or 3 according to the spatial dimension
of the problem. When the user Q-function is defined, LibCEED.jl will JIT compile
the body of the Q-function and make it available to libCEED as a C callback. In
the body of this Q-function, `dim` will be available, and its value will be a
compile-time constant, allowing for (static) dispatch based on the value of
`dim`, and eliminating branching.

Note that `dim` is also available for use in the field specifications. In this
example, the field specifications are slightly more involved that in the
previous example. The arrays are given by
```julia
    (du, :in, EVAL_GRAD, Q, dim),
    (qdata, :in, EVAL_NONE, Q, dim*(dim+1)÷2),
    (dv, :out, EVAL_GRAD, Q, dim),
```
Note that the input array `du` has [`EvalMode`](@ref) `EVAL_GRAD`, meaning that
this array stores the gradient of the trial function at each quadrature point.
Therefore, at each quadrature point, `du` stores a vector of length `dim`, and
so the shape of `du` is `(Q, dim)`. Similarly, the action of $D$ is given by
$w \det(J) J^{-\intercal} J^{-1} \nabla u$, which is also a vector of length `dim` at
each quadrature point. This means that the output array `dv` also has shape
`(Q, dim)`.

The geometric factors stored in `qdata` represent the symmetric matrix $w
\det(J) J^{-\intercal} J^{-1}$ evaluated at every quadrature point. In order to
reduce data usage, instead of storing this data as a $d \times d$ matrix, we use
the fact that we know it is symmetric to only store $d(d+1)/2$ entries, and the
remaining entries we infer by symmetry. These entries are stored using the
[Voigt convention](https://en.wikipedia.org/wiki/Voigt_notation). LibCEED.jl
provides some [utilities](Misc.md#LibCEED.getvoigt) for storing and extracting
symmetric matrices stored in this fashion.

After the field specifications, we have the body of the Q-function:
```julia
@inbounds @simd for i=1:Q
    dXdxdXdxT = getvoigt(@view(qdata[i,:]), CeedDim(dim))
    dui = SVector{dim}(@view(du[i,:]))
    dv[i,:] .= dXdxdXdxT*dui
end
```
First, the matrix $w \det(J) J^{-\intercal} J^{-1}$ is stored in the variable
`dXdxdXdxT`. The symmetric entries of this matrix are accesed using
`@view(qdata[i,:])`, which avoids allocations. [`getvoigt`](@ref) is used to
convert from Voigt notation to a symmetric matrix, which returns a statically
sized `SMatrix`. The version for the correct spatial dimension is selected using
`CeedDim(dim)`, which allows for compile-time dispatch, since `dim` is a
constant whose value is known as a constant when the Q-function is JIT compiled.

Then, the gradient of $u$ at the given quadrature point is loaded as a
fixed-size `SVector`. The result is placed into the output array, where the
[`StaticArrays.jl`](https://github.com/JuliaArrays/StaticArrays.jl) package
evaluates `dXdxdXdxT*dui` using an optimized matrix-vector product for small
matrices (since their sizes are known statically).

## GPU Kernels

If the `Ceed` resource uses a CUDA backend, then the user Q-functions defined
using [`@interior_qf`](@ref) are automatically compiled as CUDA kernels using
[`CUDA.jl`](https://github.com/JuliaGPU/CUDA.jl). Some Julia features are not
available in GPU code (for example, dynamic dispatch), so if the Q-function is
intended to be run on the GPU, the user should take care when defining the body
of the user Q-function.
