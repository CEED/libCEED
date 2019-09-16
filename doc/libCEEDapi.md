# libCEED: Documentation

This page provides a brief description of the theoretical foundations and the
practical implementation of the libCEED library.

Developers may also want to consult the automatically updated
[Doxygen documentation](https://codedocs.xyz/CEED/libCEED).

## Finite Element Operator Decomposition

Finite element operators are typically defined through weak formulations of
partial differential equations that involve integration over a computational
mesh. The required integrals are computed by splitting them as a sum over the
mesh elements, mapping each element to a simple *reference* element (e.g. the
unit square) and applying a quadrature rule in reference space.

This sequence of operations highlights an inherent hierarchical structure
present in all finite element operators where the evaluation starts on *global
(trial) degrees of freedom (dofs) on the whole mesh*, restricts to *degrees of
freedom on subdomains* (groups of elements), then moves to independent *degrees
of freedom on each element*, transitions to independent *quadrature points* in
reference space, performs the integration, and then goes back in reverse order
to global (test) degrees of freedom on the whole mesh.

This is illustrated below for the simple case of symmetric linear operator on
third order (Q3) scalar continuous (H1) elements, where we use the notions
**T-vector**, **L-vector**, **E-vector** and **Q-vector** to represent the sets
corresponding to the (true) degrees of freedom on the global mesh, the split
local degrees of freedom on the subdomains, the split degrees of freedom on the
mesh elements, and the values at quadrature points, respectively.

We refer to the operators that connect the different types of vectors as:
- Subdomain restriction **P**
- Element restriction **G**
- Basis (Dofs-to-Qpts) evaluator **B**
- Operator at quadrature points **D**

More generally, when the test and trial space differ, they get their own
versions of **P**, **G** and **B**.

![Operator Decomposition](libCEED.png "Operator Decomposition")

Note that in the case of adaptive mesh refinement (AMR), the restrictions **P**
and **G** will involve not just extracting sub-vectors, but evaluating values at
constrained degrees of freedom through the AMR interpolation. There can also be
several levels of subdomains (**P1**, **P2**, etc.), and it may be convenient to
split **D** as the product of several operators (**D1**, **D2**, etc.).

### Partial Assembly

Since the global operator **A** is just a series of variational restrictions
with **B**, **G** and **P**, starting from its point-wise kernel **D**, a
"matvec" with **A** can be performed by evaluating and storing some of the
innermost variational restriction matrices, and applying the rest of the
operators "on-the-fly". For example, one can compute and store a global matrix
on T-vector level. Alternatively, one can compute and store only the subdomain
(L-vector) or element (E-vector) matrices and perform the action of **A** using
matvecs with **P** or **P** and **G**. While these options are natural for
low-order discretizations, they are not a good fit for high-order methods due to
the amount of FLOPs needed for their evaluation, as well as the memory transfer
needed for a matvec.

Our focus in libCEED, instead, is on **partial assembly**, where we compute and
store only **D** (or portions of it) and evaluate the actions of **P**, **G**
and **B** on-the-fly. Critically for performance, we take advantage of the
tensor-product structure of the degrees of freedom and quadrature points on quad
and hex elements to perform the action of **B** without storing it as a matrix.

Implemented properly, the partial assembly algorithm requires optimal amount of
memory transfers (with respect to the polynomial order) and near-optimal FLOPs
for operator evaluation. It consists of an operator *setup* phase, that
evaluates and stores **D** and an operator *apply* (evaluation) phase that
computes the action of **A** on an input vector. When desired, the setup phase
may be done as a side-effect of evaluating a different operator, such as a
nonlinear residual. The relative costs of the setup and apply phases are
different depending on the physics being expressed and the representation of
**D**.

### Parallel Decomposition

After the application of each of the first three transition operators, **P**,
**G** and **B**, the operator evaluation is decoupled on their ranges, so **P**,
**G** and **B** allow us to "zoom-in" to subdomain, element and quadrature point
level, ignoring the coupling at higher levels.

Thus, a natural mapping of **A** on a parallel computer is to split the
**T-vector** over MPI ranks (a non-overlapping decomposition, as is typically
used for sparse matrices), and then split the rest of the vector types over
computational devices (CPUs, GPUs, etc.) as indicated by the shaded regions in
the diagram above.

One of the advantages of the decomposition perspective in these settings is that
the operators **P**, **G**, **B** and **D** clearly separate the MPI parallelism
in the operator (**P**) from the unstructured mesh topology (**G**), the choice
of the finite element space/basis (**B**) and the geometry and point-wise
physics **D**. These components also naturally fall in different classes of
numerical algorithms -- parallel (multi-device) linear algebra for **P**, sparse
(on-device) linear algebra for **G**, dense/structured linear algebra (tensor
contractions) for **B** and parallel point-wise evaluations for **D**.

Currently in libCEED, it is assumed that the host application manages the global
**T-vectors** and the required communications among devices (which are generally
on different compute nodes) with **P**. Our API is thus focused on the
**L-vector** level, where the logical devices, which in the library are
represented by the `Ceed` object, are independent. Each MPI rank can use one or
more `Ceed`s, and each `Ceed`, in turn, can represent one or more physical
devices, as long as libCEED backends support such configurations. The idea is
that every MPI rank can use any logical device it is assigned at runtime. For
example, on a node with 2 CPU sockets and 4 GPUs, one may decide to use 6 MPI
ranks (each using a single `Ceed` object): 2 ranks using 1 CPU socket each, and
4 using 1 GPU each. Another choice could be to run 1 MPI rank on the whole node
and use 5 `Ceed` objects: 1 managing all CPU cores on the 2 sockets and 4
managing 1 GPU each. The communications among the devices, e.g. required for
applying the action of **P**, are currently out of scope of libCEED. The
interface is non-blocking for all operations involving more than O(1) data,
allowing operations performed on a coprocessor or worker threads to overlap with
operations on the host.

## API Description

The libCEED API takes an algebraic approach, where the user essentially
describes in the *frontend* the operators **G**, **B** and **D** and the library
provides *backend* implementations and coordinates their action to the original
operator on **L-vector** level (i.e. independently on each device / MPI task).

One of the advantages of this purely algebraic description is that it already
includes all the finite element information, so the backends can operate on
linear algebra level without explicit finite element code. The frontend
description is general enough to support a wide variety of finite element
algorithms, as well as some other types algorithms such as spectral finite
differences. The separation of the front- and backends enables applications to
easily switch/try different backends. It also enables backend developers to
impact many applications from a single implementation.

Our long-term vision is to include a variety of backend implementations in
libCEED, ranging from reference kernels to highly optimized kernels targeting
specific devices (e.g. GPUs) or specific polynomial orders. A simple reference
backend implementation is provided in the file
[ceed-ref.c](https://github.com/CEED/libCEED/blob/master/backends/ref/ceed-ref.c).

On the frontend, the mapping between the decomposition concepts and the code
implementation is as follows:
- **L-**, **E-** and **Q-vector** are represented as variables of type `CeedVector`.
  (A backend may choose to operate incrementally without forming explicit **E-** or **Q-vectors**.)
- **G** is represented as variable of type `CeedElemRestriction`.
- **B** is represented as variable of type `CeedBasis`.
- the action of **D** is represented as variable of type `CeedQFunction`.
- the overall operator **G<sup>T</sup> B<sup>T</sup> D B G** is represented as variable of type
  `CeedOperator` and its action is accessible through `CeedOperatorApply()`.

To clarify these concepts and illustrate how they are combined in the API,
consider the implementation of the action of a simple 1D mass matrix
(cf. [tests/t500-operator.c](https://github.com/CEED/libCEED/blob/master/tests/t500-operator.c)).

\include t500-operator.c

The constructor

\snippet t500-operator.c Ceed Init

creates a logical device `ceed` on the specified *resource*, which could also be
a coprocessor such as `"/nvidia/0"`. There can be any number of such devices,
including multiple logical devices driving the same resource (though performance
may suffer in case of oversubscription). The resource is used to locate a
suitable backend which will have discretion over the implementations of all
objects created with this logical device.

The `setup` routine above computes and stores **D**, in this case a scalar value
in each quadrature point, while `mass` uses these saved values to perform the
action of **D**. These functions are turned into the `CeedQFunction` variables
`qf_setup` and `qf_mass` in the `CeedQFunctionCreateInterior()` calls:

\snippet t500-operator.c QFunction Create

A `CeedQFunction` performs independent operations at each quadrature point and
the interface is intended to facilitate vectorization.  The second argument is
an expected vector length. If greater than 1, the caller must ensure that the
number of quadrature points `Q` is divisible by the vector length. This is
often satisfied automatically due to the element size or by batching elements
together to facilitate vectorization in other stages, and can always be ensured
by padding.

In addition to the function pointers (`setup` and `mass`), `CeedQFunction`
constructors take a string representation specifying where the source for the
implementation is found. This is used by backends that support Just-In-Time
(JIT) compilation (i.e., CUDA and OCCA) to compile for coprocessors.

Different input and output fields are added individually, specifying the field
name, size of the field, and evaluation mode.

The size of the field is provided by a combination of the number of components
the effect of any basis evaluations.

The evaluation mode `CEED_EVAL_INTERP` for both input and output fields
indicates that the mass operator only contains terms of the form

<!-- \int_\Omega v f_0(u) -->
![](https://latex.codecogs.com/svg.latex?$$\int_\Omega v f_0(u)$$)

where *v* are test functions. More general operators, such as those of the form

<!-- \int_\Omega v f_0(u, \nabla u) + \nabla v \cdot f_1(u, \nabla u) -->
![](https://latex.codecogs.com/svg.latex?$$\int_\Omega v f_0(u \nabla u)+\nabla v \cdot f_1(u \nabla u)$$)

can be expressed.

For fields with derivatives, such as with the basis evaluation mode
`CEED_EVAL_GRAD`, the size of the field needs to reflect both the number of
components and the geometric dimension. A 3-dimensional gradient on four components
would therefore mean the field has a size of 12.

The **B** operators for the mesh nodes, `bx`, and the unknown field, `bu`, are
defined in the calls to the function `CeedBasisCreateTensorH1Lagrange()`. In this
example, both the mesh and the unknown field use H1 Lagrange finite elements of
order 1 and 4 respectively (the `P` argument represents the number of 1D degrees
of freedom on each element). Both basis operators use the same integration rule,
which is Gauss-Legendre with 8 points (the `Q` argument).

\snippet t500-operator.c Basis Create

Other elements with this structure can be specified in terms of the `Q×P`
matrices that evaluate values and gradients at quadrature points in one
dimension using `CeedBasisCreateTensorH1()`. Elements that do not have tensor
product structure, such as symmetric elements on simplices, will be created
using different constructors.

The **G** operators for the mesh nodes, `Erestrictx`, and the unknown field,
`Erestrictu`, are specified in the `CeedElemRestrictionCreate()`. Both of these
specify directly the dof indices for each element in the `indx` and `indu`
arrays:

\snippet t500-operator.c ElemRestr Create
\snippet t500-operator.c ElemRestrU Create

If the user has arrays available on a device, they can be provided using
`CEED_MEM_DEVICE`. This technique is used to provide no-copy interfaces in all
contexts that involve problem-sized data.

For discontinuous Galerkin and for applications such as Nek5000 that only
explicitly store **E-vectors** (inter-element continuity has been subsumed by
the parallel restriction **P**), the element restriction **G** is the identity
and `CeedElemRestrictionCreateIdentity()` is used instead. We plan to support
other structured representations of **G** which will be added according to demand.
In the case of non-conforming mesh elements, **G** needs a more general
representation that expresses values at slave nodes (which do not appear in
**L-vectors**) as linear combinations of the degrees of freedom at master nodes.

These operations, **P**, **B**, and **D**, are combined with a `CeedOperator`.
As with qfunctions, operator fields are added separately with a matching
field name, basis (**B**), element restriction (**G**), and **L-vector**. The flag
`CEED_VECTOR_ACTIVE` indicates that the vector corresponding to that field will
be provided to the operator when `CeedOperatorApply()` is called. Otherwise the
input/output will be read from/written to the specified **L-vector**.

With partial assembly, we first perform a setup stage where **D** is evaluated
and stored. This is accomplished by the operator `op_setup` and its application
to `X`, the nodes of the mesh (these are needed to compute Jacobians at
quadrature points). Note that the corresponding `CeedOperatorApply()` has no basis
evaluation on the output, as the quadrature data is not needed at the dofs:

\snippet t500-operator.c Setup Create

\snippet t500-operator.c Setup Set

\snippet t500-operator.c Setup Apply

The action of the operator is then represented by operator `op_mass` and its
`CeedOperatorApply()` to the input **L-vector** `U` with output in `V`:

\snippet t500-operator.c Operator Create

\snippet t500-operator.c Operator Set

\snippet t500-operator.c Operator Apply

A number of function calls in the interface, such as `CeedOperatorApply()`, are
intended to support asynchronous execution via their last argument,
`CeedRequest*`. The specific (pointer) value used in the above example,
`CEED_REQUEST_IMMEDIATE`, is used to express the request (from the user) for the
operation to complete before returning from the function call, i.e. to make sure
that the result of the operation is available in the output parameters
immediately after the call. For a true asynchronous call, one needs to provide
the address of a user defined variable. Such a variable can be used later to
explicitly wait for the completion of the operation.

## Interface Principles and Evolution

LibCEED is intended to be extensible via backends that are packaged with the
library and packaged separately (possibly as a binary containing proprietary
code). Backends are registered by calling

\snippet ref/ceed-ref.c Register

typically in a library initializer or "constructor" that runs automatically.
`CeedInit` uses this prefix to find an appropriate backend for the resource.

Source (API) and binary (ABI) stability are important to libCEED. LibCEED is
evolving rapidly at present, but we expect it to stabilize soon at which point
we will adopt semantic versioning. User code, including libraries of
`CeedQFunction`s, will not need to be recompiled except between major releases.
The backends currently have some dependence beyond the public user interface,
but we intent to remove that dependence and will prioritize if anyone expresses
interest in distributing a backend outside the libCEED repository.
