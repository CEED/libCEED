Interface Concepts
========================================

This page provides a brief description of the theoretical foundations and the
practical implementation of the libCEED library.

.. include:: FEMtheory.rst.inc

Finite Element Operator Decomposition
----------------------------------------

Finite element operators are typically defined through weak formulations of
partial differential equations that involve integration over a computational
mesh. The required integrals are computed by splitting them as a sum over the
mesh elements, mapping each element to a simple *reference* element (e.g. the
unit square) and applying a quadrature rule in reference space.

This sequence of operations highlights an inherent hierarchical structure
present in all finite element operators where the evaluation starts on *global
(trial) degrees of freedom (dofs) or nodes on the whole mesh*, restricts to
*dofs on subdomains* (groups of elements), then moves to independent
*dofs on each element*, transitions to independent *quadrature points* in
reference space, performs the integration, and then goes back in reverse order
to global (test) degrees of freedom on the whole mesh.

This is illustrated below for the simple case of symmetric linear operator on
third order (:math:`Q_3`) scalar continuous (:math:`H^1`) elements, where we use
the notions **T-vector**, **L-vector**, **E-vector** and **Q-vector** to represent
the sets corresponding to the (true) degrees of freedom on the global mesh, the split
local degrees of freedom on the subdomains, the split degrees of freedom on the
mesh elements, and the values at quadrature points, respectively.

We refer to the operators that connect the different types of vectors as:

- Subdomain restriction :math:`\bm{P}`
- Element restriction :math:`\bm{G}`
- Basis (Dofs-to-Qpts) evaluator :math:`\bm{B}`
- Operator at quadrature points :math:`\bm{D}`

More generally, when the test and trial space differ, they get their own
versions of :math:`\bm{P}`, :math:`\bm{G}` and :math:`\bm{B}`.

.. _fig-operator-decomp:

.. figure:: ../../img/libCEED.png

   Operator Decomposition

Note that in the case of adaptive mesh refinement (AMR), the restrictions
:math:`\bm{P}` and :math:`\bm{G}` will involve not just extracting sub-vectors,
but evaluating values at constrained degrees of freedom through the AMR interpolation.
There can also be several levels of subdomains (:math:`\bm P_1`, :math:`\bm P_2`,
etc.), and it may be convenient to split :math:`\bm{D}` as the product of several
operators (:math:`\bm D_1`, :math:`\bm D_2`, etc.).


Terminology and Notation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Vector representation/storage categories:

- True degrees of freedom/unknowns, **T-vector**:

   - each unknown :math:`i` has exactly one copy, on exactly one processor, :math:`rank(i)`

   - this is a non-overlapping vector decomposition

   - usually includes any essential (fixed) dofs.

   .. image:: ../../img/T-vector.svg

- Local (w.r.t. processors) degrees of freedom/unknowns, **L-vector**:

   - each unknown :math:`i` has exactly one copy on each processor that owns an
     element containing :math:`i`

   - this is an overlapping vector decomposition with overlaps only across
     different processors---there is no duplication of unknowns on a single
     processor

   - the shared dofs/unknowns are the overlapping dofs, i.e. the ones that have
     more than one copy, on different processors.

   .. image:: ../../img/L-vector.svg

- Per element decomposition, **E-vector**:

   - each unknown :math:`i` has as many copies as the number of elements that contain
     :math:`i`
   - usually, the copies of the unknowns are grouped by the element they belong
     to.

   .. image:: ../../img/E-vector.svg

- In the case of AMR with hanging nodes (giving rise to hanging dofs):

   - the **L-vector** is enhanced with the hanging/dependent dofs

   - the additional hanging/dependent dofs are duplicated when they are shared
     by multiple processors

   - this way, an **E-vector** can be derived from an **L-vector** without any
     communications and without additional computations to derive the dependend
     dofs

   - in other words, an entry in an **E-vector** is obtained by copying an entry
     from the corresponding **L-vector**, optionally switching the sign of the
     entry (for :math:`H(\mathrm{div})`---and :math:`H(\mathrm{curl})`-conforming spaces).

   .. image:: ../../img/L-vector-AMR.svg

- In the case of variable order spaces:

   - the dependent dofs (usually on the higher-order side of a face/edge) can
     be treated just like the hanging/dependent dofs case.

- Quadrature point vector, **Q-vector**:

   - this is similar to **E-vector** where instead of dofs, the vector represents
     values at quadrature points, grouped by element.

- In many cases it is useful to distinguish two types of vectors:

   - **X-vector**, or **primal X-vector**, and **X'-vector**, or **dual X-vector**

   - here X can be any of the T, L, E, or Q categories

   - for example, the mass matrix operator maps a **T-vector** to a **T'-vector**

   - the solutions vector is a **T-vector**, and the RHS vector is a **T'-vector**

   - using the parallel prolongation operator, one can map the solution
     **T-vector** to a solution **L-vector**, etc.

Operator representation/storage/action categories:

- Full true-dof parallel assembly, **TA**, or **A**:

   - ParCSR or similar format

   - the T in TA indicates that the data format represents an operator from a
     **T-vector** to a **T'-vector**.

- Full local assembly, **LA**:

   - CSR matrix on each rank

   - the parallel prolongation operator, :math:`\bm{P}`, (and its transpose) should use
     optimized matrix-free action

   - note that :math:`\bm{P}` is the operator mapping T-vectors to L-vectors.

- Element matrix assembly, **EA**:

   - each element matrix is stored as a dense matrix

   - optimized element and parallel prolongation operators

   - note that the element prolongation operator is the mapping from an
     **L-vector** to an **E-vector**.

- Quadrature-point/partial assembly, **QA** or **PA**:

   - precompute and store :math:`w\det(J)` at all quadrature points in all mesh elements

   - the stored data can be viewed as a **Q-vector**.

- Unassembled option,  **UA** or **U**:

   - no assembly step

   - the action uses directly the mesh node coordinates, and assumes specific
     form of the coefficient, e.g. constant, piecewise-constant, or given as a
     **Q-vector** (Q-coefficient).


Partial Assembly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since the global operator :math:`\bm{A}` is just a series of variational restrictions
with :math:`\bm{B}`, :math:`\bm{G}` and :math:`\bm{P}`, starting from its
point-wise kernel :math:`\bm{D}`, a "matvec" with :math:`\bm{A}` can be
performed by evaluating and storing some of the innermost variational restriction
matrices, and applying the rest of the operators "on-the-fly". For example, one can
compute and store a global matrix on **T-vector** level. Alternatively, one can compute
and store only the subdomain (**L-vector**) or element (**E-vector**) matrices and
perform the action of :math:`\bm{A}` using matvecs with :math:`\bm{P}` or
:math:`\bm{P}` and :math:`\bm{G}`. While these options are natural for
low-order discretizations, they are not a good fit for high-order methods due to
the amount of FLOPs needed for their evaluation, as well as the memory transfer
needed for a matvec.

Our focus in libCEED, instead, is on **partial assembly**, where we compute and
store only :math:`\bm{D}` (or portions of it) and evaluate the actions of
:math:`\bm{P}`, :math:`\bm{G}` and :math:`\bm{B}` on-the-fly.
Critically for performance, we take advantage of the tensor-product structure of the
degrees of freedom and quadrature points on *quad* and *hex* elements to perform the
action of :math:`\bm{B}` without storing it as a matrix.

Implemented properly, the partial assembly algorithm requires optimal amount of
memory transfers (with respect to the polynomial order) and near-optimal FLOPs
for operator evaluation. It consists of an operator *setup* phase, that
evaluates and stores :math:`\bm{D}` and an operator *apply* (evaluation) phase that
computes the action of :math:`\bm{A}` on an input vector. When desired, the setup
phase may be done as a side-effect of evaluating a different operator, such as a
nonlinear residual. The relative costs of the setup and apply phases are
different depending on the physics being expressed and the representation of
:math:`\bm{D}`.


Parallel Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After the application of each of the first three transition operators,
:math:`\bm{P}`, :math:`\bm{G}` and :math:`\bm{B}`, the operator evaluation
is decoupled  on their ranges, so :math:`\bm{P}`, :math:`\bm{G}` and
:math:`\bm{B}` allow us to "zoom-in" to subdomain, element and quadrature point
level, ignoring the coupling at higher levels.

Thus, a natural mapping of :math:`\bm{A}` on a parallel computer is to split the
**T-vector** over MPI ranks (a non-overlapping decomposition, as is typically
used for sparse matrices), and then split the rest of the vector types over
computational devices (CPUs, GPUs, etc.) as indicated by the shaded regions in
the diagram above.

One of the advantages of the decomposition perspective in these settings is that
the operators :math:`\bm{P}`, :math:`\bm{G}`, :math:`\bm{B}` and
:math:`\bm{D}` clearly separate the MPI parallelism
in the operator (:math:`\bm{P}`) from the unstructured mesh topology
(:math:`\bm{G}`), the choice of the finite element space/basis (:math:`\bm{B}`)
and the geometry and point-wise physics :math:`\bm{D}`. These components also
naturally fall in different classes of numerical algorithms -- parallel (multi-device)
linear algebra for :math:`\bm{P}`, sparse (on-device) linear algebra for
:math:`\bm{G}`, dense/structured linear algebra (tensor contractions) for
:math:`\bm{B}` and parallel point-wise evaluations for :math:`\bm{D}`.

Currently in libCEED, it is assumed that the host application manages the global
**T-vectors** and the required communications among devices (which are generally
on different compute nodes) with **P**. Our API is thus focused on the
**L-vector** level, where the logical devices, which in the library are
represented by the :ref:`Ceed` object, are independent. Each MPI rank can use one or
more :ref:`Ceed`\s, and each :ref:`Ceed`, in turn, can represent one or more physical
devices, as long as libCEED backends support such configurations. The idea is
that every MPI rank can use any logical device it is assigned at runtime. For
example, on a node with 2 CPU sockets and 4 GPUs, one may decide to use 6 MPI
ranks (each using a single :ref:`Ceed` object): 2 ranks using 1 CPU socket each, and
4 using 1 GPU each. Another choice could be to run 1 MPI rank on the whole node
and use 5 :ref:`Ceed` objects: 1 managing all CPU cores on the 2 sockets and 4
managing 1 GPU each. The communications among the devices, e.g. required for
applying the action of :math:`\bm{P}`, are currently out of scope of libCEED. The
interface is non-blocking for all operations involving more than O(1) data,
allowing operations performed on a coprocessor or worker threads to overlap with
operations on the host.


API Description
----------------------------------------

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
`ceed-ref.c <https://github.com/CEED/libCEED/blob/main/backends/ref/ceed-ref.c>`_.

On the frontend, the mapping between the decomposition concepts and the code
implementation is as follows:

- **L-**, **E-** and **Q-vector** are represented as variables of type :ref:`CeedVector`.
  (A backend may choose to operate incrementally without forming explicit **E-** or
  **Q-vectors**.)

- :math:`\bm{G}` is represented as variable of type :ref:`CeedElemRestriction`.

- :math:`\bm{B}` is represented as variable of type :ref:`CeedBasis`.

- the action of :math:`\bm{D}` is represented as variable of type :ref:`CeedQFunction`.

- the overall operator :math:`\bm{G}^T \bm{B}^T \bm{D} \bm{B} \bm{G}`
  is represented as variable of type
  :ref:`CeedOperator` and its action is accessible through :c:func:`CeedOperatorApply()`.

To clarify these concepts and illustrate how they are combined in the API,
consider the implementation of the action of a simple 1D mass matrix
(cf. `tests/t500-operator.c <https://github.com/CEED/libCEED/blob/main/tests/t500-operator.c>`_).

.. literalinclude::  ../../../tests/t500-operator.c
   :language: c
   :linenos:

The constructor

.. literalinclude::  ../../../tests/t500-operator.c
   :language: c
   :start-at: CeedInit
   :end-at: CeedInit

creates a logical device ``ceed`` on the specified *resource*, which could also be
a coprocessor such as ``"/nvidia/0"``. There can be any number of such devices,
including multiple logical devices driving the same resource (though performance
may suffer in case of oversubscription). The resource is used to locate a
suitable backend which will have discretion over the implementations of all
objects created with this logical device.

The ``setup`` routine above computes and stores :math:`\bm{D}`, in this case a
scalar value in each quadrature point, while ``mass`` uses these saved values to perform
the action of :math:`\bm{D}`. These functions are turned into the :ref:`CeedQFunction`
variables ``qf_setup`` and ``qf_mass`` in the :c:func:`CeedQFunctionCreateInterior()` calls:

.. literalinclude::  ../../../tests/t500-operator.c
   :language: c
   :start-after: //! [QFunction Create]
   :end-before: //! [QFunction Create]

A :ref:`CeedQFunction` performs independent operations at each quadrature point and
the interface is intended to facilitate vectorization.  The second argument is
an expected vector length. If greater than 1, the caller must ensure that the
number of quadrature points ``Q`` is divisible by the vector length. This is
often satisfied automatically due to the element size or by batching elements
together to facilitate vectorization in other stages, and can always be ensured
by padding.

In addition to the function pointers (``setup`` and ``mass``), :ref:`CeedQFunction`
constructors take a string representation specifying where the source for the
implementation is found. This is used by backends that support Just-In-Time
(JIT) compilation (i.e., CUDA and OCCA) to compile for coprocessors.
For full support across all backends, these :ref:`CeedQFunction` source files must only contain constructs mutually supported by C99, C++11, and CUDA.
For example, explict type casting of void pointers and explicit use of compatable arguments for :code:`math` library functions is required, and variable-length array (VLA) syntax for array reshaping is only avaliable via libCEED's :code:`CEED_Q_VLA` macro.

Different input and output fields are added individually, specifying the field
name, size of the field, and evaluation mode.

The size of the field is provided by a combination of the number of components
the effect of any basis evaluations.

The evaluation mode (see :ref:`CeedBasis-Typedefs and Enumerations`) ``CEED_EVAL_INTERP``
for both input and output fields indicates that the mass operator only contains terms of
the form

.. math::
   \int_\Omega v \cdot f_0 (u, \nabla u)

where :math:`v` are test functions (see the :ref:`Theoretical Framework`).
More general operators, such as those of the form

.. math::
   \int_\Omega v \cdot f_0 (u, \nabla u) + \nabla v : f_1 (u, \nabla u)

can be expressed.

For fields with derivatives, such as with the basis evaluation mode
(see :ref:`CeedBasis-Typedefs and Enumerations`) ``CEED_EVAL_GRAD``, the size of the
field needs to reflect both the number of components and the geometric dimension.
A 3-dimensional gradient on four components would therefore mean the field has a size of
12.

The :math:`\bm{B}` operators for the mesh nodes, ``bx``, and the unknown field,
``bu``, are defined in the calls to the function :c:func:`CeedBasisCreateTensorH1Lagrange()`.
In this example, both the mesh and the unknown field use :math:`H^1` Lagrange finite
elements of order 1 and 4 respectively (the ``P`` argument represents the number of 1D
degrees of freedom on each element). Both basis operators use the same integration rule,
which is Gauss-Legendre with 8 points (the ``Q`` argument).

.. literalinclude::  ../../../tests/t500-operator.c
   :language: c
   :start-after: //! [Basis Create]
   :end-before: //! [Basis Create]

Other elements with this structure can be specified in terms of the ``Q×P``
matrices that evaluate values and gradients at quadrature points in one
dimension using :c:func:`CeedBasisCreateTensorH1()`. Elements that do not have tensor
product structure, such as symmetric elements on simplices, will be created
using different constructors.

The :math:`\bm{G}` operators for the mesh nodes, ``Erestrictx``, and the unknown field,
``Erestrictu``, are specified in the :c:func:`CeedElemRestrictionCreate()`. Both of these
specify directly the dof indices for each element in the ``indx`` and ``indu``
arrays:

.. literalinclude::  ../../../tests/t500-operator.c
   :language: c
   :start-after: //! [ElemRestr Create]
   :end-before: //! [ElemRestr Create]

.. literalinclude::  ../../../tests/t500-operator.c
   :language: c
   :start-after: //! [ElemRestrU Create]
   :end-before: //! [ElemRestrU Create]

If the user has arrays available on a device, they can be provided using
``CEED_MEM_DEVICE``. This technique is used to provide no-copy interfaces in all
contexts that involve problem-sized data.

For discontinuous Galerkin and for applications such as Nek5000 that only
explicitly store **E-vectors** (inter-element continuity has been subsumed by
the parallel restriction :math:`\bm{P}`), the element restriction :math:`\bm{G}`
is the identity and :c:func:`CeedElemRestrictionCreateStrided()` is used instead.
We plan to support other structured representations of :math:`\bm{G}` which will
be added according to demand.
There are two common approaches for supporting non-conforming elements: applying the node constraints via :math:`\bm P` so that the **L-vector** can be processed uniformly and applying the constraintsss via :math:`\bm G` so that the **E-vector** is uniform.
The former can be done with the existing interface while the latter will require a generalization to element restriction that would define field values at constrained nodes as linear combinations of the values at primary nodes.

These operations, :math:`\bm{P}`, :math:`\bm{B}`, and :math:`\bm{D}`,
are combined with a :ref:`CeedOperator`. As with :ref:`CeedQFunction`\s, operator fields are added
separately with a matching field name, basis (:math:`\bm{B}`), element restriction
(:math:`\bm{G}`), and **L-vector**. The flag
``CEED_VECTOR_ACTIVE`` indicates that the vector corresponding to that field will
be provided to the operator when :c:func:`CeedOperatorApply()` is called. Otherwise the
input/output will be read from/written to the specified **L-vector**.

With partial assembly, we first perform a setup stage where :math:`\bm{D}` is evaluated
and stored. This is accomplished by the operator ``op_setup`` and its application
to ``X``, the nodes of the mesh (these are needed to compute Jacobians at
quadrature points). Note that the corresponding :c:func:`CeedOperatorApply()` has no basis
evaluation on the output, as the quadrature data is not needed at the dofs:

.. literalinclude::  ../../../tests/t500-operator.c
   :language: c
   :start-after: //! [Setup Create]
   :end-before: //! [Setup Create]

.. literalinclude::  ../../../tests/t500-operator.c
   :language: c
   :start-after: //! [Setup Set]
   :end-before: //! [Setup Set]

.. literalinclude::  ../../../tests/t500-operator.c
   :language: c
   :start-after: //! [Setup Apply]
   :end-before: //! [Setup Apply]

The action of the operator is then represented by operator ``op_mass`` and its
:c:func:`CeedOperatorApply()` to the input **L-vector** ``U`` with output in ``V``:

.. literalinclude::  ../../../tests/t500-operator.c
   :language: c
   :start-after: //! [Operator Create]
   :end-before: //! [Operator Create]

.. literalinclude::  ../../../tests/t500-operator.c
   :language: c
   :start-after: //! [Operator Set]
   :end-before: //! [Operator Set]

.. literalinclude::  ../../../tests/t500-operator.c
   :language: c
   :start-after: //! [Operator Apply]
   :end-before: //! [Operator Apply]

A number of function calls in the interface, such as :c:func:`CeedOperatorApply()`, are
intended to support asynchronous execution via their last argument,
``CeedRequest*``. The specific (pointer) value used in the above example,
``CEED_REQUEST_IMMEDIATE``, is used to express the request (from the user) for the
operation to complete before returning from the function call, i.e. to make sure
that the result of the operation is available in the output parameters
immediately after the call. For a true asynchronous call, one needs to provide
the address of a user defined variable. Such a variable can be used later to
explicitly wait for the completion of the operation.


Gallery of QFunctions
----------------------------------------

LibCEED provides a gallery of built-in :ref:`CeedQFunction`\s in the :file:`gallery/` directory.
The available QFunctions are the ones associated with the mass, the Laplacian, and
the identity operators. To illustrate how the user can declare a :ref:`CeedQFunction`
via the gallery of available QFunctions, consider the selection of the
:ref:`CeedQFunction` associated with a simple 1D mass matrix
(cf. `tests/t410-qfunction.c <https://github.com/CEED/libCEED/blob/main/tests/t410-qfunction.c>`_).

.. literalinclude::  ../../../tests/t410-qfunction.c
   :language: c
   :linenos:


Interface Principles and Evolution
----------------------------------------

LibCEED is intended to be extensible via backends that are packaged with the
library and packaged separately (possibly as a binary containing proprietary
code). Backends are registered by calling

.. literalinclude::  ../../../backends/ref/ceed-ref.c
   :language: c
   :start-after: //! [Register]
   :end-before: //! [Register]

typically in a library initializer or "constructor" that runs automatically.
``CeedInit`` uses this prefix to find an appropriate backend for the resource.

Source (API) and binary (ABI) stability are important to libCEED. LibCEED is
evolving rapidly at present, but we expect it to stabilize soon at which point
we will adopt semantic versioning. User code, including libraries of
:ref:`CeedQFunction`\s, will not need to be recompiled except between major releases.
The backends currently have some dependence beyond the public user interface,
but we intent to remove that dependence and will prioritize if anyone expresses
interest in distributing a backend outside the libCEED repository.
