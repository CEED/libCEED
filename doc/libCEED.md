# libCEED Documentation

This page provides a brief description of the theoretical foundations and the
practical implementation of the libCEED library.

## Finite Element Operator Decomposition

Finite element operators are typically defined through weak formulations of
partial differential equations, that involve integration over a computational
mesh.  The required integrals are computed by splitting them as a sum over the
mesh elements, mapping each element to a simple *reference* element (e.g. the
unit square) and applying a quadrature rule in reference space.

This sequence of operations, highlights an inherent hierarchical structure
present in all finite element operators where the evaluation starts on *global
(trial) degrees of freedom (dofs) on the whole mesh*, restricts to *degrees of
freedom on subdomains* (groups of elements), then moves to independent *degrees
of freedom on each element*, transitions to independent *quadrature points* in
reference space, performs the integration, and then goes back in reverse order
to global (test) degrees of freedom on the whole mesh.

This is illustrated below, for the simple case of symmetric linear operator on
third order (Q3) scalar continuous (H1) elements, where we use the notions
**T-vector**, **L-vector**, **E-vector** and **Q-vector** to represent the sets
corresponding to the (true) degrees of freedom on the global mesh, the split
local degrees of freedom on the subdomains, the split degrees of freedom on the
mesh elements, and the values at quadrature points, respectively.

We refer to the operators that connect the different types of vectors as

- Subdomain restriction **P**
- Element restriction **G**
- Dofs-to-Qpts evaluator **B**
- Operator at quadrature points: **D**

More generally, when the test and trial space differ, they get there own version
of **P**, **G** and **B**.

![Operator Decomposition](libCEED.png "Operator Decomposition")

Note that in the case of adaptive mesh refinement (AMR), the restrictions **P**
and **G** will involve not just extracting sub-vectors, but evaluating values at
constrained degrees of freedom through the AMR interpolation. There can also be
several levels of subdomains (**P1**, **P2**, etc.)  and it may be convenient to
split **D** as the product of several operators (**D1**, **D2**, etc.).

#### Partial Assembly

Since the global operator **A** is just a series of variational restrictions
with **B**, **G** and **P**, starting from its point-wise kernel **D**, a
"matvec" with **A** can be performed by evaluating and storing some of the
innermost variational restriction matrices, and applying the rest of the
operators "on-the-fly".  For example, one can compute and store a global matrix
on T-vector level. Alternatively, one can compute and store only the subdomain
(L-vector) or element (E-vector) matrices, and perform the action of **A** using
matvecs with **P** or **P** and **G**.  While these options are natural for
low-order discretizations, they are not a good fit for high-order methods due to
the amount of FLOPs needed for their evaluation, as well as the memory
transfer needed for a matvec.

Our focus in libCEED instead is on **partial assembly**, where we compute and
store only **D** (or portions of it) and evaluate the actions of **P**, **G**
and **B** on-the-fly.  Critically for performance, we take advantage of the
tensor-product structure of the degrees of freedom and quad and hex elements to
perform the action of **B** without storing it as a matrix.

#### Parallel Decomposition

After the application of each of the first three transition operators, **P**,
**G** and **B**, the evaluation is decoupled on their range, so **P**, **G** and
**B** allows us to "zoom-in" to subdomain, element and quadrature point level,
ignoring the coupling at higher levels.

Thus, a natural mapping of **A** on a parallel computer is to split the
**T-vector** over MPI ranks (a non-overlapping decomposition, as is typically
used for sparse matrices), and then split the rest of the vector types over
computational devices (CPUs, GPUs, etc.) as indicated by the shaded regions in
the diagram above.

Currently in libCEED we allow the application to manage the global T-vectors and
the transition to/from devices with **P**. Our API is thus focused on the
L-vector level, where devices are independent and there is one device per MPI
rank (there could be different types of devices in different ranks though).

## Algebraic Interface

The above algebraic structure is applicable to other problems, not just finite
elements...

## API Description

Make connection with the concepts and functions in the code...
