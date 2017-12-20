# libCEED Documentation

This page provides a brief description of the theoretical foundations and the
practical implementation of the libCEED library.

## Finite Element Operator Decomposition

Finite element operators are typically defined through weak formulations of
partial differential equations that involve integration over a computational
mesh.  The required integrals are computed by splitting them as a sum over the
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
- Dofs-to-Qpts evaluator **B**
- Operator at quadrature points **D**

More generally, when the test and trial space differ, they get their own
versions of **P**, **G** and **B**.

![Operator Decomposition](libCEED.png "Operator Decomposition")

Note that in the case of adaptive mesh refinement (AMR), the restrictions **P**
and **G** will involve not just extracting sub-vectors, but evaluating values at
constrained degrees of freedom through the AMR interpolation. There can also be
several levels of subdomains (**P1**, **P2**, etc.),  and it may be convenient to
split **D** as the product of several operators (**D1**, **D2**, etc.).

### Partial Assembly

Since the global operator **A** is just a series of variational restrictions
with **B**, **G** and **P**, starting from its point-wise kernel **D**, a
"matvec" with **A** can be performed by evaluating and storing some of the
innermost variational restriction matrices, and applying the rest of the
operators "on-the-fly".  For example, one can compute and store a global matrix
on T-vector level. Alternatively, one can compute and store only the subdomain
(L-vector) or element (E-vector) matrices and perform the action of **A** using
matvecs with **P** or **P** and **G**.  While these options are natural for
low-order discretizations, they are not a good fit for high-order methods due to
the amount of FLOPs needed for their evaluation, as well as the memory
transfer needed for a matvec.

Our focus in libCEED instead is on **partial assembly**, where we compute and
store only **D** (or portions of it) and evaluate the actions of **P**, **G**
and **B** on-the-fly.  Critically for performance, we take advantage of the
tensor-product structure of the degrees of freedom and quad and hex elements to
perform the action of **B** without storing it as a matrix.

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

Currently in libCEED, we allow the application to manage the global T-vectors and
the transition to/from devices with **P**. Our API is thus focused on the
L-vector level, where devices are independent and there is one device per MPI
rank (there could be different types of devices in different ranks though).

## API Description

The libCEED API takes an algebraic approach, where the user essentially
describes in the *front-end* the operators **G**, **B** and **D** and the library
provides *back-end* implementations and coordinates their action to the original
operator on L-vector level (i.e. independently on each device / MPI task).

One of the advantages of this purely algebraic description is that it already
includes all the finite element information, so the back-ends can operate on
linear algebra level without explicit finite element code. The front-end
description is general enough to support a wide variety of finite element
algorithms, as well as some other types algorithms such as spectral finite
differences. The separation of the front- and back-ends enables applications to
easily switch/try different back-ends. It also enables back-end developers to
impact many applications from a single implementation.

Our long-term vision is to include a variety of back-end implementations in
libCEED, ranging from reference kernels to highly optimized kernels targeting
specific devices (e.g. GPUs) or specific polynomial orders. A simple reference
back-end implementation is provided in the file
[ceed-ref.c](https://github.com/CEED/libCEED/blob/master/ceed-ref.c).

On the front-end, the mapping between the decomposition concepts and the code
implementation is as follows:
- L-, E- and Q-vector are represented as variables of type `CeedVector`.
- **G** is represented as variable of type `CeedElemRestriction`.
- **B** is represented as variable of type `CeedBasis`.
- the action of **D** is represented as variable of type `CeedQFunction`.
- the overall operator **G^T B^T D B G** is represented as variable of type
  `CeedOperator` and its action is accessible through `CeedOperatorApply()`.

To clarify these concepts and illustrate how they are combined in the API,
consider the implementation of the action of a simple 1D mass matrix
(cf. [tests/t30-operator.c](https://github.com/CEED/libCEED/blob/master/tests/t30-operator.c)).

```c
#include <ceed.h>

static int setup(void *ctx, void *qdata, CeedInt Q, const CeedScalar *const *u,
                 CeedScalar *const *v) {
  CeedScalar *w = qdata;
  for (CeedInt i=0; i<Q; i++) {
    w[i] = u[0][i];
  }
  return 0;
}

static int mass(void *ctx, void *qdata, CeedInt Q, const CeedScalar *const *u,
                CeedScalar *const *v) {
  const CeedScalar *w = qdata;
  for (CeedInt i=0; i<Q; i++) {
    v[0][i] = w[i] * u[0][i];
  }
  return 0;
}

int main(int argc, char **argv) {
  Ceed ceed;
  CeedElemRestriction Erestrictx, Erestrictu;
  CeedBasis bx, bu;
  CeedQFunction qf_setup, qf_mass;
  CeedOperator op_setup, op_mass;
  CeedVector qdata, X, U, V;
  CeedInt nelem = 5, P = 5, Q = 8;
  CeedInt Nx = nelem+1, Nu = nelem*(P-1)+1;
  CeedInt indx[nelem*2], indu[nelem*P];
  CeedScalar x[Nx];

  CeedInit("/cpu/self", &ceed);
  for (CeedInt i=0; i<Nx; i++) x[i] = i / (Nx - 1);
  for (CeedInt i=0; i<nelem; i++) {
    indx[2*i+0] = i;
    indx[2*i+1] = i+1;
  }
  CeedElemRestrictionCreate(ceed, nelem, 2, Nx, CEED_MEM_HOST, CEED_USE_POINTER,
                            indx, &Erestrictx);

  for (CeedInt i=0; i<nelem; i++) {
    for (CeedInt j=0; j<P; j++) {
      indu[P*i+j] = i*(P-1) + j;
    }
  }
  CeedElemRestrictionCreate(ceed, nelem, P, Nu, CEED_MEM_HOST, CEED_USE_POINTER,
                            indu, &Erestrictu);

  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, Q, CEED_GAUSS, &bx);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, P, Q, CEED_GAUSS, &bu);

  CeedQFunctionCreateInterior(ceed, 1, 1, sizeof(CeedScalar),
                              CEED_EVAL_WEIGHT, CEED_EVAL_NONE,
                              setup, __FILE__ ":setup", &qf_setup);
  CeedQFunctionCreateInterior(ceed, 1, 1, sizeof(CeedScalar),
                              CEED_EVAL_INTERP, CEED_EVAL_INTERP,
                              mass, __FILE__ ":mass", &qf_mass);

  CeedOperatorCreate(ceed, Erestrictx, bx, qf_setup, NULL, NULL, &op_setup);
  CeedOperatorCreate(ceed, Erestrictu, bu, qf_mass, NULL, NULL, &op_mass);

  CeedVectorCreate(ceed, Nx, &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);
  CeedOperatorGetQData(op_setup, &qdata);
  CeedOperatorApply(op_setup, qdata, X, NULL, CEED_REQUEST_IMMEDIATE);

  CeedVectorCreate(ceed, Nu, &U);
  CeedVectorCreate(ceed, Nu, &V);
  CeedOperatorApply(op_mass, qdata, U, V, CEED_REQUEST_IMMEDIATE);

  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_mass);
  CeedElemRestrictionDestroy(&Erestrictu);
  CeedElemRestrictionDestroy(&Erestrictx);
  CeedBasisDestroy(&bu);
  CeedBasisDestroy(&bx);
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedDestroy(&ceed);
  return 0;
}
```

The `setup` routine above computes and stores **D**, in this case a scalar value
in each quadrature point, while `mass` uses these saved values to perform the
action of **D**. These functions are turned into the `CeedQFunction` variables
`qf_setup` and `qf_mass` in the `CeedQFunctionCreateInterior()` calls:

```c
int setup(void *ctx, void *qdata, CeedInt Q, const CeedScalar *const *u, CeedScalar *const *v);
int mass(void *ctx, void *qdata, CeedInt Q, const CeedScalar *const *u, CeedScalar *const *v);

{
  CeedQFunction qf_setup, qf_mass;

  CeedQFunctionCreateInterior(ceed, 1, 1, sizeof(CeedScalar),
                              CEED_EVAL_WEIGHT, CEED_EVAL_NONE,
                              setup, __FILE__ ":setup", &qf_setup);
  CeedQFunctionCreateInterior(ceed, 1, 1, sizeof(CeedScalar),
                              CEED_EVAL_INTERP, CEED_EVAL_INTERP,
                              mass, __FILE__ ":mass", &qf_mass);
}
```

The **B** operators for the mesh nodes, `bx`, and the unknown field, `bu`, are
defined in the `CeedBasisCreateTensorH1Lagrange` calls. In this case, both the
mesh and unknown field use H1 Lagrange finite elements of order 1 and 4
respectively (the `P` argument represents the number of 1D degrees of
freedom on each element). Both basis operators use the same integration rule,
which is Gauss-Legendre with 8 points (the `Q` argument).

```c
  CeedBasis bx, bu;

  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, Q, CEED_GAUSS, &bx);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, P, Q, CEED_GAUSS, &bu);
```

The **G** operators for the mesh nodes, `Erestrictx`, and the unknown field,
`Erestrictu`, are specified in the `CeedElemRestrictionCreate()`. Both of these
specify directly the dof indices for each element in the `indx` and `indu`
arrays:

```c
  CeedInt indx[nelem*2], indu[nelem*P];

  /* indx[i] = ...; indu[i] = ...; */

  CeedElemRestrictionCreate(ceed, nelem, 2, Nx, CEED_MEM_HOST, CEED_USE_POINTER,
                            indx, &Erestrictx);
  CeedElemRestrictionCreate(ceed, nelem, P, Nu, CEED_MEM_HOST, CEED_USE_POINTER,
                            indu, &Erestrictu);
```

With partial assembly, we first perform a setup stage where **D** is evaluated
and stored. This is accomplished by the operator `op_setup` and its application
to `X`, the nodes of the mesh (these are needed to compute Jacobians at
quadrature points). Note that the corresponding `CeedOperatorApply` has only
input (the output is `NULL`):

```c
  CeedVectorCreate(ceed, Nx, &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);
  CeedOperatorGetQData(op_setup, &qdata);
  CeedOperatorApply(op_setup, qdata, X, NULL, CEED_REQUEST_IMMEDIATE);
```

The action of the operator is then represented by operator `op_mass` and its
`CeedOperatorApply` to the input L-vector `U` with output in `V`:

```c
  CeedVectorCreate(ceed, Nu, &U);
  CeedVectorCreate(ceed, Nu, &V);
  CeedOperatorApply(op_mass, qdata, U, V, CEED_REQUEST_IMMEDIATE);
```
