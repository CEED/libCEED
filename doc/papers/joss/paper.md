---
title: 'libCEED: An open-source library for efficient high-order operator evaluations'
tags:
  - high-performance computing
  - high-order methods
  - finite elements
  - spectral elements
  - matrix-free
authors:
  - name: Jed Brown
    orcid: 0000-0002-9945-0639
    affiliation: 1
  - name: Jeremy Thompson
    affiliation: 1
  - name: Valeria Barra
    orcid: 0000-0003-1129-2056
    affiliation: 1
  - name: XXXX TODO Many more authors to be added XXXX
affiliations:
 - name: University of Colorado Boulder
   index: 1
date: XXXX TODO write date XXXX
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

High-order numerical methods are widely used in Partial Differential Equation (PDE)
solvers, but software packages that provide high-performance implementations have
often been special-purpose and intrusive. ``libCEED`` is an open-source light-weight
library that offers a purely algebraic interface for efficient operator evaluation
and matrix-free preconditioning ingredients. libCEED supports run-time selection of
implementations tuned for a variety of computational architectures, including CPUs and
GPUs, and can be unobtrusively integrated in new and legacy software to provide portable
performance. We introduce ``libCEED``’s conceptual framework and its low-level C
interface, together with its newly available high-level Python interface.

In finite element formulations, the weak form of a PDE is evaluated on a subdomain
(element), and the local results are composed into a larger system of equations that
models the entire problem. In particular, when high-order finite elements or spectral
elements are used, the resulting sparse matrix representation of the global operator
is computationally expensive, with respect to both the memory transfer and floating
point operations needed for its evaluation [@Brown:2010]. ``libCEED`` provides an
interface for matrix-free operator description that enables efficient evaluation on
a variety of computational device types (selectable at run time).

# libCEED's API

``libCEED``'s Application Programming Interface (API) provides the local action of the
linear or nonlinear operator without assembling its sparse representation. Let us
define the global operator as

$$
\begin{align}\label{eq:decomposition}
A = P^T \underbrace{G^T B^T D B G}_{\text{libCEED}} P \, ,
\end{align}
$$

where $P$ is the parallel process decomposition operator (external to ``libCEED``) in
which the degrees of freedom (DOFs) are scattered to and gathered from the different
compute devices. The operator denoted by $A_L = G^T B^T D B G$ gives the local action
on a compute node or process, where $G$ is a local element restriction operation that
localizes DOFs based on the elements, $B$ defines the action of the basis functions
(or their gradients) on the nodes, and $D$ is the user-defined pointwise function
describing the physics of the problem at the quadrature points, also called the
QFunction. QFunctions, which can either be defined by the user or selected from a
gallery of available built-in functions in the library, are pointwise functions
that do not depend on element resolution, topology, or basis degree (selectable
at run time).

To achieve high performance, libCEED can take advantage of a tensor-product
finite-element basis and quadrature rule to apply the action of the basis
operator $B$ or, alternatively, efficiently operate on bases that are defined
on arbitrary-topology elements. Furthermore, the algebraic decomposition described in
Eq. (\ref{eq:decomposition}) can represent either linear/nonlinear or
symmetric/asymmetric operators and exposes opportunities for device-specific
optimizations.

![libCEED is a low-level API for finite element codes, that has specialized implementations
(backends) for heterogeneous architectures.\label{fig:libCEEDBackends}](libCEEDBackends.pdf)

Fig. \ref{fig:libCEEDBackends} shows a subset of the backend implementations available
in ``libCEED``. GPU implementations are available via pure CUDA as well as the OCCA
and MAGMA libraries. CPU implementations are available via pure C and AVX intrinsics
as well as the LIBXSMM library. libCEED provides a unified interface, so that users
only need to write a single source code and can select the desired specialized
implementation at run time. Moreover, each process or thread can instantiate an
arbitrary number of backends.


# Acknowledgements

We acknowledge the US Department of Energy (DOE) Exascale Computing Project (ECP)
(17-SC-20-SC) and thank all collaborators.

# References
