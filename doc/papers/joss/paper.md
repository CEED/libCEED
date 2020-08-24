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
  - name: Ahmad Abdelfattah
    orcid: 0000-0001-5054-4784
    affiliation: 3
  - name: Valeria Barra
    orcid: 0000-0003-1129-2056
    affiliation: 1
  - name: Natalie Beams
    orcid: 0000-0001-6060-4082
    affiliation: 3
  - name: Jean-Sylvain Camier
    orcid: 0000-0003-2421-1999
    affiliation: 2
  - name: Veselin Dobrev
    orcid: 0000-0003-1793-5622
    affiliation: 2
  - name: Yohann Dudouit
    orcid: 0000-0001-5831-561X
    affiliation: 2
  - name:  Leila Ghaffari
    orcid: 0000-0002-0965-214X
    affiliation: 1
  - name: Tzanio Kolev
    orcid: 0000-0002-2810-3090
    affiliation: 2
  - name: David Medina
    affiliation: 4
  - name: Thilina Rathnayake
    affiliation: 5
  - name: Jeremy Thompson
    orcid: 0000-0003-2980-0899
    affiliation: 1
  - name: Stan Tomov
    orcid: 0000-0002-5937-7959
    affiliation: 3
affiliations:
 - name: University of Colorado at Boulder
   index: 1
 - name: Lawrence Livermore National Laboratory
   index: 2
 - name: University of Tennessee
   index: 3
 - name: Occalytics LLC
   index: 4
 - name: University of Illinois at Urbana-Champaign
   index: 5
date: XXXX TODO write date XXXX
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

High-order numerical methods are widely used in Partial Differential Equation (PDE) solvers, but software packages that provide high-performance implementations have often been special-purpose and intrusive.
``libCEED``, the Code for Efficient Extensible Discretizations, is a new lightweight, open-source, matrix-free Finite Element library that offers a purely algebraic interface for efficient operator evaluation and preconditioning ingredients [@libceed-dev-site].
libCEED supports run-time selection of implementations tuned for a variety of computational architectures, including CPUs and GPUs, and can be unobtrusively integrated in new and legacy software to provide portable performance. We introduce libCEED’s conceptual framework and interface, and show examples of its integration with other packages, such as PETSc [@PETScUserManual], MFEM [@MFEMlibrary], and Nek5000 [@Nekwebsite].

In finite element formulations, the weak form of a PDE is evaluated on a subdomain (element), and the local results are composed into a larger system of equations that models the entire problem.
In particular, when high-order finite elements or spectral elements are used, the resulting sparse matrix representation of the global operator is computationally expensive, with respect to both the memory transfer and floating point operations needed for its evaluation [@Orszag:1980; @Brown:2010].
libCEED provides an interface for matrix-free operator description that enables efficient evaluation on a variety of computational device types (selectable at run time).

# libCEED's API

``libCEED``'s Application Programming Interface (API) provides the local action of the
linear or nonlinear operator without assembling its sparse representation. Let us
define the global operator as

\begin{align}\label{eq:decomposition}
A = P^T \underbrace{G^T B^T D B G}_{\text{libCEED}} P \, ,
\end{align}

where $P$ is the parallel process decomposition operator (external to libCEED) in
which the degrees of freedom (DOFs) are scattered to and gathered from the different
compute devices. The operator denoted by $A_L = G^T B^T D B G$ gives the local action
on a compute node or process, where $G$ is a local element restriction operation that
localizes DOFs based on the elements, $B$ defines the action of the basis functions
(or their gradients) on the nodes, and $D$ is the user-defined pointwise function
describing the physics of the problem at the quadrature points, also called the
QFunction. QFunctions, which can either be defined by the user or selected from a
gallery of available built-in functions in the library, are pointwise functions
that do not depend on element resolution, topology, or basis degree (selectable
at run time). This easily allows $hp$-refinement studies (where $h$ commonly denotes the average element size and $p$ the polynomial degree of the basis functions in 1D) and $p$-multigrid solvers. libCEED also supports composition of different operators for multiphysics problems and mixed-element meshes (see Fig. \ref{fig:schematic}).

![A schematic of element restriction and basis applicator operators for
elements with different topology. This sketch shows the independence of QFunctions
(in this case representing a Laplacian) element resolution, topology, or basis degree.\label{fig:schematic}](img/QFunctionSketch.pdf)

LibCEED is a C99 library with Fortran77 and Python interfaces. The Python interface was developed using the C Foreign Function Interface (CFFI) for Python. CFFI allows to reuse most of the C declarations and requires only a minimal adaptation of some of them. The C and Python APIs are mapped in a nearly 1:1 correspondence. For instance, a ``CeedVector`` object is exposed as ``libceed.Vector`` in Python, and may reference memory that is also accessed via Python arrays from the NumPy [@NumPy] or Numba [@Numba] packages, for handling host or device memory (when interested in GPU computations with CUDA). Flexible pointer handling in libCEED makes it easy to provide zero-copy host and (GPU) device support for any desired Python array container. The interested reader can find more details on libCEED's Python interface in [@libceed-paper-proc-scipy-2020].

To achieve high performance, libCEED can take advantage of a tensor-product
finite-element basis and quadrature rule to apply the action of the basis
operator $B$ or, alternatively, efficiently operate on bases that are defined
on arbitrary-topology elements. Furthermore, the algebraic decomposition described in
Eq. (\ref{eq:decomposition}) can represent either linear/nonlinear or
symmetric/asymmetric operators and exposes opportunities for device-specific
optimizations.

![libCEED is a low-level API for finite element codes, that has specialized implementations
(backends) for heterogeneous architectures.\label{fig:libCEEDBackends}](img/libCEEDBackends.pdf)

Fig. \ref{fig:libCEEDBackends} shows a subset of the backend implementations (backends) available in libCEED and its role, as a low-level library that allows a wide variety of applications to share highly optimized discretization kernels.
GPU implementations are available via pure CUDA [@CUDAwebsite] and pure HIP [@HIPwebsite] as well as the OCCA [@OCCAwebsite] and MAGMA [@MAGMAwebsite] libraries. CPU implementations are available via pure C and AVX intrinsics as well as the LIBXSMM library [@LIBXSMM]. libCEED provides a unified interface, so that users only need to write a single source code and can select the desired specialized implementation at run time. Moreover, each process or thread can instantiate an arbitrary number of backends.

# Performance Benchmarks

The Center for Efficient Exascale Discretizations (CEED), part of the Exascale Computing Project (ECP) uses Benchmark Problems (BPs) to test and compare the performance of high-order finite element implementations [@Fischer2020scalability]. We present here the performance of libCEED's LIBXSMM blocked backend on a 2x AMD EPYC 7452 (32-core) CPU 2.35GHz. In Fig. \ref{fig:NoetherxsmmBP1}, we measure performance over 20 iterations of unpreconditioned Conjugate Gradient (CG) for the mass operator and plot throughput, for different values of the polynomial degree $p$. In Fig. \ref{fig:NoetherxsmmBP3}, we show the measured performance to solve a Poisson's problem. For both problems the throughput is plotted versus execution time per iteration (on the left panel) and Finite Element points per compute node (on the right panel). For these tests, we use a 3D domain discretized with unstructured meshes.

![BP1 (mass operator) solved with the \texttt{xsmm/blocked} backend on
a 2x AMD EPYC 7452 (32-core) CPU 2.35GHz.\label{fig:NoetherxsmmBP1}](img/BP1.pdf)

![BP3 (Poisson's problem) solved with the \texttt{xsmm/blocked} backend on
a 2x AMD EPYC 7452 (32-core) CPU 2.35GHz.\label{fig:NoetherxsmmBP3}](img/BP3.pdf)

# Applications

To highlight the ease of library reuse for solver composition and leverage libCEED's full capability for real-world applications, libCEED comes with a suite of application examples, including problems of interest to the fluid dynamics and continuum mechanics communities.

Examples of integration of libCEED with other packages in the co-design Center for Efficient Exascale Discretizations (CEED) [@CEEDwebsite], such as PETSc, MFEM, and Nek5000, can be found in the CEED distribution, which provides the full CEED software ecosystem [@CEEDMS25; @CEEDMS34].

# Acknowledgements

This research is supported by the Exascale Computing Project (17-SC-20-SC), a collaborative effort of two U.S. Department of Energy organizations (Office of Science and the National Nuclear Security Administration) responsible for the planning and preparation of a capable exascale ecosystem, including software, applications, hardware, advanced system engineering and early testbed platforms, in support of the nations exascale computing imperative.

# References
