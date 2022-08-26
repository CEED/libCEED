(common-notation)=

# Common notation

For most of our examples, the spatial discretization uses high-order finite elements/spectral elements, namely, the high-order Lagrange polynomials defined over $P$ non-uniformly spaced nodes, the Gauss-Legendre-Lobatto (GLL) points, and quadrature points $\{q_i\}_{i=1}^Q$, with corresponding weights $\{w_i\}_{i=1}^Q$ (typically the ones given by Gauss or Gauss-Lobatto quadratures, that are built in the library).

We discretize the domain, $\Omega \subset \mathbb{R}^d$ (with $d=1,2,3$, typically) by letting $\Omega = \bigcup_{e=1}^{N_e}\Omega_e$, with $N_e$ disjoint elements.
For most examples we use unstructured meshes for which the elements are hexahedra (although this is not a requirement in libCEED).

The physical coordinates are denoted by $\bm{x}=(x,y,z) \equiv (x_0,x_1,x_2) \in\Omega_e$, while the reference coordinates are represented as $\bm{X}=(X,Y,Z) \equiv (X_0,X_1,X_2) \in \textrm{I}=[-1,1]^3$ (for $d=3$).
