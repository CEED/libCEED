.. _example-petsc-elasticity:

Solid mechanics elasticity mini-app
========================================

This example is located in the subdirectory :file:`examples/solids`.
It solves the steady-state static balance momentum equations using unstructured high-order finite/spectral element spatial discretizations.
As for the :ref:`example-petsc-navier-stokes` case, the solid mechanics elasticity example has been developed using PETSc, so that the pointwise physics (defined at quadrature points) is separated from the parallelization and meshing concerns.


.. _problem-linear-elasticity:

----------------------------------------

The strong form of the static balance of linear momentum at small strain for the three-dimensional linear elasticity problem is given by :cite:`hughes2012finite`:

.. math::
   :label: lin-elas

   \nabla \cdot \boldsymbol{\sigma} + \boldsymbol{g} = \boldsymbol{0} 


where :math:`\boldsymbol{\sigma}` and :math:`\boldsymbol{g}` are stress and forcing functions, respectively.
Integrating by parts on the divergence term, we arrive at the weak form the of equation :math:numref:`lin-elas`:

.. math::

   \int_{\Omega}{ \nabla \boldsymbol{v} \colon \boldsymbol{\sigma}} dV - \int_{d\Omega}{\boldsymbol{v} \cdot \left(\boldsymbol{\sigma}_t \cdot \hat{\boldsymbol{n}}\right)} dS + \int_{\Omega}{\boldsymbol{v} \cdot \boldsymbol{g}} dV = 0

where :math:`\boldsymbol{\sigma}_t \cdot \hat{\boldsymbol{n}}` is typically replaced with a boundary condition.

The constitutive law (stress-strain relationship) is given by:

.. math::
   :label: linear-stress-strain

   \boldsymbol{\sigma} = \mathsf{C} \!:\! \boldsymbol{\epsilon},

where 

.. math::
   :label: small-strain

   \boldsymbol{\epsilon} = \dfrac{1}{2}\left(\nabla \boldsymbol{u} + \nabla \boldsymbol{u}^T \right)

is the symmetric (small/infinitesimal) strain tensor and the colon represents a double contraction.
For notational convenience, we express the symmetric second order tensors :math:`\bm \sigma` and :math:`\bm \epsilon` as vectors of length 6 using the `Voigt notation <https://en.wikipedia.org/wiki/Voigt_notation>`_.
Hence, the fourth order elasticity tensor :math:`\mathsf C` (also known as elastic moduli tensor or material stiffness tensor) can be represented as a :math:`6\times 6` symmetric matrix

.. math::
   :label: linear-elasticity-tensor

   \mathsf C = \dfrac{E}{(1+\nu)(1-2\nu)}
   \begin{pmatrix}
     1-\nu & \nu & \nu & & & \\
     \nu & 1 - \nu & \nu & & & \\
     \nu & \nu &  1 - \nu & & & \\
     & & & \dfrac{1 - 2\nu}{2} & & \\    
     & & & &\dfrac{1 - 2\nu}{2} & \\
     & & & & & \dfrac{1 - 2\nu}{2} \\   
   \end{pmatrix},

where :math:`E` is the Young’s modulus and :math:`\nu` is the Poisson’s ratio.
An alternative formulation, in terms of the Lamé parameters,

.. math::
   \begin{aligned}
   \lambda &= \frac{E \nu}{(1 + \nu)(1 - 2 \nu)} \\
   \mu &= \frac{E}{2(1 + \nu)}
   \end{aligned}

can be found, for which the constitutive equation :math:numref:`linear-stress-strain` may be written as

.. math::
   \bm\sigma = \lambda (\operatorname{trace} \bm\epsilon) \bm I_3 + 2 \mu \bm\epsilon,

where :math:`\bm I_3` is the :math:`3 \times 3` identity matrix.
With the latter formulation, the elasticity tensor :math:numref:`linear-elasticity-tensor` becomes

.. math::

   \mathsf C = \begin{pmatrix}
   \lambda + 2\mu & \lambda & \lambda & & & \\
   \lambda & \lambda + 2\mu & \lambda & & & \\
   \lambda & \lambda & \lambda + 2\mu & & & \\
   & & & \mu & & \\
   & & & & \mu & \\
   & & & & & \mu
   \end{pmatrix}.

Note that the incompressible limit :math:`\nu \to \frac 1 2` causes :math:`\lambda \to \infty`, and thus :math:`\mathsf C` becomes singular.


.. _problem-hyper-small-strain:

Hyperelasticity at Small Strain
----------------------------------------

The the constitutive law for the small strain version of a Neo-Hookean hyperelasticity material is given as
follows:

.. math::
   :label: eq-neo-hookean-small-strain
   
   \boldsymbol{\sigma} = \lambda \log(1 + \boldsymbol{\epsilon_v)} \boldsymbol{I}_3 + 2\mu \boldsymbol{\epsilon}

where :math:`\boldsymbol{\epsilon}` is defined as in :math:numref:`small-strain`.
The trace of the strain tensor, also known as the *volumetric strain*, is denoted by :math:`\boldsymbol{\epsilon}_v = \sum_i \boldsymbol{\epsilon}_{ii}`.

Newton linearization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To derive a Newton linearization of :math:numref:`eq-neo-hookean-small-strain`, we begin by expressing the derivative,

.. math::

   d \boldsymbol{\sigma} = \dfrac{\partial \boldsymbol{\sigma}}{\partial \boldsymbol{\epsilon}} \colon d \boldsymbol{\epsilon}

where

.. math::

   d \boldsymbol{\epsilon} = \dfrac{1}{2}\left( \nabla \boldsymbol{d u} + \nabla \boldsymbol{d u}^T \right)

and 

.. math::

   d \nabla \boldsymbol{u} = \nabla \boldsymbol{d u} .

Therefore,

.. math::
   :label: derss

   d \boldsymbol{\sigma}  = \bar{\lambda} \cdot tr \left(d \boldsymbol{\epsilon} \right) \cdot \boldsymbol{I}_3 + 2\mu d \boldsymbol{\epsilon}

where we have introduced the symbol 

.. math::

   \bar{\lambda} = \dfrac{\lambda}{1 + \boldsymbol{\epsilon}_v } .

Equation :math:numref:`derss` can be written in matrix form as follows:

.. math::
   :label: mdss

   \begin{pmatrix}
     d\sigma_{11} \\
     d\sigma_{22} \\
     d\sigma_{33} \\
     d\sigma_{23} \\
     d\sigma_{13} \\
     d\sigma_{12}       
   \end{pmatrix}  = 
   \begin{pmatrix}
     2 \mu +\bar{\lambda} & \bar{\lambda} & \bar{\lambda} & & & \\
     \bar{\lambda} & 2 \mu +\bar{\lambda} & \bar{\lambda} & & & \\
     \bar{\lambda} & \bar{\lambda} & 2 \mu +\bar{\lambda} & & & \\
     & & & \mu & & \\    
     & & & & \mu & \\
     & & & & & \mu \\   
   \end{pmatrix}
   \begin{pmatrix} 
     d\epsilon_{11} \\
     d\epsilon_{22} \\
     d\epsilon_{33} \\
     d\epsilon_{23} \\
     d\epsilon_{13} \\
     d\epsilon_{12}       
   \end{pmatrix}

.. _problem-hyperelasticity-finite-strain:

Hyperelasticity at Finite Strain
----------------------------------------

In the *total Lagrangian* approach for the neo-Hookean Hyperelasticity probelm, the discrete equations are formulated with respect to the reference configuration.
In this formulation, we solve for displacement :math:`\bm u(\bm X)` in the reference frame :math:`\bm X`.
The notation for elasticity at finite strain is inspired by :cite:`holzapfel2000nonlinear` to distinguish between the current and reference configurations.
As explained in the :ref:`Common notation` section, we denote by capital letters the reference frame and by small letters the current one.

The strong form of the static balance of linear-momentum at *Finite Strain* (total Lagrangian) is given by:

.. math::
   :label: sblFinS

   \nabla_X \cdot \boldsymbol{P} + \rho_0 \boldsymbol{g} = \boldsymbol{0}
 
where the :math:`_X` in :math:`\nabla_X` indicates that the gradient is calculated with respect to the reference configuration in the finite strain regime.
:math:`\boldsymbol{P}` and :math:`\boldsymbol{g}` are the *first Piola-Kirchhoff stress* tensor and the prescribed forcing function, respectively.
:math:`\rho_0` is known as the *reference* mass density.
The tensor :math:`\bm P` is not symmetric, living in the current configuration on the left and the reference configuration on the right.

:math:`\boldsymbol{P}` can be decomposed as

.. math::
   :label: 1st2nd
   
   \boldsymbol{P} = \boldsymbol{F} \, \boldsymbol{S},

where :math:`\bm S` is the *second Piola-Kirchhoff stress* tensor, a symmetric tensor defined entirely in the reference configuration, and :math:`\boldsymbol{F} = \bm I_3 + \nabla_X \bm u` is the deformation gradient.
Different constitutive models can define :math:`\bm S`.


Constitutive modeling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In their most general form, constitutive models define :math:`\bm S` in terms of state variables.
In the model taken into consideration in the present miniapp, the state variables are constituted by the vector displacement field :math:`\bm u`, and its gradient :math:`\nabla_X \bm u`.
We begin by defining two symmetric tensors in the reference configuration, the right Cauchy-Green tensor

.. math::
   \bm C = \bm F^T \bm F

and the Green-Lagrange strain tensor

.. math::
   :label: eq-green-lagrange-strain

   \bm E = \frac 1 2 (\bm C - \bm I_3) = \frac 1 2 \Big( \nabla_X \bm u + (\nabla_X \bm u)^T + (\nabla_X \bm u)^T \nabla_X \bm u \Big),

the latter of which converges to the linear strain tensor :math:`\bm \epsilon` in the small-deformation limit.
The constitutive models considered, appropriate for large deformations, express :math:`\bm S` as a function of :math:`\bm E`, similar to the linear case, shown in equation  :math:numref:`linear-stress-strain`, which  expresses the relationship between :math:`\bm\sigma` and :math:`\bm\epsilon`.
This constitutive model :math:`\bm S(\bm E)` is a nonlinear tensor-valued function of a tensor-valued input, but an arbitrary choice of such a function will generally not be invariant under orthogonal transformations and thus will not admissible as a physical model must not depend on the coordinate system chosen to express it.
In particular, given an orthogonal transformation :math:`Q`, we desire

.. math::
   :label: elastic-invariance

   Q \bm S(\bm E) Q^T = \bm S(Q \bm E Q^T),

which means that we can change our reference frame before or after computing :math:`\bm S`, and get the same result either way.
Constitutive relations in which :math:`\bm S` is uniquely determined by :math:`\bm E` (equivalently, :math:`\bm C` or related tensors) while satisfying the invariance property :math:numref:`elastic-invariance` are known as Cauchy elastic materials.
Here, we focus on an important subset of them known as hyperelastic materials, for which we may define a strain energy density functional :math:`\Phi(\bm E) \in \mathbb R` and obtain the strain energy from its gradient,

.. math::
   :label: strain-energy-grad

   \bm S(\bm E) = \frac{\partial \Phi}{\partial \bm E}.

.. note::
   The strain energy density functional cannot be an arbitrary function :math:`\Phi(\bm E)`; it can only depend on *invariants*, scalar-valued functions :math:`\gamma` satisfying

   .. math::
      \gamma(\bm E) = \gamma(Q \bm E Q^T)

for all orthogonal matrices :math:`Q`.
Consequently, we may assume without loss of generality that :math:`\bm E` is diagonal and take its set of eigenvalues as the invariants.
It is clear that there can be only three invariants, and there are many alternate choices, such as :math:`\operatorname{trace}(\bm E), \operatorname{trace}(\bm E^2), \lvert \bm E \rvert`, and combinations thereof.
It is common in the literature for invariants to be taken from :math:`\bm C = \bm I_3 + 2 \bm E` instead of :math:`\bm E`.

For example, if we take the compressible Neo-Hookean model,

.. math::
   :label: neo-hookean-energy

   \begin{aligned}
   \Phi(\bm E) &= \frac{\lambda}{2}(\log J)^2 + \frac \mu 2 (\operatorname{trace} \bm C - 3) - \mu \log J \\
     &= \frac{\lambda}{2}(\log J)^2 + \mu \operatorname{trace} \bm E - \mu \log J,
   \end{aligned}

where :math:`J = \lvert \bm F \rvert = \sqrt{\lvert \bm C \rvert}` is the determinant of deformation (i.e., volume change) and :math:`\lambda` and :math:`\mu` are the Lamé parameters in the infinitesimal strain limit.

To evaluate :math:numref:`strain-energy-grad`, we make use of

.. math::
   \frac{\partial J}{\partial \bm E} = \frac{\partial \sqrt{\lvert \bm C \rvert}}{\partial \bm E} = \lvert \bm C \rvert^{-1/2} \lvert \bm C \rvert \bm C^{-1} = J \bm C^{-1},

where the factor of 2 has been absorbed due to :math:`\bm C = \bm I_3 + 2 \bm E.`
Carrying through the differentiation :math:numref:`strain-energy-grad` for the model :math:numref:`neo-hookean-energy`, we arrive at

.. math::
   :label: neo-hookean-stress

   \bm S = \lambda \log J \bm C^{-1} + \mu (\bm I_3 - \bm C^{-1}).

.. tip::
   An equivalent form of :math:numref:`neo-hookean-stress` is

   .. math::
      \bm S = \lambda \log J \bm C^{-1} + 2 \mu \bm C^{-1} \bm E,

   which is more numerically stable for small :math:`\bm E`, and thus preferred for computation.

.. note::
   One can linearize :math:numref:`neo-hookean-stress` around :math:`\bm E = 0`, for which :math:`\bm C = \bm I_3 + 2 \bm E \to \bm I_3` and :math:`J \to 1 + \operatorname{trace} \bm E`, therefore :math:numref:`neo-hookean-stress` reduces to

   .. math::
      :label: eq-st-venant-kirchoff

      \bm S = \lambda (\operatorname{trace} \bm E) \bm I_3 + 2 \mu \bm E,

   which is the St. Venant-Kirchoff model.
   This model can be used for geometrically nonlinear mechanics (e.g., snap-through of thin structures), but is inappropriate for large strain.

   Alternatively, one can drop geometric nonlinearities, :math:`\bm E \to \bm \epsilon` and :math:`\bm C \to \bm I_3`, while retaining the nonlinear dependence on :math:`J \to 1 + \operatorname{trace} \bm \epsilon`, thereby yielding :math:numref:`eq-neo-hookean-small-strain`.
   The effect of geometric and constitutive linearization is sketched in the diagram below.

   .. math::

      \begin{matrix}
      \text{Finite Strain Hyperelastic} & \underset{\bm S = \mathsf C \bm E}{\overset{\text{constitutive}}{\LARGE \longrightarrow}} & \text{St. Venant-Kirchoff} \\
      \text{\scriptsize geometric} {\LARGE \ \downarrow\ } \scriptsize{\bm E \to \bm \epsilon} & & \text{\scriptsize geometric} {\LARGE \ \downarrow\ } \scriptsize{\bm E \to \bm \epsilon} \\
      \text{Small Strain Hyperelastic} & \underset{\bm \sigma = \mathsf C \bm \epsilon}{\overset{\text{constitutive}}{\LARGE \longrightarrow}} & \text{Linear Elastic} \\
      \end{matrix}

Weak form
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is crucial to distinguish between the current and reference element in the *total Lagrangian* Finite Strain regime.

.. math::

    \int_{\Omega}{\boldsymbol{v} \cdot \left(\nabla_X \cdot \boldsymbol{P} + \rho_0 \boldsymbol{g}\right)} dV = \boldsymbol{0}

Integrating by parts, we arrive at the weak form:
find :math:`\bm u \in \mathcal V \equiv H^1(\Omega_0)` such that

.. math::
   :label: hyperelastic-weak-form

    \int_{\Omega}{\nabla_X \boldsymbol{v} \colon \boldsymbol{P}}dV
    + \int_{\Omega}{\boldsymbol{v} \cdot \rho_0 \boldsymbol{g}}dV
    + \int_{\partial \Omega}{\boldsymbol{v} \cdot \boldsymbol{P} \cdot \hat{\boldsymbol{N}}}dA = 0, \quad \forall \bm v \in \mathcal V,
    
where :math:`\boldsymbol{P} \cdot \hat{\boldsymbol{N}}` is replaced by any prescribed stress/traction boundary conditions written in terms of the reference configuration.
This equation contains material/constitutive nonlinearities in defining :math:`\bm S(\bm E)`, as well as geometric nonlinearities through :math:`\bm P = \bm F\, \bm S`, :math:`\bm E(\bm F)`, and the body force :math:`\bm g`, which must be pulled back from the current configuration to the reference configuration.
Discretization of :math:numref:`hyperelastic-weak-form` produces a finite-dimensional system of nonlinear algebraic equations, which we solve using Newton-Raphson methods.
One attractive feature of Galerkin discretization is that we can arrive at the same linear system by discretizing the Newton linearization of the continuous form; that is, discretization and differentiation (Newton linearization) commute.

Newton linearization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To derive a Newton linearization of :math:numref:`hyperelastic-weak-form`, we begin by expressing the derivative of :math:numref:`1st2nd` in incremental form,

.. math::
   :label: eq-diff-P

   \diff \bm P = \frac{\partial \bm P}{\partial \bm F} \!:\! \diff \bm F = \diff \bm F\, \bm S + \bm F \underbrace{\frac{\partial \bm S}{\partial \bm E} \!:\! \diff \bm E}_{\diff \bm S}

where

.. math::
   \diff \bm E = \frac{\partial \bm E}{\partial \bm F} \!:\! \diff \bm F = \frac 1 2 \Big( \diff \bm F^T \bm F + \bm F^T \diff \bm F \Big).

The quantity :math:`\frac{\partial \bm S}{\partial \bm E}` is known as the incremental elasticity tensor, and is analogous to the linear elasticity tensor :math:`\mathsf C` of :math:numref:`linear-elasticity-tensor`.
We now evaluate :math:`\diff \bm S` for the Neo-Hookean model :math:numref:`neo-hookean-stress`,

.. math::
   :label: eq-neo-hookean-incremental-stress

   \diff\bm S = \frac{\partial \bm S}{\partial \bm E} \!:\! \diff \bm E
   = \lambda (\bm C^{-1} \!:\! \diff\bm E) \bm C^{-1}
     + 2 (\mu - \lambda \log J) \bm C^{-1} \diff\bm E \, \bm C^{-1},

where we have used

.. math::
   \diff \bm C^{-1} = \frac{\partial \bm C^{-1}}{\partial \bm E} \!:\! \diff\bm E
   = -2 \bm C^{-1} \diff \bm E \, \bm C^{-1} .

.. note::
   In the small-strain limit, :math:`\bm C \to \bm I_3` and :math:`\log J \to 0`, thereby reducing :math:numref:`eq-neo-hookean-incremental-stress` to the St. Venant-Kirchoff model :math:numref:`eq-st-venant-kirchoff`.

.. note::
   Some cancellation is possible (at the expense of symmetry) if we substitute :math:numref:`eq-neo-hookean-incremental-stress` into :math:numref:`eq-diff-P`,

   .. math::
      :label: eq-diff-P-dF

      \begin{aligned}
      \diff \bm P &= \diff \bm F\, \bm S
        + \lambda (\bm C^{-1} : \diff \bm E) \bm F^{-T} + 2(\mu - \lambda \log J) \bm F^{-T} \diff\bm E \, \bm C^{-1} \\
      &= \diff \bm F\, \bm S
        + \lambda (\bm F^{-T} : \diff \bm F) \bm F^{-T} + (\mu - \lambda \log J) \bm F^{-T} (\bm F^T \diff \bm F + \diff \bm F^T \bm F) \bm C^{-1} \\
      &= \diff \bm F\, \bm S
        + \lambda (\bm F^{-T} : \diff \bm F) \bm F^{-T} + (\mu - \lambda \log J) \Big( \diff \bm F\, \bm C^{-1} + \bm F^{-T} \diff \bm F^T \bm F^{-T} \Big),
      \end{aligned}

   where we have exploited :math:`\bm F \bm C^{-1} = \bm F^{-T}` and

   .. math::
      \begin{aligned}
      \bm C^{-1} \!:\! \diff \bm E = \bm C_{IJ}^{-1} \diff \bm E_{IJ}
      &= \frac 1 2 \bm F_{Ik}^{-1} \bm F_{Jk}^{-1} (\bm F_{\ell I} \diff \bm F_{\ell J} + \diff \bm F_{\ell I} \bm F_{\ell J}) \\
      &= \frac 1 2 \Big( \delta_{\ell k} \bm F_{Jk}^{-1} \diff \bm F_{\ell J} + \delta_{\ell k} \bm F_{Ik}^{-1} \diff \bm F_{\ell I} \Big) \\
      &= \bm F_{Ik}^{-1} \diff \bm F_{kI} = \bm F^{-T} \!:\! \diff \bm F.
      \end{aligned}

   We prefer to compute with :math:numref:`eq-neo-hookean-incremental-stress` because :math:numref:`eq-diff-P-dF` is more expensive, requiring access to (nonsymmetric) :math:`\bm F^{-1}` in addition to (symmetric) :math:`\bm C^{-1} = \bm F^{-1} \bm F^{-T}`, having fewer symmetries to exploit in contractions, and being less numerically stable.

It is sometimes useful to express :math:numref:`eq-neo-hookean-incremental-stress` in index notation,

.. math::
   :label: eq-neo-hookean-incremental-stress-index

   \begin{aligned}
   \diff\bm S_{IJ} &= \frac{\partial \bm S_{IJ}}{\partial \bm E_{KL}} \diff \bm E_{KL} \\
     &= \lambda (\bm C^{-1}_{KL} \diff\bm E_{KL}) \bm C^{-1}_{IJ} + 2 (\mu - \lambda \log J) \bm C^{-1}_{IK} \diff\bm E_{KL} \bm C^{-1}_{LJ} \\
     &= \underbrace{\Big( \lambda \bm C^{-1}_{IJ} \bm C^{-1}_{KL} + 2 (\mu - \lambda \log J) \bm C^{-1}_{IK} \bm C^{-1}_{JL} \Big)}_{\mathsf C_{IJKL}} \diff \bm E_{KL} \,,
   \end{aligned}

where we have identified the effective elasticity tensor :math:`\mathsf C = \mathsf C_{IJKL}`.
It is generally not desirable to store :math:`\mathsf C`, but rather to use the earlier expressions so that only :math:`3\times 3` tensors (most of which are symmetric) must be manipulated.
That is, given the linearization point :math:`\bm F` and solution increment :math:`\diff \bm F = \nabla_X (\diff \bm u)` (which we are solving for in the Newton step), we compute :math:`\diff \bm P` via

#. recover :math:`\bm C^{-1}` and :math:`\log J` (either stored at quadrature points or recomputed),
#. proceed with :math:`3\times 3` matrix products as in :math:numref:`eq-neo-hookean-incremental-stress` or the second line of :math:numref:`eq-neo-hookean-incremental-stress-index` to compute :math:`\diff \bm S` while avoiding computation or storage of higher order tensors, and
#. conclude by :math:numref:`eq-diff-P`, where :math:`\bm S` is either stored or recomputed from its definition exactly as in the nonlinear residual evaluation.

.. note::
   The decision of whether to recompute or store functions of the current state :math:`\bm F` depend on a roofline analysis :cite:`williams2009roofline,Brown:2010` of the computation and the cost of the constitutive model.
   For low-order elements where flops tend to be in surplus relative to memory bandwidth, recomputation is likely to be preferable, where as the opposite may be true for high-order elements.
   Similarly, analysis with a simple constitutive model may see better performance while storing little or nothing while an expensive model such as Arruda-Boyce :cite:`arruda1993largestretch`, which contains many special functions, may be faster when using more storage to avoid recomputation.
   In the case where complete linearization is preferred, note the symmetry :math:`\mathsf C_{IJKL} = \mathsf C_{KLIJ}` evident in :math:numref:`eq-neo-hookean-incremental-stress-index`, thus :math:`\mathsf C` can be stored as a symmetric :math:`6\times 6` matrix, which has 21 unique entries.
   Along with 6 entries for :math:`\bm S`, this totals 27 entries of overhead compared to computing everything from :math:`\bm F`.
   This compares with 13 entries of overhead for direct storage of :math:`\{ \bm S, \bm C^{-1}, \log J \}`, which is sufficient for the Neo-Hookean model to avoid all but matrix products.
