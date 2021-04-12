.. _example-petsc-elasticity:

Solid mechanics mini-app
========================

This example is located in the subdirectory :file:`examples/solids`.
It solves the steady-state static momentum balance equations using unstructured high-order finite/spectral element spatial discretizations.
As for the :ref:`example-petsc-navier-stokes` case, the solid mechanics elasticity example has been developed using PETSc, so that the pointwise physics (defined at quadrature points) is separated from the parallelization and meshing concerns.

In this mini-app, we consider three formulations used in solid mechanics applications: linear elasticity, Neo-Hookean hyperelasticity at small strain, and Neo-Hookean hyperelasticity at finite strain.
We provide the strong and weak forms of static balance of linear momentum in the small strain and finite strain regimes.
The stress-strain relationship (constitutive law) for each of the material models is provided.
Due to the nonlinearity of material models in Neo-Hookean hyperelasticity, the Newton linearization of the material models is provided.

.. note::

   Linear elasticity and small-strain hyperelasticity can both by obtained from the finite-strain hyperelastic formulation by linearization of geometric and constitutive nonlinearities.
   The effect of these linearizations is sketched in the diagram below, where :math:`\bm \sigma` and :math:`\bm \epsilon` are stress and strain, respectively, in the small strain regime, while :math:`\bm S` and :math:`\bm E` are their finite-strain generalizations (second Piola-Kirchoff tensor and Green-Lagrange strain tensor, respectively) defined in the initial configuration, and :math:`\mathsf C` is a linearized constitutive model.

   .. math::
      :label: hyperelastic-cd

      \begin{CD}
        {\overbrace{\bm S(\bm E)}^{\text{Finite Strain Hyperelastic}}}
        @>{\text{constitutive}}>{\text{linearization}}>
        {\overbrace{\bm S = \mathsf C \bm E}^{\text{St. Venant-Kirchoff}}} \\
        @V{\text{geometric}}V{\begin{smallmatrix}\bm E \to \bm \epsilon \\ \bm S \to \bm \sigma \end{smallmatrix}}V
        @V{\begin{smallmatrix}\bm E \to \bm \epsilon \\ \bm S \to \bm \sigma \end{smallmatrix}}V{\text{geometric}}V \\
        {\underbrace{\bm \sigma(\bm \epsilon)}_\text{Small Strain Hyperelastic}}
        @>{\text{constitutive}}>\text{linearization}>
        {\underbrace{\bm \sigma = \mathsf C \bm \epsilon}_\text{Linear Elastic}}
      \end{CD}


.. _running-elasticity:

Running the mini-app
----------------------------------------

.. only:: html

   .. toctree::
      :hidden:

      README

.. include:: README.rst
   :start-after: inclusion-solids-marker


.. _problem-linear-elasticity:

Linear Elasticity
----------------------------------------

The strong form of the static balance of linear momentum at small strain for the three-dimensional linear elasticity problem is given by :cite:`hughes2012finite`:

.. math::
   :label: lin-elas

   \nabla \cdot \bm{\sigma} + \bm{g} = \bm{0}

where :math:`\bm{\sigma}` and :math:`\bm{g}` are stress and forcing functions, respectively.
We multiply :math:numref:`lin-elas` by a test function :math:`\bm v` and integrate the divergence term by parts to arrive at the weak form: find :math:`\bm u \in \mathcal V \subset H^1(\Omega)` such that

.. math::
   :label: lin-elas-weak

   \int_{\Omega}{ \nabla \bm{v} \tcolon \bm{\sigma}} \, dV
   - \int_{\partial \Omega}{\bm{v} \cdot \left(\bm{\sigma} \cdot \hat{\bm{n}}\right)} \, dS
   - \int_{\Omega}{\bm{v} \cdot \bm{g}} \, dV
   = 0, \quad \forall \bm v \in \mathcal V,

where :math:`\bm{\sigma} \cdot \hat{\bm{n}}|_{\partial \Omega}` is replaced by an applied force/traction boundary condition written in terms of the initial configuration.
When inhomogeneous Dirichlet boundary conditions are present, :math:`\mathcal V` is an affine space that satisfies those boundary conditions.

Constitutive modeling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In their most general form, constitutive models define :math:`\bm \sigma` in terms of state variables.
In the model taken into consideration in the present mini-app, the state variables are constituted by the vector displacement field :math:`\bm u`, and its gradient :math:`\nabla \bm u`.
We begin by defining the symmetric (small/infintesimal) strain tensor as

.. math::
   :label: small-strain

   \bm{\epsilon} = \dfrac{1}{2}\left(\nabla \bm{u} + \nabla \bm{u}^T \right).

This constitutive model :math:`\bm \sigma(\bm \epsilon)` is a linear tensor-valued function of a tensor-valued input, but we will consider the more general nonlinear case in other models below.
In these cases, an arbitrary choice of such a function will generally not be invariant under orthogonal transformations and thus will not admissible as a physical model must not depend on the coordinate system chosen to express it.
In particular, given an orthogonal transformation :math:`Q`, we desire

.. math::
   :label: elastic-invariance

   Q \bm \sigma(\bm \epsilon) Q^T = \bm \sigma(Q \bm \epsilon Q^T),

which means that we can change our reference frame before or after computing :math:`\bm \sigma`, and get the same result either way.
Constitutive relations in which :math:`\bm \sigma` is uniquely determined by :math:`\bm \epsilon` while satisfying the invariance property :math:numref:`elastic-invariance` are known as Cauchy elastic materials.
Here, we define a strain energy density functional :math:`\Phi(\bm \epsilon) \in \mathbb R` and obtain the strain energy from its gradient,

.. math::
   :label: strain-energy-grad

   \bm \sigma(\bm \epsilon) = \frac{\partial \Phi}{\partial \bm \epsilon}.


.. note::
   The strain energy density functional cannot be an arbitrary function :math:`\Phi(\bm \epsilon)`; it can only depend on *invariants*, scalar-valued functions :math:`\gamma` satisfying

   .. math::
      \gamma(\bm \epsilon) = \gamma(Q \bm \epsilon Q^T)

   for all orthogonal matrices :math:`Q`.

For the linear elasticity model, the strain energy density is given by

.. math::

   \bm{\Phi} = \frac{\lambda}{2} (\operatorname{trace} \bm{\epsilon})^2 + \mu \bm{\epsilon} : \bm{\epsilon} .

The constitutive law (stress-strain relationship) is therefore given by its gradient,

.. math::
   \bm\sigma = \lambda (\operatorname{trace} \bm\epsilon) \bm I_3 + 2 \mu \bm\epsilon,

where :math:`\bm I_3` is the :math:`3 \times 3` identity matrix, the colon represents a double contraction (over both indices of :math:`\bm \epsilon`), and the Lamé parameters are given by

.. math::
   \begin{aligned}
   \lambda &= \frac{E \nu}{(1 + \nu)(1 - 2 \nu)} \\
   \mu &= \frac{E}{2(1 + \nu)}
   \end{aligned}.

The constitutive law (stress-strain relationship) can also be written as

.. math::
   :label: linear-stress-strain

   \bm{\sigma} = \mathsf{C} \!:\! \bm{\epsilon}.

For notational convenience, we express the symmetric second order tensors :math:`\bm \sigma` and :math:`\bm \epsilon` as vectors of length 6 using the `Voigt notation <https://en.wikipedia.org/wiki/Voigt_notation>`_.
Hence, the fourth order elasticity tensor :math:`\mathsf C` (also known as elastic moduli tensor or material stiffness tensor) can be represented as

.. math::
   :label: linear-elasticity-tensor

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

The strong and weak forms given above, in :math:numref:`lin-elas` and :math:numref:`lin-elas-weak`, are valid for Neo-Hookean hyperelasticity at small strain.
However, the strain energy density differs and is given by

.. math::
   
   \bm{\Phi} = \lambda (1 + \operatorname{trace} \bm{\epsilon}) (\log(1 + \operatorname{trace} \bm\epsilon) - 1) + \mu \bm{\epsilon} : \bm{\epsilon} .
   
As above, we have the corresponding constitutive law given by

.. math::
   :label: eq-neo-hookean-small-strain
   
   \bm{\sigma} = \lambda \log(1 + \operatorname{trace} \bm\epsilon) \bm{I}_3 + 2\mu \bm{\epsilon}

where :math:`\bm{\epsilon}` is defined as in :math:numref:`small-strain`.


Newton linearization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Due to nonlinearity in the constitutive law, we require a Newton linearization of :math:numref:`eq-neo-hookean-small-strain`.
To derive the Newton linearization, we begin by expressing the derivative,

.. math::

   \diff \bm{\sigma} = \dfrac{\partial \bm{\sigma}}{\partial \bm{\epsilon}} \tcolon \diff \bm{\epsilon}

where

.. math::

   \diff \bm{\epsilon} = \dfrac{1}{2}\left( \nabla \diff \bm{u} + \nabla \diff \bm{u}^T \right)

and 

.. math::

   \diff \nabla \bm{u} = \nabla \diff \bm{u} .

Therefore,

.. math::
   :label: derss

   \diff \bm{\sigma}  = \bar{\lambda} \cdot \operatorname{trace} \diff \bm{\epsilon} \cdot \bm{I}_3 + 2\mu \diff \bm{\epsilon}

where we have introduced the symbol

.. math::

   \bar{\lambda} = \dfrac{\lambda}{1 + \epsilon_v }

where volumetric strain is given by :math:`\epsilon_v = \sum_i \epsilon_{ii}`.

Equation :math:numref:`derss` can be written in Voigt matrix notation as follows:

.. math::
   :label: mdss

   \begin{pmatrix}
     \diff \sigma_{11} \\
     \diff \sigma_{22} \\
     \diff \sigma_{33} \\
     \diff \sigma_{23} \\
     \diff \sigma_{13} \\
     \diff \sigma_{12}
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
     \diff \epsilon_{11} \\
     \diff \epsilon_{22} \\
     \diff \epsilon_{33} \\
     2 \diff \epsilon_{23} \\
     2 \diff \epsilon_{13} \\
     2 \diff \epsilon_{12}
   \end{pmatrix}.


.. _problem-hyperelasticity-finite-strain:

Hyperelasticity at Finite Strain
----------------------------------------

In the *total Lagrangian* approach for the Neo-Hookean hyperelasticity problem, the discrete equations are formulated with respect to the initial configuration.
In this formulation, we solve for displacement :math:`\bm u(\bm X)` in the reference frame :math:`\bm X`.
The notation for elasticity at finite strain is inspired by :cite:`holzapfel2000nonlinear` to distinguish between the current and initial configurations.
As explained in the :ref:`Common notation` section, we denote by capital letters the reference frame and by small letters the current one.

The strong form of the static balance of linear-momentum at *finite strain* (total Lagrangian) is given by:

.. math::
   :label: sblFinS

   - \nabla_X \cdot \bm{P} - \rho_0 \bm{g} = \bm{0}
 
where the :math:`_X` in :math:`\nabla_X` indicates that the gradient is calculated with respect to the initial configuration in the finite strain regime.
:math:`\bm{P}` and :math:`\bm{g}` are the *first Piola-Kirchhoff stress* tensor and the prescribed forcing function, respectively.
:math:`\rho_0` is known as the *initial* mass density.
The tensor :math:`\bm P` is not symmetric, living in the current configuration on the left and the initial configuration on the right.

:math:`\bm{P}` can be decomposed as

.. math::
   :label: 1st2nd
   
   \bm{P} = \bm{F} \, \bm{S},

where :math:`\bm S` is the *second Piola-Kirchhoff stress* tensor, a symmetric tensor defined entirely in the initial configuration, and :math:`\bm{F} = \bm I_3 + \nabla_X \bm u` is the deformation gradient.
Different constitutive models can define :math:`\bm S`.


Constitutive modeling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the constitutive modeling of hyperelasticity at finite strain, we begin by defining two symmetric tensors in the initial configuration, the right Cauchy-Green tensor

.. math::
   \bm C = \bm F^T \bm F

and the Green-Lagrange strain tensor

.. math::
   :label: eq-green-lagrange-strain

   \bm E = \frac 1 2 (\bm C - \bm I_3) = \frac 1 2 \Big( \nabla_X \bm u + (\nabla_X \bm u)^T + (\nabla_X \bm u)^T \nabla_X \bm u \Big),

the latter of which converges to the linear strain tensor :math:`\bm \epsilon` in the small-deformation limit.
The constitutive models considered, appropriate for large deformations, express :math:`\bm S` as a function of :math:`\bm E`, similar to the linear case, shown in equation  :math:numref:`linear-stress-strain`, which  expresses the relationship between :math:`\bm\sigma` and :math:`\bm\epsilon`.

Recall that the strain energy density functional can only depend upon invariants.
We will assume without loss of generality that :math:`\bm E` is diagonal and take its set of eigenvalues as the invariants.
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

where the factor of :math:`\frac 1 2` has been absorbed due to :math:`\bm C = \bm I_3 + 2 \bm E.`
Carrying through the differentiation :math:numref:`strain-energy-grad` for the model :math:numref:`neo-hookean-energy`, we arrive at

.. math::
   :label: neo-hookean-stress

   \bm S = \lambda \log J \bm C^{-1} + \mu (\bm I_3 - \bm C^{-1}).

.. tip::
   An equivalent form of :math:numref:`neo-hookean-stress` is

   .. math::
      :label: neo-hookean-stress-stable

      \bm S = \lambda \log J \bm C^{-1} + 2 \mu \bm C^{-1} \bm E,

   which is more numerically stable for small :math:`\bm E`, and thus preferred for computation.
   Note that the product :math:`\bm C^{-1} \bm E` is also symmetric, and that :math:`\bm E` should be computed using :math:numref:`eq-green-lagrange-strain`.

   Similarly, it is preferable to compute :math:`\log J` using ``log1p``, especially in case of nearly incompressible materials.
   To sketch this idea, suppose we have the :math:`2\times 2` non-symmetric matrix :math:`\bm{F} = \left( \begin{smallmatrix} 1 + u_{0,0} & u_{0,1} \\ u_{1,0} & 1 + u_{1,1} \end{smallmatrix} \right)`.
   Then we compute

   .. math::
      :label: log1p

      \log J = \mathtt{log1p}(u_{0,0} + u_{1,1} + u_{0,0} u_{1,1} - u_{0,1} u_{1,0}),

   which gives accurate results even in the limit when the entries :math:`u_{i,j}` are very small.
   For example, if :math:`u_{i,j} \sim 10^{-8}`, then naive computation of :math:`\bm I_3 - \bm C^{-1}` and :math:`\log J` will have a relative accuracy of order :math:`10^{-8}` in double precision and no correct digits in single precision.
   When using the stable choices above, these quantities retain full :math:`\varepsilon_{\text{machine}}` relative accuracy.

.. admonition:: Mooney-Rivlin model
   :class: dropdown

   Constitutive models of rubber and other nearly-incompressible materials are often expressed in terms of invariants of an isochoric Cauchy-Green tensor :math:`\bar{\bm C} = J^{-2/3} \bm C`, typically defined as

   .. math::
      \begin{aligned}
      \mathbb{\bar I_1} &= \operatorname{trace} \bm{\bar C} \\
                        &= J^{-2/3} \operatorname{trace} \bm C \\
                        &= J^{-2/3} \mathbb I_1 \\
      \mathbb{\bar I_2} &= \frac 1 2 \Big( (\operatorname{trace} \bm{\bar C})^2 - \bm{\bar C} \!:\! \bm{\bar C} \Big) \\
                        &= \frac 1 2 \Big( \mathbb{\bar I_1^2} - J^{-4/3} \bm C \!:\! \bm C \Big) \\
                        &= J^{-4/3} \mathbb I_2
      \end{aligned}

   along with the determinant :math:`J = \sqrt{\lvert \bm C \rvert}`.

   :math:`\mathbb{\bar I_1}` and :math:`\mathbb{\bar I_2}` are derived as follows:

   .. math::

      \begin{aligned}
      \frac{\partial \mathbb{I_1}}{\partial \bm E} &= 2 \bm I_3 &
      \frac{\partial \mathbb{\bar I_1}}{\partial \bm E} &= 2 J^{-2/3} \big(\bm I_3 - \frac 1 3 \mathbb I_1 \bm C^{-1} \big) \\
      \frac{\partial \mathbb{I_2}}{\partial \bm E} &= 2 \mathbb I_1 \bm I_3 - 2 \bm C &
      \frac{\partial \mathbb{\bar I_2}}{\partial \bm E} &= 2 J^{-4/3} \big(\mathbb I_1 \bm I_3 - \bm C - \frac 2 3 \mathbb I_2 C^{-1} \big) .
      \end{aligned}

   The Mooney-Rivlin strain energy density (cf. Neo-Hookean :math:numref:`neo-hookean-energy`) is :cite:`bower2010applied`

   .. math::
      :label: mooney-rivlin-energy

      \Phi(\mathbb{\bar I_1}, \mathbb{\bar I_2}, J) = \frac{\mu_1}{2} (\mathbb{\bar I_1} - 3) + \frac{\mu_2}{2} (\mathbb{\bar I_2} - 3) + \frac{k_1}{2} (J - 1)^2,

   which we differentiate as in the Neo-Hookean case :math:numref:`neo-hookean-stress` to yield the second Piola-Kirchoff tensor,

   .. math::
      :label: mooney-rivlin-stress

      \bm S = \mu _1 J^{-2/3} \big(\bm I_3 - \frac 1 3 \mathbb I_1 \bm C^{-1} \big) + \mu _2 J^{-4/3} \big(\mathbb I_1 \bm I_3 - \bm C - \frac 2 3 \mathbb I_2 \bm C^{-1} \big) + k_1(J^2 -J)\bm C^{-1} ,

   The derivative of :math:numref:`mooney-rivlin-stress`, :math:`dS`, is
   .. math::
      :label: mooney-rivlin-dS
      \diff\bm S = \frac{\partial \bm S}{\partial \bm E} \!:\! \diff \bm E = -\frac{2}{3}\mu_1 J^{-2/3} \bm C^{-1} (2 \bm I_3 - \diff \bm E \bm C^{-1} \mathbb I_1 - \frac 1 3 C^{-1} \mathbb I_1)\\
        + \mu_2 J^{-4/3}\left( 2(\bm I_3 - \frac 4 3 - \mathbb I_2\bm C^{-1}\diff \bm E \bm C^{-1}) - \frac 4 3 \bm C^{-1}(\mathbb I_1 \bm I_3 - \bm C - \frac 2 3 \mathbb I_2 \bm C^{-1})\right)\\
        + k_1 \left( (J^2 -J)(-2\bm C^{-1}\diff \bm E \bm C^{-1}) + J\bm C^{-2}(2J -1)\right)
   which can be used in the Newton linearization. 


.. admonition:: Generalized Polynomial model
   :class: dropdown

   The Generalized Polynomial strain energy density (cf. Neo-Hookean :math:numref:`neo-hookean-energy`) is :cite:`bower2010applied`

   .. math::
       :label: generalized-polynomial-energy

       \Phi(\mathbb{\bar I_1}, \mathbb{\bar I_2}, J) = \sum_{i + j = 1}^N C_{ij}(\mathbb{\bar I}_1 -3)^i(\mathbb{\bar I}_2 -3)^j + \sum_{i = 1}^N \frac{k_i}{2}(J -1)^{2i}

   which we differentiate, using the derivaties defined in Mooney-Rivlin, as in the Neo-Hookean case :math:numref:`neo-hookean-stress` to yield the second Piola-Kirchoff tensor,

   .. math::
      :label: generalized-polynomial-stress

      \bm S = \sum_{i + j = 1}^N 2C_{ij}\left( j(\mathbb{\bar I}_1 -3)^i(\mathbb{\bar I}_2 -3)^{j-1}J^{-4/3} \big(\mathbb I_1 \bm I_3 - \bm C - \frac 2 3 \mathbb I_2 \bm C^{-1} \big) + i(\mathbb{\bar I}_2 -3)^j(\mathbb{\bar I}_1 -3)^{i-1} J^{-2/3} \big(\bm I_3 - \frac 1 3 \mathbb I_1 \bm C^{-1} \big) \right) + \sum_{i = 1}^N k_i i(J -1)^{2i-1}J\bm C^{-1},

.. note::
   One can linearize :math:numref:`neo-hookean-stress` around :math:`\bm E = 0`, for which :math:`\bm C = \bm I_3 + 2 \bm E \to \bm I_3` and :math:`J \to 1 + \operatorname{trace} \bm E`, therefore :math:numref:`neo-hookean-stress` reduces to
 
   .. math::
      :label: eq-st-venant-kirchoff

      \bm S = \lambda (\operatorname{trace} \bm E) \bm I_3 + 2 \mu \bm E,
 
   which is the St. Venant-Kirchoff model (constitutive linearization without geometric linearization; see :math:numref:`hyperelastic-cd`).

   This model can be used for geometrically nonlinear mechanics (e.g., snap-through of thin structures), but is inappropriate for large strain.

   Alternatively, one can drop geometric nonlinearities, :math:`\bm E \to \bm \epsilon` and :math:`\bm C \to \bm I_3`, while retaining the nonlinear dependence on :math:`J \to 1 + \operatorname{trace} \bm \epsilon`, thereby yielding :math:numref:`eq-neo-hookean-small-strain` (see :math:numref:`hyperelastic-cd`).


Weak form
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We multiply :math:numref:`sblFinS` by a test function :math:`\bm v` and integrate by parts to obtain the weak form for finite-strain hyperelasticity:
find :math:`\bm u \in \mathcal V \subset H^1(\Omega_0)` such that

.. math::
   :label: hyperelastic-weak-form-initial

    \int_{\Omega_0}{\nabla_X \bm{v} \tcolon \bm{P}} \, dV
    - \int_{\Omega_0}{\bm{v} \cdot \rho_0 \bm{g}} \, dV
    - \int_{\partial \Omega_0}{\bm{v} \cdot (\bm{P} \cdot \hat{\bm{N}})} \, dS
    = 0, \quad \forall \bm v \in \mathcal V,
    
where :math:`\bm{P} \cdot \hat{\bm{N}}|_{\partial\Omega}` is replaced by any prescribed force/traction boundary condition written in terms of the initial configuration.
This equation contains material/constitutive nonlinearities in defining :math:`\bm S(\bm E)`, as well as geometric nonlinearities through :math:`\bm P = \bm F\, \bm S`, :math:`\bm E(\bm F)`, and the body force :math:`\bm g`, which must be pulled back from the current configuration to the initial configuration.
Discretization of :math:numref:`hyperelastic-weak-form-initial` produces a finite-dimensional system of nonlinear algebraic equations, which we solve using Newton-Raphson methods.
One attractive feature of Galerkin discretization is that we can arrive at the same linear system by discretizing the Newton linearization of the continuous form; that is, discretization and differentiation (Newton linearization) commute.


Newton linearization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To derive a Newton linearization of :math:numref:`hyperelastic-weak-form-initial`, we begin by expressing the derivative of :math:numref:`1st2nd` in incremental form,

.. math::
   :label: eq-diff-P

   \diff \bm P = \frac{\partial \bm P}{\partial \bm F} \!:\! \diff \bm F = \diff \bm F\, \bm S + \bm F \underbrace{\frac{\partial \bm S}{\partial \bm E} \!:\! \diff \bm E}_{\diff \bm S}

where

.. math::
   \diff \bm E = \frac{\partial \bm E}{\partial \bm F} \!:\! \diff \bm F = \frac 1 2 \Big( \diff \bm F^T \bm F + \bm F^T \diff \bm F \Big)

and :math:`\diff\bm F = \nabla_X\diff\bm u`.
The quantity :math:`{\partial \bm S} / {\partial \bm E}` is known as the incremental elasticity tensor, and is analogous to the linear elasticity tensor :math:`\mathsf C` of :math:numref:`linear-elasticity-tensor`.
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

.. dropdown:: Cancellation vs symmetry

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

   We prefer to compute with :math:numref:`eq-neo-hookean-incremental-stress` because :math:numref:`eq-diff-P-dF` is more expensive, requiring access to (non-symmetric) :math:`\bm F^{-1}` in addition to (symmetric) :math:`\bm C^{-1} = \bm F^{-1} \bm F^{-T}`, having fewer symmetries to exploit in contractions, and being less numerically stable.

.. dropdown:: :math:`\diff\bm S` in index notation

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

Note that the Newton linearization of :math:numref:`hyperelastic-weak-form-initial` may be written as a weak form for linear operators: find :math:`\diff\bm u \in \mathcal V_0` such that

.. math::

   \int_{\Omega_0} \nabla_X \bm v \!:\! \diff\bm P dV = \text{rhs}, \quad \forall \bm v \in \mathcal V_0,

where :math:`\diff \bm P` is defined by :math:numref:`eq-diff-P` and :math:numref:`eq-neo-hookean-incremental-stress`, and :math:`\mathcal V_0` is the homogeneous space corresponding to :math:`\mathcal V`.

.. note::
   The decision of whether to recompute or store functions of the current state :math:`\bm F` depends on a roofline analysis :cite:`williams2009roofline,brown2010` of the computation and the cost of the constitutive model.
   For low-order elements where flops tend to be in surplus relative to memory bandwidth, recomputation is likely to be preferable, where as the opposite may be true for high-order elements.
   Similarly, analysis with a simple constitutive model may see better performance while storing little or nothing while an expensive model such as Arruda-Boyce :cite:`arruda1993largestretch`, which contains many special functions, may be faster when using more storage to avoid recomputation.
   In the case where complete linearization is preferred, note the symmetry :math:`\mathsf C_{IJKL} = \mathsf C_{KLIJ}` evident in :math:numref:`eq-neo-hookean-incremental-stress-index`, thus :math:`\mathsf C` can be stored as a symmetric :math:`6\times 6` matrix, which has 21 unique entries.
   Along with 6 entries for :math:`\bm S`, this totals 27 entries of overhead compared to computing everything from :math:`\bm F`.
   This compares with 13 entries of overhead for direct storage of :math:`\{ \bm S, \bm C^{-1}, \log J \}`, which is sufficient for the Neo-Hookean model to avoid all but matrix products.


.. _problem-hyperelasticity-finite-strain-current-configuration:

Hyperelasticity in current configuration
--------------------------------------------

In the preceeding discussion, all equations have been formulated in the initial configuration.
This may feel convenient in that the computational domain is clearly independent of the solution, but there are some advantages to defining the equations in the current configuration.

#. Body forces (like gravity), traction, and contact are more easily defined in the current configuration.
#. Mesh quality in the initial configuration can be very bad for large deformation.
#. The required storage and numerical representation can be smaller in the current configuration.

Most of the benefit in case 3 can be attained solely by moving the Jacobian representation to the current configuration :cite:`davydov2020matrix`, though residual evaluation may also be slightly faster in current configuration.
There are multiple commuting paths from the nonlinear weak form in initial configuration :math:numref:`hyperelastic-weak-form-initial` to the Jacobian weak form in current configuration :math:numref:`jacobian-weak-form-current`.
One may push forward to the current configuration and then linearize or linearize in initial configuration and then push forward, as summarized below.

.. math::
   :label: initial-current-linearize

   \begin{CD}
     {\overbrace{\nabla_X \bm{v} \tcolon \bm{FS}}^{\text{Initial Residual}}}
     @>{\text{push forward}}>{}>
     {\overbrace{\nabla_x \bm{v} \tcolon \bm{\tau}}^{\text{Current Residual}}} \\
     @V{\text{linearize}}V{\begin{smallmatrix} \diff\bm F = \nabla_X\diff\bm u \\ \diff\bm S(\diff\bm E) \end{smallmatrix}}V
     @V{\begin{smallmatrix} \diff\nabla_x\bm v = -\nabla_x\bm v \nabla_x \diff\bm u \\ \diff\bm\tau(\diff\bm\epsilon) \end{smallmatrix}}V{\text{linearize}}V \\
     {\underbrace{\nabla_X\bm{v}\tcolon \Big(\diff\bm{F}\bm{S} + \bm{F}\diff\bm{S}\Big)}_\text{Initial Jacobian}}
     @>{\text{push forward}}>{}>
     {\underbrace{\nabla_x\bm{v}\tcolon \Big(\diff\bm{\tau} -\bm{\tau}(\nabla_x \diff\bm{u})^T \Big)}_\text{Current Jacobian}}
   \end{CD}

We will follow both paths for consistency and because both intermediate representations may be useful for implementation.


Push forward, then linearize
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first term of :math:numref:`hyperelastic-weak-form-initial` can be rewritten in terms of the symmetric Kirchhoff stress tensor
:math:`\bm{\tau}=J\bm{\sigma}=\bm{P}\bm{F}^T = \bm F \bm S \bm F^T` as

.. math::
   \nabla_X \bm{v} \tcolon \bm{P} = \nabla_X \bm{v} \tcolon \bm{\tau}\bm{F}^{-T} = \nabla_X \bm{v}\bm{F}^{-1} \tcolon \bm{\tau} = \nabla_x \bm{v} \tcolon \bm{\tau}

therefore, the weak form in terms of :math:`\bm{\tau}` and :math:`\nabla_x` with integral over :math:`\Omega_0` is

.. math::
   :label: hyperelastic-weak-form-current

    \int_{\Omega_0}{\nabla_x \bm{v} \tcolon \bm{\tau}} \, dV
    - \int_{\Omega_0}{\bm{v} \cdot \rho_0 \bm{g}} \, dV
    - \int_{\partial \Omega_0}{\bm{v}\cdot(\bm{P}\cdot\hat{\bm{N}})} \, dS
    = 0, \quad \forall \bm v \in \mathcal V.
    

Linearize in current configuration
""""""""""""""""""""""""""""""""""

To derive a Newton linearization of :math:numref:`hyperelastic-weak-form-current`, first we define

.. math::
   :label: nabla_xdu

   \nabla_x \diff \bm{u} = \nabla_X \diff \bm{u} \  \bm{F}^{-1} = \diff \bm{F} \bm{F}^{-1}

and :math:`\bm{\tau}` for Neo-Hookean materials as the push forward of :math:numref:`neo-hookean-stress`

.. math::
   :label: tau-neo-hookean

    \bm{\tau} = \bm{F}\bm{S}\bm{F}^T = \mu (\bm{b} - \bm I_3) + \lambda \log J \bm{I}_3,

where :math:`\bm{b} = \bm{F} \bm{F}^T`, is the left Cauchy-Green tensor. 
Then by expanding the directional derivative of :math:`\nabla_x \bm{v} \tcolon \bm{\tau}`, we arrive at

.. math::
   :label: hyperelastic-linearization-current1

    \diff \ (\nabla_x \bm{v} \tcolon \bm{\tau}) = \diff \ (\nabla_x \bm{v})\tcolon \bm{\tau} + \nabla_x \bm{v} \tcolon \diff \bm{\tau} .

The first term of :math:numref:`hyperelastic-linearization-current1` can be written as

.. math::
   \begin{aligned}
   \diff \ (\nabla_x \bm{v})\tcolon \bm{\tau} &= \diff \ (\nabla_X \bm{v} \bm{F}^{-1})\tcolon \bm{\tau} = \Big(\underbrace{\nabla_X (\diff \bm{v})}_{0}\bm{F}^{-1} +  \nabla_X \bm{v}\diff \bm{F}^{-1}\Big)\tcolon \bm{\tau}\\
     &= \Big(-\nabla_X \bm{v} \bm{F}^{-1}\diff\bm{F}\bm{F}^{-1}\Big)\tcolon \bm{\tau}=\Big(-\nabla_x \bm{v} \diff\bm{F}\bm{F}^{-1}\Big)\tcolon \bm{\tau}\\
     &= \Big(-\nabla_x \bm{v} \nabla_x \diff\bm{u} \Big)\tcolon \bm{\tau}= -\nabla_x \bm{v}\tcolon\bm{\tau}(\nabla_x \diff\bm{u})^T \,,
   \end{aligned}

where we have used :math:`\diff \bm{F}^{-1}=-\bm{F}^{-1} \diff \bm{F} \bm{F}^{-1}` and :math:numref:`nabla_xdu`.
Using this and :math:numref:`hyperelastic-linearization-current1` in :math:numref:`hyperelastic-weak-form-current` yields the weak form in the current configuration

.. math::
   :label: jacobian-weak-form-current

   \int_{\Omega_0} \nabla_x \bm v \tcolon \Big(\diff\bm\tau - \bm\tau (\nabla_x \diff\bm u)^T \Big) = \text{rhs}.

In the following, we will sometimes make use of the incremental strain tensor in the current configuration,

.. math::
   \diff\bm\epsilon \equiv \frac{1}{2}\Big(\nabla_x \diff\bm{u} + (\nabla_x \diff\bm{u})^T   \Big) .


.. dropdown:: Deriving :math:`\diff\bm\tau` for Neo-Hookean material

   To derive a useful expression of :math:`\diff\bm\tau` for Neo-Hookean materials, we will use the representations

   .. math::

      \begin{aligned}
      \diff \bm{b} &= \diff \bm{F} \bm{F}^T + \bm{F} \diff \bm{F}^T \\
      &= \nabla_x \diff \bm{u} \ \bm{b} + \bm{b} \ (\nabla_x \diff \bm{u})^T \\
      &= (\nabla_x \diff\bm u)(\bm b - \bm I_3) + (\bm b - \bm I_3) (\nabla_x \diff\bm u)^T + 2 \diff\bm\epsilon
      \end{aligned}

   and

   .. math::
      \begin{aligned}
      \diff\ (\log J) &= \frac{\partial \log J}{\partial \bm{b}}\tcolon \diff \bm{b} = \frac{\partial J}{J\partial \bm{b}}\tcolon \diff \bm{b}=\frac{1}{2}\bm{b}^{-1}\tcolon \diff \bm{b} \\
      &= \frac 1 2 \bm b^{-1} \tcolon \Big(\nabla_x \diff\bm u \ \bm b + \bm b (\nabla_x \diff\bm u)^T \Big) \\
      &= \trace (\nabla_x \diff\bm u) \\
      &= \trace \diff\bm\epsilon .
      \end{aligned}

   Substituting into :math:numref:`tau-neo-hookean` gives

   .. math::
      :label: dtau-neo-hookean

      \begin{aligned}
      \diff \bm{\tau} &= \mu \diff \bm{b} + \lambda \trace (\diff\bm\epsilon) \bm I_3 \\
      &= \underbrace{2 \mu \diff\bm\epsilon + \lambda \trace (\diff\bm\epsilon) \bm I_3 - 2\lambda \log J \diff\bm\epsilon}_{\bm F \diff\bm S \bm F^T} \\
      &\quad + (\nabla_x \diff\bm u)\underbrace{\Big( \mu (\bm b - \bm I_3) + \lambda \log J \bm I_3 \Big)}_{\bm\tau} \\
      &\quad + \underbrace{\Big( \mu (\bm b - \bm I_3) + \lambda \log J \bm I_3 \Big)}_{\bm\tau}  (\nabla_x \diff\bm u)^T ,
      \end{aligned}

   where the final expression has been identified according to

   .. math::
      \diff\bm\tau = \diff\ (\bm F \bm S \bm F^T) = (\nabla_x \diff\bm u) \bm\tau + \bm F \diff\bm S \bm F^T + \bm\tau(\nabla_x \diff\bm u)^T.

Collecting terms, we may thus opt to use either of the two forms

.. math::
   :label: cur_simp_Jac

   \begin{aligned}
   \diff \bm{\tau} -\bm{\tau}(\nabla_x \diff\bm{u})^T &= (\nabla_x \diff\bm u)\bm\tau + \bm F \diff\bm S \bm F^T \\
   &= (\nabla_x \diff\bm u)\bm\tau + \lambda \trace(\diff\bm\epsilon) \bm I_3 + 2(\mu - \lambda \log J) \diff\bm\epsilon,
   \end{aligned}

with the last line showing the especially compact representation available for Neo-Hookean materials.


Linearize, then push forward
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can move the derivatives to the current configuration via

.. math::

   \nabla_X \bm v \!:\! \diff\bm P = (\nabla_X \bm v) \bm F^{-1} \!:\! \diff \bm P \bm F^T = \nabla_x \bm v \!:\! \diff\bm P \bm F^T

and expand

.. math::

   \begin{aligned}
   \diff\bm P \bm F^T &= \diff\bm F \bm S \bm F^T + \bm F \diff\bm S \bm F^T \\
   &= \underbrace{\diff\bm F \bm F^{-1}}_{\nabla_x \diff\bm u} \underbrace{\bm F \bm S \bm F^T}_{\bm\tau} + \bm F \diff\bm S \bm F^T .
   \end{aligned}

.. dropdown:: Representation of :math:`\bm F \diff\bm S \bm F^T` for Neo-Hookean materials

   Now we push :math:numref:`eq-neo-hookean-incremental-stress` forward via

   .. math::

      \begin{aligned}
      \bm F \diff\bm S \bm F^T &= \lambda (\bm C^{-1} \!:\! \diff\bm E) \bm F \bm C^{-1} \bm F^T
        + 2 (\mu - \lambda \log J) \bm F \bm C^{-1} \diff\bm E \, \bm C^{-1} \bm F^T \\
          &= \lambda (\bm C^{-1} \!:\! \diff\bm E) \bm I_3 + 2 (\mu - \lambda \log J) \bm F^{-T} \diff\bm E \, \bm F^{-1} \\
          &= \lambda \operatorname{trace}(\nabla_x \diff\bm u) \bm I_3 + 2 (\mu - \lambda \log J) \diff\bm \epsilon
      \end{aligned}

   where we have used

   .. math::

      \begin{aligned}
      \bm C^{-1} \!:\! \diff\bm E &= \bm F^{-1} \bm F^{-T} \!:\! \bm F^T \diff\bm F \\
      &= \operatorname{trace}(\bm F^{-1} \bm F^{-T} \bm F^T \diff \bm F) \\
      &= \operatorname{trace}(\bm F^{-1} \diff\bm F) \\
      &= \operatorname{trace}(\diff \bm F \bm F^{-1}) \\
      &= \operatorname{trace}(\nabla_x \diff\bm u)
      \end{aligned}

   and

   .. math::

      \begin{aligned}
      \bm F^{-T} \diff\bm E \, \bm F^{-1} &= \frac 1 2 \bm F^{-T} (\bm F^T \diff\bm F + \diff\bm F^T \bm F) \bm F^{-1} \\
      &= \frac 1 2 (\diff \bm F \bm F^{-1} + \bm F^{-T} \diff\bm F^T) \\
      &= \frac 1 2 \Big(\nabla_x \diff\bm u + (\nabla_x\diff\bm u)^T \Big) \equiv \diff\bm\epsilon.
      \end{aligned}

Collecting terms, the weak form of the Newton linearization for Neo-Hookean materials in the current configuration is

.. math::
   :label: jacobian-weak-form-current2

   \int_{\Omega_0} \nabla_x \bm v \!:\! \Big( (\nabla_x \diff\bm u) \bm\tau + \lambda \operatorname{trace}(\diff\bm\epsilon)\bm I_3 + 2(\mu - \lambda\log J)\diff \bm\epsilon \Big) dV = \text{rhs},

which equivalent to Algorithm 2 of :cite:`davydov2020matrix` and requires only derivatives with respect to the current configuration. Note that :math:numref:`cur_simp_Jac` and :math:numref:`jacobian-weak-form-current2` have recovered the same representation 
using different algebraic manipulations. 

.. tip::
   We define a second order *Green-Euler* strain tensor (cf. Green-Lagrange strain :math:numref:`eq-green-lagrange-strain`) as

   .. math::
      :label: green-euler-strain

      \bm e = \frac 1 2 \Big(\bm{b} - \bm{I}_3 \Big) = \frac 1 2 \Big( \nabla_X \bm{u} + (\nabla_X \bm{u})^T + \nabla_X \bm{u} \, (\nabla_X \bm{u})^T \Big).

   Then, the Kirchhoff stress tensor :math:numref:`tau-neo-hookean` can be written as

   .. math::
      :label: tau-neo-hookean-stable
      
      \bm \tau = \lambda \log J \bm I_{3} + 2\mu \bm e,

   which is more numerically stable for small strain, and thus preferred for computation. Note that the :math:`\log J` is computed via ``log1p`` :math:numref:`log1p`, as we discussed in the previous tip.


Jacobian representation
^^^^^^^^^^^^^^^^^^^^^^^

We have implemented four storage variants for the Jacobian in our finite strain hyperelasticity. In each case, some variables are computed during residual evaluation and used during Jacobian application.


.. list-table:: Four algorithms for Jacobian action in finite strain hyperelasticity problem
   :header-rows: 1

   * - Option :code:`-problem`
     - Static storage
     - Computed storage
     - # scalars
     - Equations


   * - :code:`FSInitial-NH1`
     - :math:`\nabla_{X} \hat X, \operatorname{det}\nabla_{\hat X} X`
     - :math:`\nabla_X \bm u`
     - 19
     - :math:numref:`eq-diff-P` :math:numref:`eq-neo-hookean-incremental-stress`

   * - :code:`FSInitial-NH2`
     - :math:`\nabla_{X} \hat X, \operatorname{det}\nabla_{\hat X} X`
     - :math:`\nabla_X \bm u, \bm C^{-1}, \lambda \log J`
     - 26
     - :math:numref:`eq-diff-P` :math:numref:`eq-neo-hookean-incremental-stress`

   * - :code:`FSCurrent-NH1`
     - :math:`\nabla_{X} \hat X, \operatorname{det}\nabla_{\hat X} X`
     - :math:`\nabla_X \bm u`
     - 19
     - :math:numref:`jacobian-weak-form-current` :math:numref:`nabla_xdu`

   * - :code:`FSCurrent-NH2`
     - :math:`\operatorname{det}\nabla_{\hat X} X`
     - :math:`\nabla_x \hat X, \bm \tau, \lambda \log J`
     - 17
     - :math:numref:`jacobian-weak-form-current` :math:numref:`jacobian-weak-form-current2`
