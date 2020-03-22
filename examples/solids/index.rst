.. _example-petsc-elasticity:

Solid mechanics elasticity mini-app
========================================

This example is located in the subdirectory :file:`examples/solids`. It solves
the steady-state static balance momentum equations using unstructured
high-order finite/spectral element spatial discretizations. As for the
:ref:`example-petsc-navier-stokes` case, the solid mechanics elasticity example
has been developed using PETSc, so that the pointwise physics (defined at
quadrature points) is separated from the parallelization and meshing concerns.


.. _problem-linear-elasticity:

----------------------------------------

The strong form of the static balance of linear momentum at small strain for the three-dimensional linear elasticity problem is given by :cite:`hughes2012finite`:

.. math::
   :label: lin-elas
   
   \nabla \cdot \boldsymbol{\sigma} + \boldsymbol{g} = \boldsymbol{0} 


where :math:`\boldsymbol{\sigma}` and :math:`\boldsymbol{g}` are stress and forcing functions,
respectively. Integrating by parts on the divergence term, we arrive at the weak form the of equation :math:numref:`lin-elas`:

.. math::

   \int_{\Omega}{ \nabla \boldsymbol{v} \colon \boldsymbol{\sigma}} dV - \int_{d\Omega}{\boldsymbol{v} \cdot \left(\boldsymbol{\sigma}_t \cdot \hat{\boldsymbol{n}}\right)} dS + \int_{\Omega}{\boldsymbol{v} \cdot \boldsymbol{g}} dV = 0

where :math:`\boldsymbol{\sigma}_t \cdot \hat{\boldsymbol{n}}` is typically
replaced with a boundary condition.

The constitutive law (stress-strain relationship) is given by:

.. math::
   :label: linear-stress-strain

   \boldsymbol{\sigma} = \mathsf{C} \boldsymbol{\epsilon},

where 

.. math::
   :label: small-strain

   \boldsymbol{\epsilon} = \dfrac{1}{2}\left(\nabla \boldsymbol{u} + \nabla \boldsymbol{u}^T \right)

is the symmetric (small/infinitesimal) strain tensor.
For notational convenience, we express the symmetric second order tensors :math:`\bm \sigma` and :math:`\bm \epsilon` as vectors of length 6 using the `Voigt notation <https://en.wikipedia.org/wiki/Voigt_notation>`_. Hence, the fourth order material stiffness tensor :math:`\mathsf C` can be represented as a :math:`6\times 6` symmetric matrix

.. math::

   \mathsf C = \dfrac{E}{(1+\nu)(1-2\nu)}
   \begin{bmatrix}
        1-\nu & \nu & \nu & & & \\
          \nu & 1 - \nu & \nu & & & \\
          \nu & \nu &  1 - \nu & & & \\
          & & & \dfrac{1 - 2\nu}{2} & & \\    
         & & & &\dfrac{1 - 2\nu}{2} & \\
         & & & & & \dfrac{1 - 2\nu}{2} \\   
   \end{bmatrix},

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
With the latter formulation, the stiffness tensor becomes

.. math::

   \mathsf C = \begin{bmatrix}
   \lambda + 2\mu & \lambda & \lambda & & & \\
   \lambda & \lambda + 2\mu & \lambda & & & \\
   \lambda & \lambda & \lambda + 2\mu & & & \\
   & & & \mu & & \\
   & & & & \mu & \\
   & & & & & \mu
   \end{bmatrix}.

Note that the incompressible limit :math:`\nu \to \frac 1 2` causes :math:`\lambda \to \infty`, and thus :math:`\mathsf C` becomes singular.


.. _problem-hyper-small-strain:

Hyperelasticity at Small Strain
----------------------------------------

The small strain version of a Neo-Hookean hyperelasticity material is given as
follows:

.. math::
   :label: clss
   
   \boldsymbol{\sigma} = \lambda \ln(1 + \boldsymbol{\epsilon_v)} \boldsymbol{I}_3 + 2\mu \boldsymbol{\epsilon}

where :math:`\boldsymbol{\epsilon}` is defined as in :math:numref:`small-strain`. The trace of the strain tensor, also known as the *volumetric strain* and denoted by :math:`\boldsymbol{\epsilon}_v`, in three dimensions is :math:`\boldsymbol{\epsilon}_v = \boldsymbol{\epsilon}_{11} + \boldsymbol{\epsilon}_{22} + \boldsymbol{\epsilon}_{33}`. To easily represent spatial derivatives, we rewrite equation :math:numref:`clss` in indicial notation:

.. math::
   \sigma_{ij} = \lambda ln(1 + \epsilon_v)\delta_{ij} + 2\mu\epsilon_{ij}

so that its derivative is:

.. math::
   :label: derss

   \dfrac{\partial{\sigma_{ij}}}{\partial{\epsilon_{kl}}} = \bar{\lambda}\delta_{ij}\delta_{kl} + 2\mu \delta_{ik} \delta_{jl} ,

where we have introduced the symbol

.. math::

   \bar{\lambda} = \dfrac{\lambda}{1+\epsilon_v} .

Consequently, equation :math:numref:`derss` can be written in matrix form as follows:

.. math::
   :label: mdss

   \left[
     \begin{array}{c} 
       d\sigma_{11} \\
       d\sigma_{22} \\
       d\sigma_{33} \\
       d\sigma_{12} \\
       d\sigma_{13} \\
       d\sigma_{23}       
    \end {array}
   \right]  = 
   \left[
     \begin{array}{cccccc} 
       2\mu +\bar{\lambda} & \bar{\lambda} & \bar{\lambda} & & & \\
        \bar{\lambda} & 2\mu +\bar{\lambda} & \bar{\lambda} & & & \\
        \bar{\lambda} & \bar{\lambda} & 2\mu +\bar{\lambda} & & & \\
        & & & \mu & & \\    
        & & & &\mu & \\
        & & & & & \mu \\   
     \end {array}
   \right] 
   \left[
     \begin{array}{c} 
       d\epsilon_{11} \\
       d\epsilon_{22} \\
       d\epsilon_{33} \\
       d\epsilon_{12} \\
       d\epsilon_{13} \\
       d\epsilon_{23}       
     \end {array}
   \right]
   

.. _problem-hyperelasticity-finite-strain:

Hyperelasticity at Finite Strain
----------------------------------------

In the *total Lagrangian* approach for the neo-Hookean Hyperelasticity
probelm, the discrete equations are formulated with respect to the reference
configuration. In this formulation, we solve for displacement :math:`\bm u(\bm X)` in the reference frame :math:`\bm X`.
The notation for elasticity at finite strain is inspired by :cite:`holzapfel2000nonlinear` to
distinguish between the current and reference configurations. As explained in the
:ref:`Common notation` section, we denote by capital letters the reference frame and by small
letters the current one.

The strong form of the static balance of linear-momentum at
*Finite Strain* (total Lagrangian) is given by:

.. math::
   :label: sblFinS

   \nabla_X \cdot \boldsymbol{P} + \rho_0 \boldsymbol{g} = \boldsymbol{0}
 
where the :math:`_X` in :math:`\nabla_X` indicates that the gradient is calculated with
respect to the reference configuration in the finite strain regime.
:math:`\boldsymbol{P}` and :math:`\boldsymbol{g}` are
the *first Piola-Kirchhoff stress* tensor and the prescribed forcing,
function, respectively. :math:`\rho_0` is known as the *reference* mass
density.
The tensor :math:`\bm P` is not symmetric, living in the current configuration on the left and the reference configuration on the right.
:math:`\boldsymbol{P}` can be decomposed as

.. math::
   :label: 1st2nd
   
   \boldsymbol{P} = \boldsymbol{F} \cdot \boldsymbol{S},

where :math:`\bm S` is the *second Piola-Kirchhoff stress* tensor, a symmetric tensor
defined entirely in the reference configuration,
and :math:`\boldsymbol{F} = \bm I_3 + \nabla_X \bm u` is the deformation gradient.
Different constitutive models can define :math:`\bm S`.


Constitutive modeling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In their most general form, constitutive models define :math:`\bm S` in terms of state
variables. In the model taken into consideration in the present miniapp, the state variables are constituted by the vector displacement field
:math:`\bm u`, and its gradient :math:`\nabla_X \bm u`.
We begin by defining two symmetric tensors in the reference configuration, the right Cauchy-Green tensor

.. math::
   \bm C = \bm F^T \bm F

and the Green-Lagrange strain tensor

.. math::
   :label: eq-non-linear-strain

   \bm E = \frac 1 2 (\bm C - \bm I_3) = \frac 1 2 \Big( \nabla_X \bm u + (\nabla_X \bm u)^T + (\nabla_X \bm u)^T \nabla_X \bm u \Big),

the latter of which converges to the linear strain tensor :math:`\bm \epsilon` in the small-deformation limit.
The constitutive models considered, appropriate for large deformations, express :math:`\bm S` as a function of :math:`\bm E`, similar to the linear case, showed in equation  :math:numref:`linear-stress-strain`, which  expresses the relationship between :math:`\bm\sigma` and :math:`\bm\epsilon`.

Equation :math:numref:`eq-non-linear-strain` represents a nonlinear tensor-valued function of a tensor-valued input, but an arbitrary choice of such a function will generally not be invariant under orthogonal transformations, thus will not admissible because a physical model must not depend on the choice of coordinate system chosen to express it.
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
   The strain energy density functional cannot be an arbitrary function :math:`\Phi(\bm E)`, but can only depend on *invariants*, scalar-valued functions :math:`\gamma` satisfying

   .. math::
      \gamma(\bm E) = \gamma(Q \bm E Q^T)

   for all orthogonal matrices :math:`Q`.
   Consequently, we may assume without loss of generality that :math:`\bm E` is diagonal, and take its set of eigenvalues as the invariants.
   It is clear that there can be only three invariants, and there are many alternate choices, such as
   :math:`\operatorname{trace}(\bm E), \operatorname{trace}(\bm E^2), \lvert E \rvert` and combinations thereof.
   It is common in the literature for invariants to be taken from :math:`\bm C = \bm I_3 + 2 \bm E` instead of :math:`\bm E`.

For example, if we take the compressible Neo-Hookean model,

.. math::
   :label: neo-hookean-energy

   \begin{aligned}
   \Phi(\bm E) &= \frac{\lambda}{2}(\log J)^2 + \frac \mu 2 (\operatorname{trace} \bm C - 3) - \mu \log J \\
     &= \frac{\lambda}{2}(\log J)^2 + \mu \operatorname{trace} \bm E - \mu \log J,
   \end{aligned}

where :math:`J = \lvert \bm F \rvert = \sqrt{\lvert \bm C \rvert}` is the determinant of deformation (i.e., volume change)
and :math:`\lambda` and :math:`\mu` are the Lamé parameters in the infinitesimal strain limit.

To evaluate :math:numref:`strain-energy-grad`, we make use of

.. math::
   \frac{\partial J}{\partial \bm E} = \frac{\partial \sqrt{\lvert \bm C \rvert}}{\partial \bm E} = \lvert \bm C \rvert^{-1/2} \lvert \bm C \rvert \bm C^{-1} = J \bm C^{-1},

where the factor of 2 has been absorbed due to :math:`\bm C = \bm I_3 + 2 \bm E`.
Carrying through the differentiation :math:numref:`strain-energy-grad` for the model :math:numref:`neo-hookean-energy`, we arrive at

.. math::
   :label: neo-hookean-stress

   \bm S = \lambda \log J \bm C^{-1} + \mu (\bm I_3 - \bm C^{-1}).

.. note::
   One can linearize :math:numref:`neo-hookean-stress` around :math:`\bm E = 0` and make use of

   .. math::
      \bm C^{-1} = (\bm I_3 + 2 \bm E)^{-1} = \bm I_3 - 2 \bm E + O\left(\lVert \bm E\rVert^2 \right),

   in which case :math:numref:`neo-hookean-stress` reduces to

   .. math::
      \bm S = \lambda (\operatorname{trace} \bm E) \bm I_3 + 2 \mu \bm E,

   which is the St. Venant-Kirchoff model.
   This model can be used for geometrically nonlinear mechanics (e.g., snap-through of thin structures), but is inappropriate for large strain.

Weak form
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is crucial to distinguish between the current and reference element in the Total Lagrangian Finite Strain regime.

.. math::

    \int_{\Omega}{\boldsymbol{v} \cdot \left(\nabla_X \cdot \boldsymbol{P} + \rho_0 \boldsymbol{g}\right)} dV = \boldsymbol{0}

Integrating by parts, we arrive at the weak form:

.. math::

    \int_{\Omega}{\nabla_X \boldsymbol{v} \colon \boldsymbol{P}}dV =
  - \int_{\Omega}{\boldsymbol{v} \cdot \rho_0 \boldsymbol{g}}dV
  - \int_{\partial \Omega}{\boldsymbol{v} \cdot \boldsymbol{P} \cdot \hat{\boldsymbol{N}}}dA
    
where :math:`\boldsymbol{P} \cdot \hat{\boldsymbol{N}}` is a prescribed boundary
condition written in terms of the reference configuration.

Equation :math:numref:`1st2nd` represents a hyperelastic solid with a Neo-Hookean constitutive law. This is a non-linear isotropic material model that produces a non-linear systems of algebraic equations. Therefore, linearization of the constitutive law is empolyed in the Newton-Raphson step. Evaluating the derivative of the material model yields a forth order tensor:

.. math::
   
  \partial \bm P = \frac{\partial \bm P}{\partial \bm F} \delta \bm F = \delta \bm F \bm S + \bm F \frac{\partial \bm S}{\partial \bm E} \delta \bm E

where

.. math::

  \delta \bm E = \frac{\partial \bm E}{\partial \bm F} \delta \bm F

and

.. math::
   :label: mtfs

   \frac{\partial \bm S}{\partial \bm E} \delta \bm E = \bm S + \left[ \lambda \bm F^{-1} \otimes \bm F^{-1} \left(\lambda \log(J) - \mu \right) \left(\bm F^{-1} \otimes \bm F^{-1} + \bm I_3 \odot \bm C^{-1} \right) \right]
  
Equations :math:numref:`1st2nd` and :math:numref:`mtfs`  may be expressed in indicial notation, respectively, as:

.. math::

   P_{iI} = F_{iB}S_{BI}\, ,

and

.. math::
   :label: mtfsIndicial

   \frac{\partial P_{iI}}{\partial F_{aA}} = \delta_{ai}S_{AI} + \left[\lambda F_{Aa}^{-1} F_{Ii}^{-1} 
   -\left( \lambda \log(J) - \mu\right)\left(F_{Ai}^{-1} F_{Ia}^{-1} + \delta_{ai} C^{-1}_{AI}  \right)   \right] \,.
