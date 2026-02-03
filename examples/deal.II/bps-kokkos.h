// ---------------------------------------------------------------------
//
// Copyright (C) 2023 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
//  Authors: Peter Munch, Martin Kronbichler
//
// ---------------------------------------------------------------------

#pragma once
#ifndef bps_kokkos_h
#  define bps_kokkos_h

// deal.II includes
#  include <deal.II/dofs/dof_tools.h>

#  include <deal.II/fe/mapping.h>

#  include <deal.II/lac/la_parallel_vector.h>

#  include <deal.II/matrix_free/fe_evaluation.h>
#  include <deal.II/matrix_free/matrix_free.h>
#  include <deal.II/matrix_free/shape_info.h>
#  include <deal.II/matrix_free/tools.h>

// local includes
#  include "bps.h"

using namespace dealii;



template <int dim, int fe_degree, int n_q_points_1d, int n_components, typename Number>
class OperatorDealiiMassQuad
{
public:
  DEAL_II_HOST_DEVICE void
  operator()(Portable::FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number> *fe_eval,
             const int q_point) const
  {
    fe_eval->submit_value(fe_eval->get_value(q_point), q_point);
  }
};



template <int dim, int fe_degree, int n_q_points_1d, int n_components, typename Number>
class OperatorDealiiLaplaceQuad
{
public:
  DEAL_II_HOST_DEVICE void
  operator()(Portable::FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number> *fe_eval,
             const int q_point) const
  {
    fe_eval->submit_gradient(fe_eval->get_gradient(q_point), q_point);
  }
};



template <int dim, int fe_degree, int n_q_points_1d, int n_components, typename Number>
class OperatorDealiiMassLocal
{
public:
  DEAL_II_HOST_DEVICE void
  operator()(const typename Portable::MatrixFree<dim, Number>::Data *data,
             const Portable::DeviceVector<Number>                   &src,
             Portable::DeviceVector<Number>                         &dst) const
  {
    Portable::FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number> fe_eval(data);
    fe_eval.read_dof_values(src);
    fe_eval.evaluate(EvaluationFlags::values);
    fe_eval.apply_for_each_quad_point(
      OperatorDealiiMassQuad<dim, fe_degree, n_q_points_1d, n_components, Number>());
    fe_eval.integrate(EvaluationFlags::values);
    fe_eval.distribute_local_to_global(dst);
  }

  static const unsigned int n_local_dofs = Utilities::pow(fe_degree + 1, dim) * n_components;
  static const unsigned int n_q_points   = Utilities::pow(n_q_points_1d, dim);
};



template <int dim, int fe_degree, int n_q_points_1d, int n_components, typename Number>
class OperatorDealiiLaplaceLocal
{
public:
  DEAL_II_HOST_DEVICE void
  operator()(const typename Portable::MatrixFree<dim, Number>::Data *data,
             const Portable::DeviceVector<Number>                   &src,
             Portable::DeviceVector<Number>                         &dst) const
  {
    Portable::FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number> fe_eval(data);
    fe_eval.read_dof_values(src);
    fe_eval.evaluate(EvaluationFlags::gradients);
    fe_eval.apply_for_each_quad_point(
      OperatorDealiiLaplaceQuad<dim, fe_degree, n_q_points_1d, n_components, Number>());
    fe_eval.integrate(EvaluationFlags::gradients);
    fe_eval.distribute_local_to_global(dst);
  }

  static const unsigned int n_local_dofs = Utilities::pow(fe_degree + 1, dim) * n_components;
  static const unsigned int n_q_points   = Utilities::pow(n_q_points_1d, dim);
};



/**
 * Operator GPU implementation using deal.II.
 */
template <int dim, typename Number>
class OperatorDealii : public OperatorBase<Number, MemorySpace::Default>
{
public:
  using VectorType = typename OperatorBase<Number, MemorySpace::Default>::VectorType;

  /**
   * Constructor.
   */
  OperatorDealii(const Mapping<dim>              &mapping,
                 const DoFHandler<dim>           &dof_handler,
                 const AffineConstraints<Number> &constraints,
                 const Quadrature<dim>           &quadrature,
                 const BPType                    &bp)
    : mapping(mapping)
    , dof_handler(dof_handler)
    , constraints(constraints)
    , quadrature(quadrature)
    , bp(bp)
  {
    reinit();
  }

  /**
   * Destructor.
   */
  ~OperatorDealii() = default;

  /**
   * Initialized internal data structures, particularly, MatrixFree.
   */
  void
  reinit() override
  {
    // configure MatrixFree
    typename Portable::MatrixFree<dim, Number>::AdditionalData additional_data;

    if (bp <= BPType::BP2) // mass matrix
      additional_data.mapping_update_flags = update_JxW_values | update_values;
    else
      additional_data.mapping_update_flags = update_JxW_values | update_gradients;

    // create MatrixFree
    AssertThrow(quadrature.is_tensor_product(), ExcNotImplemented());
    matrix_free.reinit(
      mapping, dof_handler, constraints, quadrature.get_tensor_basis()[0], additional_data);
  }

  /**
   * Matrix-vector product.
   */
  void
  vmult(VectorType &dst, const VectorType &src) const override
  {
    dst = 0.0;

    const unsigned int n_components  = dof_handler.get_fe().n_components();
    const unsigned int fe_degree     = dof_handler.get_fe().tensor_degree();
    const unsigned int n_q_points_1d = quadrature.get_tensor_basis()[0].size();

    if (n_components == 1 && fe_degree == 1 && n_q_points_1d == 2)
      this->vmult_internal<1, 1, 2>(dst, src);
    else if (n_components == 1 && fe_degree == 2 && n_q_points_1d == 3)
      this->vmult_internal<1, 2, 3>(dst, src);
    else if (n_components == dim && fe_degree == 1 && n_q_points_1d == 2)
      this->vmult_internal<dim, 1, 2>(dst, src);
    else if (n_components == dim && fe_degree == 2 && n_q_points_1d == 3)
      this->vmult_internal<dim, 2, 3>(dst, src);
    else if (n_components == 1 && fe_degree == 1 && n_q_points_1d == 3)
      this->vmult_internal<1, 1, 3>(dst, src);
    else if (n_components == 1 && fe_degree == 2 && n_q_points_1d == 4)
      this->vmult_internal<1, 2, 4>(dst, src);
    else if (n_components == dim && fe_degree == 1 && n_q_points_1d == 3)
      this->vmult_internal<dim, 1, 3>(dst, src);
    else if (n_components == dim && fe_degree == 2 && n_q_points_1d == 4)
      this->vmult_internal<dim, 2, 4>(dst, src);
    else
      AssertThrow(false, ExcInternalError());

    matrix_free.copy_constrained_values(src, dst);
  }

  /**
   * Initialize vector.
   */
  void
  initialize_dof_vector(VectorType &vec) const override
  {
    matrix_free.initialize_dof_vector(vec);
  }

  /**
   * Compute inverse of diagonal.
   */
  void
  compute_inverse_diagonal(VectorType &diagonal) const override
  {
    this->initialize_dof_vector(diagonal);

    const unsigned int n_components  = dof_handler.get_fe().n_components();
    const unsigned int fe_degree     = dof_handler.get_fe().tensor_degree();
    const unsigned int n_q_points_1d = quadrature.get_tensor_basis()[0].size();

    if (n_components == 1 && fe_degree == 1 && n_q_points_1d == 2)
      this->compute_inverse_diagonal_internal<1, 1, 2>(diagonal);
    else if (n_components == 1 && fe_degree == 2 && n_q_points_1d == 3)
      this->compute_inverse_diagonal_internal<1, 2, 3>(diagonal);
    else if (n_components == dim && fe_degree == 1 && n_q_points_1d == 2)
      this->compute_inverse_diagonal_internal<dim, 1, 2>(diagonal);
    else if (n_components == dim && fe_degree == 2 && n_q_points_1d == 3)
      this->compute_inverse_diagonal_internal<dim, 2, 3>(diagonal);
    else if (n_components == 1 && fe_degree == 1 && n_q_points_1d == 3)
      this->compute_inverse_diagonal_internal<1, 1, 3>(diagonal);
    else if (n_components == 1 && fe_degree == 2 && n_q_points_1d == 4)
      this->compute_inverse_diagonal_internal<1, 2, 4>(diagonal);
    else if (n_components == dim && fe_degree == 1 && n_q_points_1d == 3)
      this->compute_inverse_diagonal_internal<dim, 1, 3>(diagonal);
    else if (n_components == dim && fe_degree == 2 && n_q_points_1d == 4)
      this->compute_inverse_diagonal_internal<dim, 2, 4>(diagonal);
    else
      AssertThrow(false, ExcInternalError());
  }

private:
  /**
   * Templated vmult function.
   */
  template <int n_components, int fe_degree, int n_q_points_1d>
  void
  vmult_internal(VectorType &dst, const VectorType &src) const
  {
    if (bp <= BPType::BP2) // mass matrix
      {
        OperatorDealiiMassLocal<dim, fe_degree, n_q_points_1d, n_components, Number> mass_operator;
        matrix_free.cell_loop(mass_operator, src, dst);
      }
    else
      {
        OperatorDealiiLaplaceLocal<dim, fe_degree, n_q_points_1d, n_components, Number>
          local_operator;
        matrix_free.cell_loop(local_operator, src, dst);
      }
  }

  /**
   * Templated compute_inverse_diagonal function.
   */
  template <int n_components, int fe_degree, int n_q_points_1d>
  void
  compute_inverse_diagonal_internal(VectorType &diagonal) const
  {
    if (bp <= BPType::BP2) // mass matrix
      {
        OperatorDealiiMassQuad<dim, fe_degree, n_q_points_1d, n_components, Number> op_quad;

        MatrixFreeTools::compute_diagonal<dim, fe_degree, n_q_points_1d, n_components, Number>(
          matrix_free, diagonal, op_quad, EvaluationFlags::values, EvaluationFlags::values);
      }
    else
      {
        OperatorDealiiLaplaceQuad<dim, fe_degree, n_q_points_1d, n_components, Number> op_quad;

        MatrixFreeTools::compute_diagonal<dim, fe_degree, n_q_points_1d, n_components, Number>(
          matrix_free, diagonal, op_quad, EvaluationFlags::gradients, EvaluationFlags::gradients);
      }


    Number *diagonal_ptr = diagonal.get_values();

    Kokkos::parallel_for(
      "lethe::invert_vector",
      Kokkos::RangePolicy<MemorySpace::Default::kokkos_space::execution_space>(
        0, diagonal.locally_owned_size()),
      KOKKOS_LAMBDA(int i) { diagonal_ptr[i] = 1.0 / diagonal_ptr[i]; });
  }

  /**
   * Mapping object passed to the constructor.
   */
  const Mapping<dim> &mapping;

  /**
   * DoFHandler object passed to the constructor.
   */
  const DoFHandler<dim> &dof_handler;

  /**
   * Constraints object passed to the constructor.
   */
  const AffineConstraints<Number> &constraints;

  /**
   * Quadrature rule object passed to the constructor.
   */
  const Quadrature<dim> &quadrature;

  /**
   * Selected BP.
   */
  const BPType bp;

  /**
   * MatrixFree object.
   */
  Portable::MatrixFree<dim, Number> matrix_free;
};

#endif
