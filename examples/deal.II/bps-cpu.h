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
#ifndef bps_cpu_h
#  define bps_cpu_h

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



/**
 * Operator CPU implementation using deal.II.
 */
template <int dim, typename Number>
class OperatorDealii : public OperatorBase<Number, MemorySpace::Host>
{
public:
  using VectorType = typename OperatorBase<Number, MemorySpace::Host>::VectorType;

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
    typename MatrixFree<dim, Number>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme =
      MatrixFree<dim, Number>::AdditionalData::TasksParallelScheme::none;

    // create MatrixFree
    matrix_free.reinit(mapping, dof_handler, constraints, quadrature, additional_data);
  }

  /**
   * Matrix-vector product.
   */
  void
  vmult(VectorType &dst, const VectorType &src) const override
  {
    if (dof_handler.get_fe().n_components() == 1)
      {
        matrix_free.cell_loop(&OperatorDealii::do_cell_integral_range<1>, this, dst, src, true);
      }
    else
      {
        AssertThrow(dof_handler.get_fe().n_components() == dim, ExcInternalError());

        matrix_free.cell_loop(&OperatorDealii::do_cell_integral_range<dim>, this, dst, src, true);
      }
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

    if (dof_handler.get_fe().n_components() == 1)
      {
        MatrixFreeTools::compute_diagonal(matrix_free,
                                          diagonal,
                                          &OperatorDealii::do_cell_integral_local<1>,
                                          this);
      }
    else
      {
        AssertThrow(dof_handler.get_fe().n_components() == dim, ExcInternalError());

        MatrixFreeTools::compute_diagonal(matrix_free,
                                          diagonal,
                                          &OperatorDealii::do_cell_integral_local<dim>,
                                          this);
      }

    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;
  }

private:
  /**
   * Cell integral without vector access.
   */
  template <int n_components>
  void
  do_cell_integral_local(FEEvaluation<dim, -1, 0, n_components, Number> &phi) const
  {
    if (bp <= BPType::BP2) // mass matrix
      {
        phi.evaluate(EvaluationFlags::values);
        for (const auto q : phi.quadrature_point_indices())
          phi.submit_value(phi.get_value(q), q);
        phi.integrate(EvaluationFlags::values);
      }
    else // Poisson operator
      {
        phi.evaluate(EvaluationFlags::gradients);
        for (const auto q : phi.quadrature_point_indices())
          phi.submit_gradient(phi.get_gradient(q), q);
        phi.integrate(EvaluationFlags::gradients);
      }
  }

  /**
   * Cell integral on a range of cells.
   */
  template <int n_components>
  void
  do_cell_integral_range(const MatrixFree<dim, Number>               &matrix_free,
                         VectorType                                  &dst,
                         const VectorType                            &src,
                         const std::pair<unsigned int, unsigned int> &range) const
  {
    FEEvaluation<dim, -1, 0, n_components, Number> phi(matrix_free, range);

    for (unsigned cell = range.first; cell < range.second; ++cell)
      {
        phi.reinit(cell);
        phi.read_dof_values(src);            // read source vector
        do_cell_integral_local(phi);         // cell integral
        phi.distribute_local_to_global(dst); // write to destination vector
      }
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
  MatrixFree<dim, Number> matrix_free;
};

#endif
