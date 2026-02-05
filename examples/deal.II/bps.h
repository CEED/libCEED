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
#ifndef bps_h
#  define bps_h

// deal.II includes
#  include <deal.II/dofs/dof_tools.h>

#  include <deal.II/fe/mapping.h>

#  include <deal.II/lac/la_parallel_vector.h>

#  include <deal.II/matrix_free/fe_evaluation.h>
#  include <deal.II/matrix_free/matrix_free.h>
#  include <deal.II/matrix_free/shape_info.h>
#  include <deal.II/matrix_free/tools.h>

using namespace dealii;



/**
 * BP types. For more details, see https://ceed.exascaleproject.org/bps/.
 */
enum class BPType : unsigned int
{
  BP1,
  BP2,
  BP3,
  BP4,
  BP5,
  BP6
};



/**
 * Struct storing relevant information regarding each BP.
 */
struct BPInfo
{
  BPInfo(const BPType type, const int dim, const int fe_degree)
    : type(type)
    , dim(dim)
    , fe_degree(fe_degree)
  {
    if (type == BPType::BP1)
      type_string = "BP1";
    else if (type == BPType::BP2)
      type_string = "BP2";
    else if (type == BPType::BP3)
      type_string = "BP3";
    else if (type == BPType::BP4)
      type_string = "BP4";
    else if (type == BPType::BP5)
      type_string = "BP5";
    else if (type == BPType::BP6)
      type_string = "BP6";

    this->n_q_points_1d = (type <= BPType::BP4) ? (fe_degree + 2) : (fe_degree + 1);

    this->n_components =
      (type == BPType::BP1 || type == BPType::BP3 || type == BPType::BP5) ? 1 : dim;
  }


  BPType       type;
  std::string  type_string;
  unsigned int dim;
  unsigned int fe_degree;
  unsigned int n_q_points_1d;
  unsigned int n_components;
};



/**
 * Base class of operators.
 */
template <typename Number, typename MemorySpace>
class OperatorBase
{
public:
  /**
   * deal.II vector type
   */
  using VectorType = LinearAlgebra::distributed::Vector<Number, MemorySpace>;

  /**
   * Initialize vector.
   */
  virtual void
  reinit() = 0;

  /**
   * Perform matrix-vector product
   */
  virtual void
  vmult(VectorType &dst, const VectorType &src) const = 0;

  /**
   * Initialize vector.
   */
  virtual void
  initialize_dof_vector(VectorType &vec) const = 0;

  /**
   * Compute inverse of diagonal.
   */
  virtual void
  compute_inverse_diagonal(VectorType &diagonal) const = 0;
};

#endif
