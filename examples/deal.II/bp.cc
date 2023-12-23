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

// deal.II includes
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

// include operators
#include "bp.h"


/**
 * Relevant parameters.
 */
struct Parameters
{
  BPType       bp                   = BPType::BP5;
  unsigned int n_global_refinements = 1;
  unsigned int fe_degree            = 2;
  bool         print_timings        = true;

  void
  parse(const std::string file_name)
  {
    dealii::ParameterHandler prm;
    add_parameters(prm);

    prm.parse_input(file_name, "", true);

    if (bp_string == "BP1")
      bp = BPType::BP1;
    else if (bp_string == "BP2")
      bp = BPType::BP2;
    else if (bp_string == "BP3")
      bp = BPType::BP3;
    else if (bp_string == "BP4")
      bp = BPType::BP4;
    else if (bp_string == "BP5")
      bp = BPType::BP5;
    else if (bp_string == "BP6")
      bp = BPType::BP6;
    else
      AssertThrow(false, ExcInternalError());
  }

  void
  print()
  {
    dealii::ParameterHandler prm;
    add_parameters(prm);
    prm.print_parameters(std::cout, dealii::ParameterHandler::OutputStyle::ShortJSON);
  }

private:
  void
  add_parameters(ParameterHandler &prm)
  {
    prm.add_parameter("bp", bp_string, "", Patterns::Selection("BP1|BP2|BP3|BP4|BP5|BP6"));
    prm.add_parameter("n global refinements", n_global_refinements);
    prm.add_parameter("fe degree", fe_degree);
    prm.add_parameter("print timings", print_timings);
  }

private:
  std::string bp_string = "BP5";
};



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  Parameters params;

  if (argc > 1)
    params.parse(std::string(argv[1]));
  else
    {
      params.print();
      return 0;
    }

  ConditionalOStream pout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  //  configuration
  const BPType bp = params.bp;

  using Number                     = double;
  using VectorType                 = LinearAlgebra::distributed::Vector<Number>;
  const unsigned int dim           = 2;
  const unsigned int fe_degree     = params.fe_degree;
  const unsigned int n_q_points    = (bp <= BPType::BP4) ? (fe_degree + 2) : (fe_degree + 1);
  const unsigned int n_refinements = params.n_global_refinements;
  const unsigned int n_components =
    (bp == BPType::BP1 || bp == BPType::BP3 || bp == BPType::BP5) ? 1 : dim;

  // create mapping, quadrature, fe, mesh, ...
  MappingQ1<dim> mapping;
  QGauss<dim>    quadrature(n_q_points);
  FESystem<dim>  fe(FE_Q<dim>(fe_degree), n_components);

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tria);
  tria.refine_global(n_refinements);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<Number> constraints;

  if (!(bp == BPType::BP1 || bp == BPType::BP2))
    {
      // for stiffness matrix
      DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
      constraints.close();
    }

  DoFRenumbering::support_point_wise(dof_handler);

  const auto test = [&](const std::string &label, const auto &op) {
    (void)label;

    // initialize vector
    VectorType u, v;
    op.initialize_dof_vector(u);
    op.initialize_dof_vector(v);
    u = 1.0;

    constraints.set_zero(u);

    // perform matrix-vector product
    op.vmult(v, u);

    // create solver
    ReductionControl reduction_control(100, 1e-20, 1e-6);

    // create preconditioner
    DiagonalMatrix<VectorType> diagonal_matrix;
    op.compute_inverse_diagonal(diagonal_matrix.get_vector());

    std::chrono::time_point<std::chrono::system_clock> now;

    try
      {
        // solve problem
        SolverCG<VectorType> solver(reduction_control);
        now = std::chrono::system_clock::now();
        solver.solve(op, v, u, diagonal_matrix);
      }
    catch (const SolverControl::NoConvergence &)
      {
        // nothing to do
      }


    const auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - now)
        .count() /
      1e9;


    pout << label << ": " << reduction_control.last_step() << " " << v.l2_norm() << " "
         << (params.print_timings ? time : 0.0) << std::endl;
  };

  // create and test the libCEED operator
  OperatorCeed<dim, Number> op_ceed(mapping, dof_handler, constraints, quadrature, bp);
  test("ceed", op_ceed);

  // create and test a native deal.II operator
  OperatorDealii<dim, Number> op_dealii(mapping, dof_handler, constraints, quadrature, bp);
  test("dealii", op_dealii);
}
