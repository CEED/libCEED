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

#include "bps.h"

// deal.II includes
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

// boost
#include <boost/algorithm/string.hpp>

#include <sstream>

// include operators

// Test cases
// TESTARGS(name="BP1") --resource {ceed_resource} --bp BP1 --fe_degree 2 --print_timings 0
// TESTARGS(name="BP4") --resource {ceed_resource} --bp BP4 --fe_degree 3 --print_timings 0

/**
 * Relevant parameters.
 */
struct Parameters
{
  BPType       bp                   = BPType::BP5;
  unsigned int n_global_refinements = 1;
  unsigned int fe_degree            = 2;
  bool         print_timings        = true;
  std::string  libCEED_resource     = "/cpu/self/avx/blocked";

  bool
  parse(int argc, char *argv[])
  {
    if (argc == 1 && (std::string(argv[0]) == "--help"))
      {
        std::cout << "Usage: ./bp [OPTION]..." << std::endl;
        std::cout << std::endl;
        std::cout << "--bp             name of benchmark (BP1-BP6)" << std::endl;
        std::cout << "--n_refinements  number of refinements (0-)" << std::endl;
        std::cout << "--fe_degree      polynomial degree (1-)" << std::endl;
        std::cout << "--print_timings  name of benchmark (0, 1)" << std::endl;
        std::cout << "--resource       name of resource (e.g., /cpu/self/avx/blocked)" << std::endl;

        return true;
      }

    AssertThrow(argc % 2 == 0, ExcInternalError());

    while (argc > 0)
      {
        std::string label(argv[0]);

        if ("--bp" == label)
          {
            std::string bp_string(argv[1]);

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
        else if ("--n_refinements" == label)
          {
            n_global_refinements = std::atoi(argv[1]);
          }
        else if ("--fe_degree" == label)
          {
            fe_degree = std::atoi(argv[1]);
          }
        else if ("--print_timings" == label)
          {
            print_timings = std::atoi(argv[1]);
          }
        else if ("--resource" == label)
          {
            libCEED_resource = std::string(argv[1]);
          }
        else
          {
            AssertThrow(false, ExcNotImplemented());
          }


        argc -= 2;
        argv += 2;
      }

    return false;
  }
};



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



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  Parameters params;
  if (params.parse(argc - 1, argv + 1))
    return 0;

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

#ifdef DEAL_II_WITH_P4EST
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
#else
  parallel::shared::Triangulation<dim> tria(MPI_COMM_WORLD, ::Triangulation<dim>::none, true);
#endif

  GridGenerator::hyper_cube(tria);
  tria.refine_global(n_refinements);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  DoFRenumbering::support_point_wise(dof_handler);

  AffineConstraints<Number> constraints;

  if (!(bp == BPType::BP1 || bp == BPType::BP2))
    {
      // for stiffness matrix
      DoFTools::make_zero_boundary_constraints(dof_handler, constraints);
      constraints.close();
    }

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

    bool not_converged = false;

    try
      {
        // solve problem
        SolverCG<VectorType> solver(reduction_control);
        now = std::chrono::system_clock::now();
        solver.solve(op, v, u, diagonal_matrix);
      }
    catch (const SolverControl::NoConvergence &)
      {
        pout << "Error: solver failed to converge with" << std::endl;
        not_converged = true;
      }


    const auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - now)
        .count() /
      1e9;


    if (params.print_timings || not_converged)
      {
        pout << label << ": " << reduction_control.last_step() << " " << v.l2_norm() << " "
             << (params.print_timings ? time : 0.0) << std::endl;
      }
  };

  // create and test the libCEED operator
  OperatorCeed<dim, Number> op_ceed(
    mapping, dof_handler, constraints, quadrature, bp, params.libCEED_resource);
  test("ceed", op_ceed);

  // create and test a native deal.II operator
  OperatorDealii<dim, Number> op_dealii(mapping, dof_handler, constraints, quadrature, bp);
  test("dealii", op_dealii);
}
