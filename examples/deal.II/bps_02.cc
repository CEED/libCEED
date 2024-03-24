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

#include <deal.II/matrix_free/cuda_fe_evaluation.h>
#include <deal.II/matrix_free/cuda_matrix_free.h>

// boost
#include <boost/algorithm/string.hpp>

#include <sstream>

// include operators

// Test cases
// TESTARGS(name="BP1") --resource {ceed_resource} --bp BP1 --fe_degree 2 --print_timings 0
// TESTARGS(name="BP4") --resource {ceed_resource} --bp BP5 --fe_degree 3 --print_timings 0

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
        std::cout << "--bp             name of benchmark (BP1, BP5)" << std::endl;
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
              bp = BPType::BP1; // with q = p + 1
            else if (bp_string == "BP5")
              bp = BPType::BP5;
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



template <int dim, int fe_degree, typename Number>
class OperatorDealiiMassQuad
{
public:
  DEAL_II_HOST_DEVICE void
  operator()(CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> *fe_eval,
             const int                                                             q_point) const
  {
    fe_eval->submit_value(fe_eval->get_value(q_point), q_point);
  }
};



template <int dim, int fe_degree, typename Number>
class OperatorDealiiLaplaceQuad
{
public:
  DEAL_II_HOST_DEVICE void
  operator()(CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> *fe_eval,
             const int                                                             q_point) const
  {
    fe_eval->submit_gradient(fe_eval->get_gradient(q_point), q_point);
  }
};



template <int dim, int fe_degree, typename Number>
class OperatorDealiiMassLocal
{
public:
  DEAL_II_HOST_DEVICE void
  operator()(const unsigned int                                          cell,
             const typename CUDAWrappers::MatrixFree<dim, Number>::Data *gpu_data,
             CUDAWrappers::SharedData<dim, Number>                      *shared_data,
             const Number                                               *src,
             Number                                                     *dst) const
  {
    (void)cell;

    CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> fe_eval(/*cell,*/ gpu_data,
                                                                                 shared_data);
    fe_eval.read_dof_values(src);
    fe_eval.evaluate(EvaluationFlags::values);
    fe_eval.apply_for_each_quad_point(OperatorDealiiMassQuad<dim, fe_degree, Number>());
    fe_eval.integrate(EvaluationFlags::values);
    fe_eval.distribute_local_to_global(dst);
  }
  static const unsigned int n_dofs_1d    = fe_degree + 1;
  static const unsigned int n_local_dofs = Utilities::pow(fe_degree + 1, dim);
  static const unsigned int n_q_points   = Utilities::pow(fe_degree + 1, dim);
};



template <int dim, int fe_degree, typename Number>
class OperatorDealiiLaplaceLocal
{
public:
  DEAL_II_HOST_DEVICE void
  operator()(const unsigned int                                          cell,
             const typename CUDAWrappers::MatrixFree<dim, Number>::Data *gpu_data,
             CUDAWrappers::SharedData<dim, Number>                      *shared_data,
             const Number                                               *src,
             Number                                                     *dst) const
  {
    (void)cell;

    CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> fe_eval(/*cell,*/ gpu_data,
                                                                                 shared_data);
    fe_eval.read_dof_values(src);
    fe_eval.evaluate(EvaluationFlags::gradients);
    fe_eval.apply_for_each_quad_point(OperatorDealiiLaplaceQuad<dim, fe_degree, Number>());
    fe_eval.integrate(EvaluationFlags::gradients);
    fe_eval.distribute_local_to_global(dst);
  }
  static const unsigned int n_dofs_1d    = fe_degree + 1;
  static const unsigned int n_local_dofs = Utilities::pow(fe_degree + 1, dim);
  static const unsigned int n_q_points   = Utilities::pow(fe_degree + 1, dim);
};



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
    typename CUDAWrappers::MatrixFree<dim, Number>::AdditionalData additional_data;

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

    const unsigned int fe_degree = dof_handler.get_fe().tensor_degree();

    if (fe_degree == 1)
      this->vmult_internal<1>(dst, src);
    else if (fe_degree == 2)
      this->vmult_internal<2>(dst, src);
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
  compute_inverse_diagonal(VectorType &) const override
  {
    AssertThrow(false, ExcNotImplemented());
  }

private:
  /**
   * Templated vmult function.
   */
  template <int fe_degree>
  void
  vmult_internal(VectorType &dst, const VectorType &src) const
  {
    if (bp == BPType::BP1)
      {
        OperatorDealiiMassLocal<dim, fe_degree, Number> mass_operator;
        matrix_free.cell_loop(mass_operator, src, dst);
      }
    else if (bp == BPType::BP5)
      {
        OperatorDealiiLaplaceLocal<dim, fe_degree, Number> local_operator;
        matrix_free.cell_loop(local_operator, src, dst);
      }
    else
      {
        AssertThrow(false, ExcNotImplemented());
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
  CUDAWrappers::MatrixFree<dim, Number> matrix_free;
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

  using Number                  = double;
  using VectorType              = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;
  const unsigned int dim        = 2;
  const unsigned int fe_degree  = params.fe_degree;
  const unsigned int n_q_points = fe_degree + 1;
  const unsigned int n_refinements = params.n_global_refinements;
  const unsigned int n_components  = 1;

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

    std::chrono::time_point<std::chrono::system_clock> now;

    bool not_converged = false;

    try
      {
        // solve problem
        SolverCG<VectorType> solver(reduction_control);
        now = std::chrono::system_clock::now();
        solver.solve(op, v, u, PreconditionIdentity());
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
  OperatorCeed<dim, Number, MemorySpace::Default> op_ceed(
    mapping, dof_handler, constraints, quadrature, bp, params.libCEED_resource);
  test("ceed", op_ceed);

  // create and test a native deal.II operator
  OperatorDealii<dim, Number> op_dealii(mapping, dof_handler, constraints, quadrature, bp);
  test("dealii", op_dealii);
}
