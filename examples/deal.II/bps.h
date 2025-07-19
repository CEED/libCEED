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
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/shape_info.h>
#include <deal.II/matrix_free/tools.h>

// libCEED includes
#include <ceed.h>
#include <ceed/backend.h>

// QFunction source
#include "bps-qfunctions.h"

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
template <typename Number>
class OperatorBase
{
public:
  /**
   * deal.II vector type
   */
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

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


/**
 * Operator implementation using libCEED.
 */
template <int dim, typename Number>
class OperatorCeed : public OperatorBase<Number>
{
public:
  using VectorType = typename OperatorBase<Number>::VectorType;

  /**
   * Constructor.
   */
  OperatorCeed(const Mapping<dim>              &mapping,
               const DoFHandler<dim>           &dof_handler,
               const AffineConstraints<Number> &constraints,
               const Quadrature<dim>           &quadrature,
               const BPType                    &bp,
               const std::string               &resource)
    : mapping(mapping)
    , dof_handler(dof_handler)
    , constraints(constraints)
    , quadrature(quadrature)
    , bp(bp)
    , resource(resource)
  {
    reinit();
  }

  /**
   * Destructor.
   */
  ~OperatorCeed()
  {
    CeedVectorDestroy(&src_ceed);
    CeedVectorDestroy(&dst_ceed);
    CeedOperatorDestroy(&op_apply);
    CeedDestroy(&ceed);
  }

  /**
   * Initialized internal data structures, particularly, libCEED.
   */
  void
  reinit() override
  {
    CeedVector           q_data;
    CeedBasis            sol_basis;
    CeedElemRestriction  sol_restriction;
    CeedElemRestriction  q_data_restriction;
    BuildContext         build_ctx_data;
    CeedQFunctionContext build_ctx;
    CeedQFunction        qf_apply;

    const auto &tria = dof_handler.get_triangulation();
    const auto &fe   = dof_handler.get_fe();

    const auto n_components = fe.n_components();

    if (bp == BPType::BP1 || bp == BPType::BP3 || bp == BPType::BP5)
      {
        AssertThrow(n_components == 1, ExcInternalError());
      }
    else
      {
        AssertThrow(n_components == dim, ExcInternalError());
      }

    // 1) create CEED instance -> "MatrixFree"
    const char *ceed_spec = resource.c_str();
    CeedInit(ceed_spec, &ceed);

    // 2) create shape functions -> "ShapeInfo"
    const unsigned int fe_degree  = fe.tensor_degree();
    const unsigned int n_q_points = quadrature.get_tensor_basis()[0].size();
    {
      const dealii::internal::MatrixFreeFunctions::ShapeInfo<double> shape_info(quadrature, fe, 0);
      const auto             &shape_data = shape_info.get_shape_data();
      std::vector<CeedScalar> q_ref_1d;
      for (const auto q : shape_data.quadrature.get_points())
        q_ref_1d.push_back(q(0));

      // transpose bases for compatibility with restriction
      std::vector<CeedScalar> interp_1d(shape_data.shape_values.size());
      std::vector<CeedScalar> grad_1d(shape_data.shape_gradients.size());
      for (unsigned int i = 0; i < n_q_points; ++i)
        for (unsigned int j = 0; j < fe_degree + 1; ++j)
          {
            interp_1d[j + i * (fe_degree + 1)] = shape_data.shape_values[j * n_q_points + i];
            grad_1d[j + i * (fe_degree + 1)]   = shape_data.shape_gradients[j * n_q_points + i];
          }

      CeedBasisCreateTensorH1(ceed,
                              dim,
                              n_components,
                              fe_degree + 1,
                              n_q_points,
                              interp_1d.data(),
                              grad_1d.data(),
                              q_ref_1d.data(),
                              quadrature.get_tensor_basis()[0].get_weights().data(),
                              &sol_basis);
    }

    // 3) create restriction matrix -> DoFInfo
    unsigned int n_local_active_cells = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        n_local_active_cells++;

    partitioner =
      std::make_shared<Utilities::MPI::Partitioner>(dof_handler.locally_owned_dofs(),
                                                    DoFTools::extract_locally_active_dofs(
                                                      dof_handler),
                                                    dof_handler.get_communicator());

    std::vector<CeedInt> indices;
    indices.reserve(n_local_active_cells * fe.n_dofs_per_cell() / n_components);

    const auto dof_mapping = FETools::lexicographic_to_hierarchic_numbering<dim>(fe_degree);

    std::vector<types::global_dof_index> local_indices(fe.n_dofs_per_cell());

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(local_indices);

          for (const auto i : dof_mapping)
            indices.emplace_back(
              partitioner->global_to_local(local_indices[fe.component_to_system_index(0, i)]));
        }

    CeedElemRestrictionCreate(ceed,
                              n_local_active_cells,
                              fe.n_dofs_per_cell() / n_components,
                              n_components,
                              1,
                              this->extended_local_size(),
                              CEED_MEM_HOST,
                              CEED_COPY_VALUES,
                              indices.data(),
                              &sol_restriction);

    // 4) create mapping -> MappingInfo
    const unsigned int n_components_metric = (bp <= BPType::BP2) ? 1 : (dim * (dim + 1) / 2);

    this->metric_data = compute_metric_data(ceed, mapping, tria, quadrature, bp);

    strides = {{1,
                static_cast<int>(quadrature.size()),
                static_cast<int>(quadrature.size() * n_components_metric)}};
    CeedVectorCreate(ceed, metric_data.size(), &q_data);
    CeedVectorSetArray(q_data, CEED_MEM_HOST, CEED_USE_POINTER, metric_data.data());
    CeedElemRestrictionCreateStrided(ceed,
                                     n_local_active_cells,
                                     quadrature.size(),
                                     n_components_metric,
                                     metric_data.size(),
                                     strides.data(),
                                     &q_data_restriction);

    build_ctx_data.dim       = dim;
    build_ctx_data.space_dim = dim;

    CeedQFunctionContextCreate(ceed, &build_ctx);
    CeedQFunctionContextSetData(
      build_ctx, CEED_MEM_HOST, CEED_COPY_VALUES, sizeof(build_ctx_data), &build_ctx_data);

    // 5) create q operation
    if (bp == BPType::BP1)
      CeedQFunctionCreateInterior(ceed, 1, f_apply_mass, f_apply_mass_loc, &qf_apply);
    else if (bp == BPType::BP2)
      CeedQFunctionCreateInterior(ceed, 1, f_apply_mass_vec, f_apply_mass_vec_loc, &qf_apply);
    else if (bp == BPType::BP3 || bp == BPType::BP5)
      CeedQFunctionCreateInterior(ceed, 1, f_apply_poisson, f_apply_poisson_loc, &qf_apply);
    else if (bp == BPType::BP4 || bp == BPType::BP6)
      CeedQFunctionCreateInterior(ceed, 1, f_apply_poisson_vec, f_apply_poisson_vec_loc, &qf_apply);
    else
      AssertThrow(false, ExcInternalError());

    if (bp <= BPType::BP2)
      CeedQFunctionAddInput(qf_apply, "u", n_components, CEED_EVAL_INTERP);
    else
      CeedQFunctionAddInput(qf_apply, "u", dim * n_components, CEED_EVAL_GRAD);

    CeedQFunctionAddInput(qf_apply, "qdata", n_components_metric, CEED_EVAL_NONE);

    if (bp <= BPType::BP2)
      CeedQFunctionAddOutput(qf_apply, "v", n_components, CEED_EVAL_INTERP);
    else
      CeedQFunctionAddOutput(qf_apply, "v", dim * n_components, CEED_EVAL_GRAD);

    CeedQFunctionSetContext(qf_apply, build_ctx);

    // 6) put everything together
    CeedOperatorCreate(ceed, qf_apply, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_apply);

    CeedOperatorSetField(op_apply, "u", sol_restriction, sol_basis, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(op_apply, "qdata", q_data_restriction, CEED_BASIS_NONE, q_data);
    CeedOperatorSetField(op_apply, "v", sol_restriction, sol_basis, CEED_VECTOR_ACTIVE);

    // 7) libCEED vectors
    CeedElemRestrictionCreateVector(sol_restriction, &src_ceed, NULL);
    CeedElemRestrictionCreateVector(sol_restriction, &dst_ceed, NULL);

    // 8) cleanup
    CeedVectorDestroy(&q_data);
    CeedElemRestrictionDestroy(&q_data_restriction);
    CeedElemRestrictionDestroy(&sol_restriction);
    CeedBasisDestroy(&sol_basis);
    CeedQFunctionContextDestroy(&build_ctx);
    CeedQFunctionDestroy(&qf_apply);
  }

  /**
   * Perform matrix-vector product.
   */
  void
  vmult(VectorType &dst, const VectorType &src) const override
  {
    // communicate: update ghost values
    src.update_ghost_values();

    // pass memory buffers to libCEED
    VectorTypeCeed x(src_ceed);
    VectorTypeCeed y(dst_ceed);
    x.import_array(src, CEED_MEM_HOST);
    y.import_array(dst, CEED_MEM_HOST);

    // apply operator
    CeedOperatorApply(op_apply, x(), y(), CEED_REQUEST_IMMEDIATE);

    // pull arrays back to deal.II
    x.take_array();
    y.take_array();

    // communicate: compress
    src.zero_out_ghost_values();
    dst.compress(VectorOperation::add);

    // apply constraints: we assume homogeneous DBC
    constraints.set_zero(dst);
  }

  /**
   * Initialized vector.
   */
  void
  initialize_dof_vector(VectorType &vec) const override
  {
    vec.reinit(partitioner);
  }

  /**
   * Compute inverse of diagonal.
   */
  void
  compute_inverse_diagonal(VectorType &diagonal) const override
  {
    this->initialize_dof_vector(diagonal);

    // pass memory buffer to libCEED
    VectorTypeCeed y(dst_ceed);
    y.import_array(diagonal, CEED_MEM_HOST);

    CeedOperatorLinearAssembleDiagonal(op_apply, y(), CEED_REQUEST_IMMEDIATE);

    // pull array back to deal.II
    y.take_array();

    diagonal.compress(VectorOperation::add);

    // apply constraints: we assume homogeneous DBC
    constraints.set_zero(diagonal);

    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;
  }

private:
  /**
   * Wrapper around a deal.II vector to create a libCEED vector view.
   */
  class VectorTypeCeed
  {
  public:
    /**
     * Constructor.
     */
    VectorTypeCeed(const CeedVector &vec_orig)
    {
      vec_ceed = NULL;
      CeedVectorReferenceCopy(vec_orig, &vec_ceed);
    }

    /**
     * Return libCEED vector view.
     */
    CeedVector &
    operator()()
    {
      return vec_ceed;
    }

    /**
     * Set deal.II memory in libCEED vector.
     */
    void
    import_array(const VectorType &vec, const CeedMemType space)
    {
      mem_space = space;
      CeedVectorSetArray(vec_ceed, mem_space, CEED_USE_POINTER, vec.get_values());
    }

    /**
     * Sync memory from device to host.
     */
    void
    sync_array()
    {
      CeedVectorSyncArray(vec_ceed, mem_space);
    }

    /**
     * Take previously set deal.II array from libCEED vector
     */
    void
    take_array()
    {
      CeedScalar *ptr;
      CeedVectorTakeArray(vec_ceed, mem_space, &ptr);
    }

    /**
     * Destructor: destroy vector view.
     */
    ~VectorTypeCeed()
    {
      bool has_array;
      CeedVectorHasBorrowedArrayOfType(vec_ceed, mem_space, &has_array);
      if (has_array)
        {
          CeedScalar *ptr;
          CeedVectorTakeArray(vec_ceed, mem_space, &ptr);
        }
      CeedVectorDestroy(&vec_ceed);
    }

  private:
    /**
     * libCEED vector view.
     */
    CeedMemType mem_space;
    CeedVector  vec_ceed;
  };

  /**
   * Number of locally active DoFs.
   */
  unsigned int
  extended_local_size() const
  {
    return partitioner->locally_owned_size() + partitioner->n_ghost_indices();
  }

  /**
   * Compute metric data: Jacobian, ...
   */
  static std::vector<double>
  compute_metric_data(const Ceed               &ceed,
                      const Mapping<dim>       &mapping,
                      const Triangulation<dim> &tria,
                      const Quadrature<dim>    &quadrature,
                      const BPType              bp)
  {
    std::vector<double> metric_data;

    CeedBasis            geo_basis;
    CeedVector           q_data;
    CeedElemRestriction  q_data_restriction;
    CeedVector           node_coords;
    CeedElemRestriction  geo_restriction;
    CeedQFunctionContext build_ctx;
    CeedQFunction        qf_build;
    CeedOperator         op_build;

    const unsigned int n_q_points = quadrature.get_tensor_basis()[0].size();

    const unsigned int n_components_metric = (bp <= BPType::BP2) ? 1 : (dim * (dim + 1) / 2);

    const auto mapping_q = dynamic_cast<const MappingQ<dim> *>(&mapping);

    AssertThrow(mapping_q, ExcMessage("Wrong mapping!"));

    const unsigned int fe_degree = mapping_q->get_degree();

    FE_Q<dim> geo_fe(fe_degree);

    {
      const dealii::internal::MatrixFreeFunctions::ShapeInfo<double> shape_info(quadrature,
                                                                                geo_fe,
                                                                                0);
      const auto             &shape_data = shape_info.get_shape_data();
      std::vector<CeedScalar> q_ref_1d;
      for (const auto q : shape_data.quadrature.get_points())
        q_ref_1d.push_back(q(0));

      // transpose bases for compatibility with restriction
      std::vector<CeedScalar> interp_1d(shape_data.shape_values.size());
      std::vector<CeedScalar> grad_1d(shape_data.shape_gradients.size());
      for (unsigned int i = 0; i < n_q_points; ++i)
        for (unsigned int j = 0; j < fe_degree + 1; ++j)
          {
            interp_1d[j + i * (fe_degree + 1)] = shape_data.shape_values[j * n_q_points + i];
            grad_1d[j + i * (fe_degree + 1)]   = shape_data.shape_gradients[j * n_q_points + i];
          }

      CeedBasisCreateTensorH1(ceed,
                              dim,
                              dim,
                              fe_degree + 1,
                              n_q_points,
                              interp_1d.data(),
                              grad_1d.data(),
                              q_ref_1d.data(),
                              quadrature.get_tensor_basis()[0].get_weights().data(),
                              &geo_basis);
    }

    unsigned int n_local_active_cells = 0;

    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned())
        n_local_active_cells++;

    std::vector<double>  geo_support_points;
    std::vector<CeedInt> geo_indices;

    DoFHandler<dim> geo_dof_handler(tria);
    geo_dof_handler.distribute_dofs(geo_fe);

    const auto geo_partitioner =
      std::make_shared<Utilities::MPI::Partitioner>(geo_dof_handler.locally_owned_dofs(),
                                                    DoFTools::extract_locally_active_dofs(
                                                      geo_dof_handler),
                                                    geo_dof_handler.get_communicator());

    geo_indices.reserve(n_local_active_cells * geo_fe.n_dofs_per_cell());

    const auto dof_mapping = FETools::lexicographic_to_hierarchic_numbering<dim>(fe_degree);

    FEValues<dim> fe_values(mapping,
                            geo_fe,
                            geo_fe.get_unit_support_points(),
                            update_quadrature_points);

    std::vector<types::global_dof_index> local_indices(geo_fe.n_dofs_per_cell());

    const unsigned int n_points =
      geo_partitioner->locally_owned_size() + geo_partitioner->n_ghost_indices();

    geo_support_points.resize(dim * n_points);

    for (const auto &cell : geo_dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          cell->get_dof_indices(local_indices);

          for (const auto i : dof_mapping)
            {
              const auto index = geo_partitioner->global_to_local(local_indices[i]);
              geo_indices.emplace_back(index * dim);

              const auto point = fe_values.quadrature_point(i);

              for (unsigned int d = 0; d < dim; ++d)
                geo_support_points[index * dim + d] = point[d];
            }
        }

    metric_data.resize(n_local_active_cells * quadrature.size() * n_components_metric);

    CeedInt strides[3] = {1,
                          static_cast<int>(quadrature.size()),
                          static_cast<int>(quadrature.size() * n_components_metric)};

    CeedVectorCreate(ceed, metric_data.size(), &q_data);
    CeedVectorSetArray(q_data, CEED_MEM_HOST, CEED_USE_POINTER, metric_data.data());
    CeedElemRestrictionCreateStrided(ceed,
                                     n_local_active_cells,
                                     quadrature.size(),
                                     n_components_metric,
                                     metric_data.size(),
                                     strides,
                                     &q_data_restriction);

    CeedVectorCreate(ceed, geo_support_points.size(), &node_coords);
    CeedVectorSetArray(node_coords, CEED_MEM_HOST, CEED_USE_POINTER, geo_support_points.data());

    CeedElemRestrictionCreate(ceed,
                              n_local_active_cells,
                              geo_fe.n_dofs_per_cell(),
                              dim,
                              1,
                              geo_support_points.size(),
                              CEED_MEM_HOST,
                              CEED_COPY_VALUES,
                              geo_indices.data(),
                              &geo_restriction);

    BuildContext build_ctx_data;
    build_ctx_data.dim       = dim;
    build_ctx_data.space_dim = dim;

    CeedQFunctionContextCreate(ceed, &build_ctx);
    CeedQFunctionContextSetData(
      build_ctx, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(build_ctx_data), &build_ctx_data);

    // 5) create q operation
    if (bp <= BPType::BP2)
      CeedQFunctionCreateInterior(ceed, 1, f_build_mass, f_build_mass_loc, &qf_build);
    else
      CeedQFunctionCreateInterior(ceed, 1, f_build_poisson, f_build_poisson_loc, &qf_build);

    CeedQFunctionAddInput(qf_build, "geo", dim * dim, CEED_EVAL_GRAD);
    CeedQFunctionAddInput(qf_build, "metric_data", 1, CEED_EVAL_WEIGHT);
    CeedQFunctionAddOutput(qf_build, "qdata", n_components_metric, CEED_EVAL_NONE);
    CeedQFunctionSetContext(qf_build, build_ctx);

    // 6) put everything together
    CeedOperatorCreate(ceed, qf_build, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_build);
    CeedOperatorSetField(op_build, "geo", geo_restriction, geo_basis, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(
      op_build, "metric_data", CEED_ELEMRESTRICTION_NONE, geo_basis, CEED_VECTOR_NONE);
    CeedOperatorSetField(
      op_build, "qdata", q_data_restriction, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);

    CeedOperatorApply(op_build, node_coords, q_data, CEED_REQUEST_IMMEDIATE);

    CeedVectorDestroy(&node_coords);
    CeedVectorSyncArray(q_data, CEED_MEM_HOST);
    CeedVectorDestroy(&q_data);
    CeedElemRestrictionDestroy(&geo_restriction);
    CeedElemRestrictionDestroy(&q_data_restriction);
    CeedBasisDestroy(&geo_basis);
    CeedQFunctionContextDestroy(&build_ctx);
    CeedQFunctionDestroy(&qf_build);
    CeedOperatorDestroy(&op_build);

    return metric_data;
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
   * Resource name.
   */
  const std::string resource;

  /**
   * Partitioner for distributed vectors.
   */
  std::shared_ptr<Utilities::MPI::Partitioner> partitioner;

  /**
   * libCEED data structures.
   */
  Ceed                   ceed;
  std::vector<double>    metric_data;
  std::array<CeedInt, 3> strides;
  CeedVector             src_ceed;
  CeedVector             dst_ceed;
  CeedOperator           op_apply;
};



template <int dim, typename Number>
class OperatorDealii : public OperatorBase<Number>
{
public:
  using VectorType = typename OperatorBase<Number>::VectorType;

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
