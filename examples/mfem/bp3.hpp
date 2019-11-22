// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

/// @file
/// Diffusion operator example using MFEM
#include <ceed.h>
#include <mfem.hpp>

#include "bp3.h"

/// Wrapper for a diffusion CeedOperator as an mfem::Operator
class CeedDiffusionOperator : public mfem::Operator {
 protected:
  const mfem::FiniteElementSpace *fes;
  CeedOperator build_oper, oper;
  CeedBasis basis, mesh_basis;
  CeedElemRestriction restr, mesh_restr, restr_i, mesh_restr_i;
  CeedQFunction apply_qfunc, build_qfunc;
  CeedVector node_coords, qdata;

  BuildContext build_ctx;

  CeedVector u, v;

  static void FESpace2Ceed(const mfem::FiniteElementSpace *fes,
                           const mfem::IntegrationRule &ir,
                           Ceed ceed, CeedBasis *basis,
                           CeedElemRestriction *restr) {
    mfem::Mesh *mesh = fes->GetMesh();
    const mfem::FiniteElement *fe = fes->GetFE(0);
    const int order = fes->GetOrder(0);
    mfem::Array<int> dof_map;
    switch (mesh->Dimension()) {
    case 1: {
      const mfem::H1_SegmentElement *h1_fe =
        dynamic_cast<const mfem::H1_SegmentElement *>(fe);
      MFEM_VERIFY(h1_fe, "invalid FE");
      h1_fe->GetDofMap().Copy(dof_map);
      break;
    }
    case 2: {
      const mfem::H1_QuadrilateralElement *h1_fe =
        dynamic_cast<const mfem::H1_QuadrilateralElement *>(fe);
      MFEM_VERIFY(h1_fe, "invalid FE");
      h1_fe->GetDofMap().Copy(dof_map);
      break;
    }
    case 3: {
      const mfem::H1_HexahedronElement *h1_fe =
        dynamic_cast<const mfem::H1_HexahedronElement *>(fe);
      MFEM_VERIFY(h1_fe, "invalid FE");
      h1_fe->GetDofMap().Copy(dof_map);
      break;
    }
    }
    const mfem::FiniteElement *fe1d =
      fes->FEColl()->FiniteElementForGeometry(mfem::Geometry::SEGMENT);
    mfem::DenseMatrix shape1d(fe1d->GetDof(), ir.GetNPoints());
    mfem::DenseMatrix grad1d(fe1d->GetDof(), ir.GetNPoints());
    mfem::Vector qref1d(ir.GetNPoints()), qweight1d(ir.GetNPoints());
    mfem::Vector shape_i(shape1d.Height());
    mfem::DenseMatrix grad_i(grad1d.Height(), 1);
    const mfem::H1_SegmentElement *h1_fe1d =
      dynamic_cast<const mfem::H1_SegmentElement *>(fe1d);
    MFEM_VERIFY(h1_fe1d, "invalid FE");
    const mfem::Array<int> &dof_map_1d = h1_fe1d->GetDofMap();
    for (int i = 0; i < ir.GetNPoints(); i++) {
      const mfem::IntegrationPoint &ip = ir.IntPoint(i);
      qref1d(i) = ip.x;
      qweight1d(i) = ip.weight;
      fe1d->CalcShape(ip, shape_i);
      fe1d->CalcDShape(ip, grad_i);
      for (int j = 0; j < shape1d.Height(); j++) {
        shape1d(j,i) = shape_i(dof_map_1d[j]);
        grad1d(j,i) = grad_i(dof_map_1d[j],0);
      }
    }
    CeedBasisCreateTensorH1(ceed, mesh->Dimension(), fes->GetVDim(), order+1,
                            ir.GetNPoints(), shape1d.GetData(),
                            grad1d.GetData(), qref1d.GetData(),
                            qweight1d.GetData(), basis);

    const mfem::Table &el_dof = fes->GetElementToDofTable();
    mfem::Array<int> tp_el_dof(el_dof.Size_of_connections());
    for (int i = 0; i < mesh->GetNE(); i++) {
      const int el_offset = fe->GetDof()*i;
      for (int j = 0; j < fe->GetDof(); j++) {
        tp_el_dof[j + el_offset] = el_dof.GetJ()[dof_map[j] + el_offset];
      }
    }
    CeedElemRestrictionCreate(ceed, mesh->GetNE(), fe->GetDof(),
                              fes->GetNDofs(), fes->GetVDim(), CEED_MEM_HOST,
                              CEED_COPY_VALUES, tp_el_dof.GetData(), restr);
  }

 public:
  /// Constructor. Assumes @a fes is a scalar FE space.
  CeedDiffusionOperator(Ceed ceed, const mfem::FiniteElementSpace *fes)
    : Operator(fes->GetNDofs()),
      fes(fes) {
    mfem::Mesh *mesh = fes->GetMesh();
    const int order = fes->GetOrder(0);
    const int ir_order = 2*(order + 2) - 1; // <-----
    const mfem::IntegrationRule &ir =
      mfem::IntRules.Get(mfem::Geometry::SEGMENT, ir_order);
    CeedInt nelem = mesh->GetNE(), dim = mesh->SpaceDimension(),
            ncompx = dim, nqpts;

    FESpace2Ceed(fes, ir, ceed, &basis, &restr);

    const mfem::FiniteElementSpace *mesh_fes = mesh->GetNodalFESpace();
    MFEM_VERIFY(mesh_fes, "the Mesh has no nodal FE space");
    FESpace2Ceed(mesh_fes, ir, ceed, &mesh_basis, &mesh_restr);
    CeedBasisGetNumQuadraturePoints(basis, &nqpts);

    CeedElemRestrictionCreateIdentity(ceed, nelem, nqpts,
                                      nqpts*nelem, dim*(dim+1)/2, &restr_i);
    CeedElemRestrictionCreateIdentity(ceed, nelem, nqpts,
                                      nqpts*nelem, 1, &mesh_restr_i);

    CeedVectorCreate(ceed, mesh->GetNodes()->Size(), &node_coords);
    CeedVectorSetArray(node_coords, CEED_MEM_HOST, CEED_USE_POINTER,
                       mesh->GetNodes()->GetData());

    CeedVectorCreate(ceed, nelem*nqpts*dim*(dim+1)/2, &qdata);

    // Context data to be passed to the 'f_build_diff' Q-function.
    build_ctx.dim = mesh->Dimension();
    build_ctx.space_dim = mesh->SpaceDimension();

    // Create the Q-function that builds the diff operator (i.e. computes its
    // quadrature data) and set its context data.
    CeedQFunctionCreateInterior(ceed, 1, f_build_diff,
                                f_build_diff_loc, &build_qfunc);
    CeedQFunctionAddInput(build_qfunc, "dx", ncompx*dim, CEED_EVAL_GRAD);
    CeedQFunctionAddInput(build_qfunc, "weights", 1, CEED_EVAL_WEIGHT);
    CeedQFunctionAddOutput(build_qfunc, "qdata", dim*(dim+1)/2, CEED_EVAL_NONE);
    CeedQFunctionSetContext(build_qfunc, &build_ctx, sizeof(build_ctx));

    // Create the operator that builds the quadrature data for the diff operator.
    CeedOperatorCreate(ceed, build_qfunc, CEED_QFUNCTION_NONE,
                       CEED_QFUNCTION_NONE, &build_oper);
    CeedOperatorSetField(build_oper, "dx", mesh_restr, CEED_NOTRANSPOSE,
                         mesh_basis, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(build_oper, "weights", mesh_restr_i, CEED_NOTRANSPOSE,
                         mesh_basis, CEED_VECTOR_NONE);
    CeedOperatorSetField(build_oper, "qdata", restr_i, CEED_NOTRANSPOSE,
                         CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

    // Compute the quadrature data for the diff operator.
    CeedOperatorApply(build_oper, node_coords, qdata,
                      CEED_REQUEST_IMMEDIATE);

    // Create the Q-function that defines the action of the diff operator.
    CeedQFunctionCreateInterior(ceed, 1, f_apply_diff,
                                f_apply_diff_loc, &apply_qfunc);
    CeedQFunctionAddInput(apply_qfunc, "u", dim, CEED_EVAL_GRAD);
    CeedQFunctionAddInput(apply_qfunc, "qdata", dim*(dim+1)/2, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(apply_qfunc, "v", dim, CEED_EVAL_GRAD);
    CeedQFunctionSetContext(apply_qfunc, &build_ctx, sizeof(build_ctx));

    // Create the diff operator.
    CeedOperatorCreate(ceed, apply_qfunc, CEED_QFUNCTION_NONE,
                       CEED_QFUNCTION_NONE, &oper);
    CeedOperatorSetField(oper, "u", restr, CEED_NOTRANSPOSE,
                         basis, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(oper, "qdata", restr_i, CEED_NOTRANSPOSE,
                         CEED_BASIS_COLLOCATED, qdata);
    CeedOperatorSetField(oper, "v", restr, CEED_NOTRANSPOSE,
                         basis, CEED_VECTOR_ACTIVE);

    CeedVectorCreate(ceed, fes->GetNDofs(), &u);
    CeedVectorCreate(ceed, fes->GetNDofs(), &v);
  }

  /// Destructor
  ~CeedDiffusionOperator() {
    CeedVectorDestroy(&u);
    CeedVectorDestroy(&v);
    CeedVectorDestroy(&qdata);
    CeedVectorDestroy(&node_coords);
    CeedElemRestrictionDestroy(&restr);
    CeedElemRestrictionDestroy(&mesh_restr);
    CeedElemRestrictionDestroy(&restr_i);
    CeedElemRestrictionDestroy(&mesh_restr_i);
    CeedBasisDestroy(&basis);
    CeedBasisDestroy(&mesh_basis);
    CeedQFunctionDestroy(&build_qfunc);
    CeedOperatorDestroy(&build_oper);
    CeedQFunctionDestroy(&apply_qfunc);
    CeedOperatorDestroy(&oper);
  }

  /// Operator action
  virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const {
    CeedVectorSetArray(u, CEED_MEM_HOST, CEED_USE_POINTER, x.GetData());
    CeedVectorSetArray(v, CEED_MEM_HOST, CEED_USE_POINTER, y.GetData());

    CeedOperatorApply(oper, u, v, CEED_REQUEST_IMMEDIATE);
    CeedVectorSyncArray(v, CEED_MEM_HOST);
  }
};
