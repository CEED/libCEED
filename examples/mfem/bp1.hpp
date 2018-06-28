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
/// MFEM mass operator based on libCEED

#include <ceed.h>
#include <mfem.hpp>

/// A structure used to pass additional data to f_build_mass
struct BuildContext { CeedInt dim, space_dim; };

/// libCEED Q-function for building quadrature data for a mass operator
static int f_build_mass(void *ctx, CeedInt Q,
                        const CeedScalar *const *in, CeedScalar *const *out) {
  // in[0] is Jacobians, size (Q x nc x dim) with column-major layout
  // in[1] is quadrature weights, size (Q)
  BuildContext *bc = (BuildContext*)ctx;
  CeedScalar *v = (CeedScalar*)out[0];
  CeedScalar *qd = (CeedScalar*)in[2];
  CeedScalar *J = (CeedScalar*)in[0], *qw = (CeedScalar*)in[1];
  switch (bc->dim + 10*bc->space_dim) {
  case 11:
    for (CeedInt i=0; i<Q; i++) {
      qd[i] = J[i] * qw[i];
    }
    break;
  case 22:
    for (CeedInt i=0; i<Q; i++) {
      // 0 2
      // 1 3
      qd[i] = (J[i+Q*0]*J[i+Q*3] - J[i+Q*1]*J[i+Q*2]) * qw[i];
    }
    break;
  case 33:
    for (CeedInt i=0; i<Q; i++) {
      // 0 3 6
      // 1 4 7
      // 2 5 8
      qd[i] = (J[i+Q*0]*(J[i+Q*4]*J[i+Q*8] - J[i+Q*5]*J[i+Q*7]) -
               J[i+Q*1]*(J[i+Q*3]*J[i+Q*8] - J[i+Q*5]*J[i+Q*6]) +
               J[i+Q*2]*(J[i+Q*3]*J[i+Q*7] - J[i+Q*4]*J[i+Q*6])) * qw[i];
    }
    break;
  default:
    return CeedError(NULL, 1, "dim=%d, space_dim=%d is not supported",
                     bc->dim, bc->space_dim);
  }
  return 0;
}

/// libCEED Q-function for applying a mass operator
static int f_apply_mass(void *ctx, CeedInt Q,
                        const CeedScalar *const *in, CeedScalar *const *out) {
  CeedScalar *u = (CeedScalar*)in[0];
  CeedScalar *v = (CeedScalar*)out[0];
  CeedScalar *w = (CeedScalar*)in[1];
  for (CeedInt i=0; i<Q; i++) {
    v[i] = w[i] * u[i];
  }
  return 0;
}

/// Wrapper for a mass CeedOperator as an mfem::Operator
class CeedMassOperator : public mfem::Operator {
 protected:
  const mfem::FiniteElementSpace *fes;
  CeedOperator build_oper, oper;
  CeedBasis basis, mesh_basis;
  CeedElemRestriction restr, mesh_restr;
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
        dynamic_cast<const mfem::H1_SegmentElement*>(fe);
      MFEM_VERIFY(h1_fe, "invalid FE");
      h1_fe->GetDofMap().Copy(dof_map);
      break;
    }
    case 2: {
      const mfem::H1_QuadrilateralElement *h1_fe =
        dynamic_cast<const mfem::H1_QuadrilateralElement*>(fe);
      MFEM_VERIFY(h1_fe, "invalid FE");
      h1_fe->GetDofMap().Copy(dof_map);
      break;
    }
    case 3: {
      const mfem::H1_HexahedronElement *h1_fe =
        dynamic_cast<const mfem::H1_HexahedronElement*>(fe);
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
      dynamic_cast<const mfem::H1_SegmentElement*>(fe1d);
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
                              fes->GetNDofs(), 1, CEED_MEM_HOST, CEED_COPY_VALUES,
                              tp_el_dof.GetData(), restr);
  }

 public:
  /// Constructor. Assumes @a fes is a scalar FE space.
  CeedMassOperator(Ceed ceed, const mfem::FiniteElementSpace *fes)
    : Operator(fes->GetNDofs()),
      fes(fes) {
    mfem::Mesh *mesh = fes->GetMesh();
    const int order = fes->GetOrder(0);
    const int ir_order = 2*(order + 2) - 1; // <-----
    const mfem::IntegrationRule &ir =
      mfem::IntRules.Get(mfem::Geometry::SEGMENT, ir_order);

    FESpace2Ceed(fes, ir, ceed, &basis, &restr);

    const mfem::FiniteElementSpace *mesh_fes = mesh->GetNodalFESpace();
    MFEM_VERIFY(mesh_fes, "the Mesh has no nodal FE space");
    FESpace2Ceed(mesh_fes, ir, ceed, &mesh_basis, &mesh_restr);

    CeedVectorCreate(ceed, mesh->GetNodes()->Size(), &node_coords);
    CeedVectorSetArray(node_coords, CEED_MEM_HOST, CEED_USE_POINTER,
                       mesh->GetNodes()->GetData());

    build_ctx.dim = mesh->Dimension();
    build_ctx.space_dim = mesh->SpaceDimension();

    CeedQFunctionCreateInterior(ceed, 1, f_build_mass,
                                __FILE__":f_build_mass", &build_qfunc);
    CeedQFunctionAddInput(build_qfunc, "x", 1, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(build_qfunc, "weight", 1, CEED_EVAL_WEIGHT);
    CeedQFunctionAddOutput(build_qfunc, "qdata", 1, CEED_EVAL_NONE);

    CeedOperatorCreate(ceed, build_qfunc, NULL, NULL, &build_oper);
    CeedOperatorSetField(build_oper, "x", mesh_restr, mesh_basis,
                         CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(build_oper, "weight", CEED_RESTRICTION_IDENTITY,
                         mesh_basis, CEED_VECTOR_NONE);
    CeedOperatorSetField(build_oper, "qdata", CEED_RESTRICTION_IDENTITY,
                         CEED_BASIS_COLOCATED, CEED_VECTOR_ACTIVE);
    CeedOperatorApply(build_oper, node_coords, qdata,
                      CEED_REQUEST_IMMEDIATE);

    CeedQFunctionCreateInterior(ceed, 1, f_apply_mass,
                                __FILE__":f_apply_mass", &apply_qfunc);
    CeedQFunctionAddInput(apply_qfunc, "u", 1, CEED_EVAL_INTERP);
    CeedQFunctionAddInput(apply_qfunc, "qdata", 1, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(apply_qfunc, "v", 1, CEED_EVAL_INTERP);

    CeedOperatorCreate(ceed, apply_qfunc, NULL, NULL, &oper);
    CeedOperatorSetField(oper, "u", restr, basis, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(oper, "qdata", CEED_RESTRICTION_IDENTITY,
                          CEED_BASIS_COLOCATED, qdata);
    CeedOperatorSetField(oper, "v", restr, basis, CEED_VECTOR_ACTIVE);

    CeedVectorCreate(ceed, fes->GetNDofs(), &u);
    CeedVectorCreate(ceed, fes->GetNDofs(), &v);
  }

  /// Destructor
  ~CeedMassOperator() {
    CeedVectorDestroy(&v);
    CeedVectorDestroy(&u);
    CeedOperatorDestroy(&oper);
    CeedQFunctionDestroy(&apply_qfunc);
    CeedVectorDestroy(&qdata);
    CeedOperatorDestroy(&build_oper);
    CeedQFunctionDestroy(&build_qfunc);
    CeedVectorDestroy(&node_coords);
    CeedElemRestrictionDestroy(&mesh_restr);
    CeedBasisDestroy(&mesh_basis);
    CeedElemRestrictionDestroy(&restr);
    CeedBasisDestroy(&basis);
  }

  /// Operator action
  virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const {
    CeedVectorSetArray(u, CEED_MEM_HOST, CEED_USE_POINTER, x.GetData());
    CeedVectorSetArray(v, CEED_MEM_HOST, CEED_USE_POINTER, y.GetData());
    CeedOperatorApply(oper, u, v, CEED_REQUEST_IMMEDIATE);
  }
};
