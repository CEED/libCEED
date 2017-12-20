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

#include <ceed.h>
#include <mfem.hpp>

class CeedMassOperator : public mfem::Operator {
 protected:
  const mfem::FiniteElementSpace *fes;
  CeedOperator build_oper, oper;
  CeedBasis basis, mesh_basis;
  CeedElemRestriction restr, mesh_restr;
  CeedQFunction apply_qfunc, build_qfunc;
  CeedVector qdata;

  CeedVector u, v;

  static void FESpace2Ceed(const mfem::FiniteElementSpace *fes,
                           const mfem::IntegrationRule &ir,
                           Ceed ceed, CeedBasis *basis,
                           CeedElemRestriction *restr)
  {
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
    mfem::Vector shape_i;
    mfem::DenseMatrix grad_i;
    for (int i = 0; i < ir.GetNPoints(); i++) {
      const mfem::IntegrationPoint &ip = ir.IntPoint(i);
      qref1d(i) = ip.x;
      qweight1d(i) = ip.weight;
      shape1d.GetColumnReference(i, shape_i);
      grad_i.UseExternalData(&grad1d(0,i), grad1d.Height(), 1);
      fe1d->CalcShape(ip, shape_i);
      fe1d->CalcDShape(ip, grad_i);
    }
    shape1d.Transpose();
    grad1d.Transpose();
    CeedBasisCreateTensorH1(ceed, mesh->Dimension(), 1, order+1,
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
                              fes->GetNDofs(), CEED_MEM_HOST, CEED_COPY_VALUES,
                              tp_el_dof.GetData(), restr);
  }

 public:
  /// Assuming a scalar FE space.
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

    build_qfunc = NULL; // TODO
    CeedOperatorCreate(ceed, mesh_restr, mesh_basis, build_qfunc, NULL, NULL,
                       &build_oper);
    CeedOperatorGetQData(build_oper, &qdata);

    apply_qfunc = NULL; // TODO
    CeedOperatorCreate(ceed, restr, basis, apply_qfunc, NULL, NULL, &oper);

    CeedVectorCreate(ceed, fes->GetNDofs(), &u);
    CeedVectorCreate(ceed, fes->GetNDofs(), &v);
  }

  ~CeedMassOperator() {
    CeedVectorDestroy(&v);
    CeedVectorDestroy(&u);
    CeedOperatorDestroy(&oper);
    // CeedVectorDestroy(&qdata); // qdata is owned by build_oper
    CeedOperatorDestroy(&build_oper);
    CeedElemRestrictionDestroy(&mesh_restr);
    CeedBasisDestroy(&mesh_basis);
    CeedElemRestrictionDestroy(&restr);
    CeedBasisDestroy(&basis);
  }

  virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const {
    CeedVectorSetArray(u, CEED_MEM_HOST, CEED_USE_POINTER, x.GetData());
    CeedVectorSetArray(v, CEED_MEM_HOST, CEED_USE_POINTER, y.GetData());

    CeedOperatorApply(oper, qdata, u, v, CEED_REQUEST_IMMEDIATE);
  }
};

int main(int argc, char *argv[]) {
  // 1. Parse command-line options.
  const char *ceed_spec = "/cpu/self";
  const char *mesh_file = "../../../mfem/data/star.mesh";
  int order = 1;
  bool visualization = true;

  mfem::OptionsParser args(argc, argv);
  args.AddOption(&ceed_spec, "-c", "--ceed-spec", "Ceed specification.");
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree).");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Enable or disable GLVis visualization.");
  args.Parse();
  if (!args.Good())
  {
     args.PrintUsage(mfem::out);
     return 1;
  }
  args.PrintOptions(mfem::out);

  // 2. Initialize a Ceed object using the given Ceed specification.
  Ceed ceed;
  CeedInit(ceed_spec, &ceed);

  // 3. Read the mesh from the given mesh file.
  mfem::Mesh *mesh = new mfem::Mesh(mesh_file, 1, 1);
  int dim = mesh->Dimension();

  // 4. Refine the mesh to increase the resolution. In this example we do
  //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
  //    largest number that gives a final mesh with no more than 50,000
  //    elements.
  {
     int ref_levels =
        (int)floor(log(50000./mesh->GetNE())/log(2.)/dim);
     for (int l = 0; l < ref_levels; l++)
     {
        mesh->UniformRefinement();
     }
  }
  if (mesh->GetNodalFESpace() == NULL) { mesh->SetCurvature(1); }

  // 5. Define a finite element space on the mesh. Here we use continuous
  //    Lagrange finite elements of the specified order.
  MFEM_VERIFY(order > 0, "invalid order");
  mfem::FiniteElementCollection *fec = new mfem::H1_FECollection(order, dim);
  mfem::FiniteElementSpace *fespace = new mfem::FiniteElementSpace(mesh, fec);
  mfem::out << "Number of finite element unknowns: "
            << fespace->GetTrueVSize() << std::endl;

  CeedMassOperator mass(ceed, fespace);

  // TODO

  delete fespace;
  delete fec;
  delete mesh;
  CeedDestroy(&ceed);
  return 0;
}
