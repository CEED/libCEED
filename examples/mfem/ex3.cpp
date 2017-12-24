//                           libCEED + MFEM Example 1
//
// This example illustrates a simple usage of libCEED with the MFEM (mfem.org)
// finite element library.
//
// The example reads a mesh from a file and solves a simple linear system with a
// diff matrix (L2-projection of a given analytic function provided by
// 'solution'). The diff matrix required for performing the projection is
// expressed as a new class, CeedDiffusionOperator, derived from mfem::Operator.
// Internally, CeedDiffusionOperator uses a CeedOperator object constructed based on
// an mfem::FiniteElementSpace. All libCEED objects use a Ceed device object
// constructed based on a command line argument.
//
// The diff matrix is inverted using a simple conjugate gradient algorithm
// corresponding to CEED BP1, see http://ceed.exascaleproject.org/bps. Arbitrary
// mesh and solution orders in 1D, 2D and 3D are supported from the same code.
//
// Build with:
//
//     make ex1 [MFEM_DIR=</path/to/mfem>]
//
// Sample runs:
//
//     ex1
//     ex1 -m ../../../mfem/data/fichera.mesh
//     ex1 -m ../../../mfem/data/star.vtk  -o 3
//     ex1 -m ../../../mfem/data/inline-segment.mesh -o 8

#include <ceed.h>
#include <mfem.hpp>

/// Exact solution
double solution(const mfem::Vector &pt) {
  static const double x[3] = { -0.32, 0.15, 0.24 };
  static const double k[3] = { 1.21, 1.45, 1.37 };
  double val = sin(M_PI*(x[0]+k[0]*pt(0)));
  for (int d = 1; d < pt.Size(); d++)
    val *= sin(M_PI*(x[d]+k[d]*pt(d)));
  return val;
}

/// Right-hand side
double rhs(const mfem::Vector &pt) {
  static const double x[3] = { -0.32, 0.15, 0.24 };
  static const double k[3] = { 1.21, 1.45, 1.37 };
  double f[3], l[3], val, lap;
  f[0] = sin(M_PI*(x[0]+k[0]*pt(0)));
  l[0] = M_PI*M_PI*k[0]*k[0]*f[0];
  val = f[0];
  lap = l[0];
  for (int d = 1; d < pt.Size(); d++) {
    f[d] = sin(M_PI*(x[d]+k[d]*pt(d)));
    l[d] = M_PI*M_PI*k[d]*k[d]*f[d];
    lap = lap*f[d] + val*l[d];
    val = val*f[d];
  }
  return lap;
}

/// A structure used to pass additional data to f_build_diff and f_apply_diff
struct DiffContext { CeedInt dim, space_dim; };

/// libCEED Q-function for building quadrature data for a diffusion operator
static int f_build_diff(void *ctx, void *qdata, CeedInt Q,
                        const CeedScalar *const *u, CeedScalar *const *v) {
  // u[1] is Jacobians, size (Q x nc x dim) with column-major layout
  // u[4] is quadrature weights, size (Q)
  //
  // At every quadrature point, compute qw/det(J).adj(J).adj(J)^T and store
  // the symmetric part of the result.
  DiffContext *dc = (DiffContext*)ctx;
  CeedScalar *qd = (CeedScalar*)qdata;
  const CeedScalar *J = u[1], *qw = u[4];
  switch (dc->dim + 10*dc->space_dim) {
  case 11:
    for (CeedInt i=0; i<Q; i++) {
      qd[i] = qw[i] / J[i];
    }
    break;
  case 22:
    for (CeedInt i=0; i<Q; i++) {
      // J: 0 2   qd: 0 1   adj(J):  J22 -J12
      //    1 3       1 2           -J21  J11
      const CeedScalar J11 = J[i+Q*0];
      const CeedScalar J21 = J[i+Q*1];
      const CeedScalar J12 = J[i+Q*2];
      const CeedScalar J22 = J[i+Q*3];
      const CeedScalar w = qw[i] / (J11*J22 - J21*J12);
      qd[i+Q*0] =   w * (J12*J12 + J22*J22);
      qd[i+Q*1] = - w * (J11*J12 + J21*J22);
      qd[i+Q*2] =   w * (J11*J11 + J21*J21);
    }
    break;
  case 33:
    for (CeedInt i=0; i<Q; i++) {
      // J: 0 3 6   qd: 0 1 2
      //    1 4 7       1 3 4
      //    2 5 8       2 4 5
      const CeedScalar J11 = J[i+Q*0];
      const CeedScalar J21 = J[i+Q*1];
      const CeedScalar J31 = J[i+Q*2];
      const CeedScalar J12 = J[i+Q*3];
      const CeedScalar J22 = J[i+Q*4];
      const CeedScalar J32 = J[i+Q*5];
      const CeedScalar J13 = J[i+Q*6];
      const CeedScalar J23 = J[i+Q*7];
      const CeedScalar J33 = J[i+Q*8];
      const CeedScalar A11 = J22*J33 - J23*J32;
      const CeedScalar A12 = J13*J32 - J12*J33;
      const CeedScalar A13 = J12*J23 - J13*J22;
      const CeedScalar A21 = J23*J31 - J21*J33;
      const CeedScalar A22 = J11*J33 - J13*J31;
      const CeedScalar A23 = J13*J21 - J11*J23;
      const CeedScalar A31 = J21*J32 - J22*J31;
      const CeedScalar A32 = J12*J31 - J11*J32;
      const CeedScalar A33 = J11*J22 - J12*J21;
      const CeedScalar w = qw[i] / (J11*A11 + J21*A12 + J31*A13);
      qd[i+Q*0] = w * (A11*A11 + A12*A12 + A13*A13);
      qd[i+Q*1] = w * (A11*A21 + A12*A22 + A13*A23);
      qd[i+Q*2] = w * (A11*A31 + A12*A32 + A13*A33);
      qd[i+Q*3] = w * (A21*A21 + A22*A22 + A23*A23);
      qd[i+Q*4] = w * (A21*A31 + A22*A32 + A23*A33);
      qd[i+Q*5] = w * (A31*A31 + A32*A32 + A33*A33);
    }
    break;
  default:
    return CeedError(NULL, 1, "dim=%d, space_dim=%d is not supported",
                     dc->dim, dc->space_dim);
  }
  return 0;
}

/// libCEED Q-function for applying a diff operator
static int f_apply_diff(void *ctx, void *qdata, CeedInt Q,
                        const CeedScalar *const *u, CeedScalar *const *v) {
  DiffContext *dc = (DiffContext*)ctx;
  const CeedScalar *qd = (const CeedScalar*)qdata;
  // u[1], v[1]: size: (Q x nc x dim) with column-major layout (nc == 1)
  const CeedScalar *ug = u[1];
  CeedScalar *vg = v[1];
  switch (dc->dim) {
  case 1:
    for (CeedInt i=0; i<Q; i++) {
      vg[i] = ug[i] * qd[i];
    }
    break;
  case 2:
    for (CeedInt i=0; i<Q; i++) {
      const CeedScalar ug0 = ug[i+Q*0];
      const CeedScalar ug1 = ug[i+Q*1];
      vg[i+Q*0] = qd[i+Q*0]*ug0 + qd[i+Q*1]*ug1;
      vg[i+Q*1] = qd[i+Q*1]*ug0 + qd[i+Q*2]*ug1;
    }
    break;
  case 3:
    for (CeedInt i=0; i<Q; i++) {
      const CeedScalar ug0 = ug[i+Q*0];
      const CeedScalar ug1 = ug[i+Q*1];
      const CeedScalar ug2 = ug[i+Q*2];
      vg[i+Q*0] = qd[i+Q*0]*ug0 + qd[i+Q*1]*ug1 + qd[i+Q*2]*ug2;
      vg[i+Q*1] = qd[i+Q*1]*ug0 + qd[i+Q*3]*ug1 + qd[i+Q*4]*ug2;
      vg[i+Q*2] = qd[i+Q*2]*ug0 + qd[i+Q*4]*ug1 + qd[i+Q*5]*ug2;
    }
    break;
  default:
    return CeedError(NULL, 1, "topo_dim=%d is not supported", dc->dim);
  }
  return 0;
}

/// Wrapper for a diffusion CeedOperator as an mfem::Operator
class CeedDiffusionOperator : public mfem::Operator {
 protected:
  const mfem::FiniteElementSpace *fes;
  CeedOperator build_oper, oper;
  CeedBasis basis, mesh_basis;
  CeedElemRestriction restr, mesh_restr;
  CeedQFunction apply_qfunc, build_qfunc;
  CeedVector node_coords, qdata;

  DiffContext diff_ctx;

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
                              fes->GetNDofs(), CEED_MEM_HOST, CEED_COPY_VALUES,
                              tp_el_dof.GetData(), restr);
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

    FESpace2Ceed(fes, ir, ceed, &basis, &restr);

    const mfem::FiniteElementSpace *mesh_fes = mesh->GetNodalFESpace();
    MFEM_VERIFY(mesh_fes, "the Mesh has no nodal FE space");
    FESpace2Ceed(mesh_fes, ir, ceed, &mesh_basis, &mesh_restr);

    CeedVectorCreate(ceed, mesh->GetNodes()->Size(), &node_coords);
    CeedVectorSetArray(node_coords, CEED_MEM_HOST, CEED_USE_POINTER,
                       mesh->GetNodes()->GetData());

    const int dim = mesh->Dimension();
    diff_ctx.dim = dim;
    diff_ctx.space_dim = mesh->SpaceDimension();

    const int qsize = dim*(dim+1)/2;
    CeedQFunctionCreateInterior(ceed, 1, 1, qsize*sizeof(CeedScalar),
                                (CeedEvalMode)(CEED_EVAL_GRAD|CEED_EVAL_WEIGHT),
                                CEED_EVAL_NONE, f_build_diff,
                                __FILE__":f_build_diff", &build_qfunc);
    CeedQFunctionSetContext(build_qfunc, &diff_ctx, sizeof(diff_ctx));
    CeedOperatorCreate(ceed, mesh_restr, mesh_basis, build_qfunc, NULL, NULL,
                       &build_oper);
    CeedOperatorGetQData(build_oper, &qdata);
    CeedOperatorApply(build_oper, qdata, node_coords, NULL,
                      CEED_REQUEST_IMMEDIATE);

    CeedQFunctionCreateInterior(ceed, 1, 1, qsize*sizeof(CeedScalar),
                                CEED_EVAL_GRAD, CEED_EVAL_GRAD, f_apply_diff,
                                __FILE__":f_apply_diff", &apply_qfunc);
    CeedQFunctionSetContext(apply_qfunc, &diff_ctx, sizeof(diff_ctx));
    CeedOperatorCreate(ceed, restr, basis, apply_qfunc, NULL, NULL, &oper);

    CeedVectorCreate(ceed, fes->GetNDofs(), &u);
    CeedVectorCreate(ceed, fes->GetNDofs(), &v);
  }

  /// Destructor
  ~CeedDiffusionOperator() {
    CeedVectorDestroy(&v);
    CeedVectorDestroy(&u);
    CeedOperatorDestroy(&oper);
    CeedQFunctionDestroy(&apply_qfunc);
    // qdata is owned by build_oper
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
  if (!args.Good()) {
    args.PrintUsage(std::cout);
    return 1;
  }
  args.PrintOptions(std::cout);

  // 2. Initialize a Ceed device object using the given Ceed specification.
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
    for (int l = 0; l < ref_levels; l++) {
      mesh->UniformRefinement();
    }
  }
  if (mesh->GetNodalFESpace() == NULL) {
    mesh->SetCurvature(1, false, -1, mfem::Ordering::byNODES);
  }

  // 5. Define a finite element space on the mesh. Here we use continuous
  //    Lagrange finite elements of the specified order.
  MFEM_VERIFY(order > 0, "invalid order");
  mfem::FiniteElementCollection *fec = new mfem::H1_FECollection(order, dim);
  mfem::FiniteElementSpace *fespace = new mfem::FiniteElementSpace(mesh, fec);
  std::cout << "Number of finite element unknowns: "
            << fespace->GetTrueVSize() << std::endl;

  mfem::FunctionCoefficient sol_coeff(solution);
  mfem::Array<int> ess_tdof_list;
  mfem::GridFunction sol(fespace);
  if (mesh->bdr_attributes.Size())
  {
     mfem::Array<int> ess_bdr(mesh->bdr_attributes.Max());
     ess_bdr = 1;
     fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
     sol.ProjectBdrCoefficient(sol_coeff, ess_bdr);
  }

  // 6. Construct a rhs vector using the linear form f(v) = (rhs, v), where
  //    v is a test function.
  mfem::LinearForm b(fespace);
  mfem::FunctionCoefficient rhs_coeff(rhs);
  b.AddDomainIntegrator(new mfem::DomainLFIntegrator(rhs_coeff));
  b.Assemble();

  // 7. Construct a CeedDiffusionOperator utilizing the 'ceed' device and using the
  //    'fespace' object to extract data needed by the Ceed objects.
  CeedDiffusionOperator diff(ceed, fespace);

  mfem::Operator *D;
  mfem::Vector X, B;
  diff.FormLinearSystem(ess_tdof_list, sol, b, D, X, B);

  // 8. Solve the discrete system using the conjugate gradients (CG) method.
  mfem::CGSolver cg;
  cg.SetRelTol(1e-6);
  cg.SetMaxIter(1000);
  cg.SetPrintLevel(3);
  cg.SetOperator(*D);

  cg.Mult(B, X);

  // 9. Compute and print the L2 projection error.
  std::cout << "L2 norm of the error: " << sol.ComputeL2Error(sol_coeff)
            << std::endl;

  // 10. Open a socket connection to GLVis and send the mesh and solution for
  //     visualization.
  if (visualization) {
    char vishost[] = "localhost";
    int  visport   = 19916;
    mfem::socketstream sol_sock(vishost, visport);
    sol_sock.precision(8);
    sol_sock << "solution\n" << *mesh << sol << std::flush;
  }

  // 11. Free memory and exit.
  delete fespace;
  delete fec;
  delete mesh;
  CeedDestroy(&ceed);
  return 0;
}
