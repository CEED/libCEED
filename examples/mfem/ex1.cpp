//                           libCEED + MFEM Example 1
//
// This example illustrates a simple usage of libCEED with the MFEM (mfem.org)
// finite element library.
//
// The example reads a mesh from a file and solves a simple linear system with a
// mass matrix (L2-projection of a given analytic function provided by
// 'solution'). The mass matrix required for performing the projection is
// expressed as a new class, CeedMassOperator, derived from mfem::Operator.
// Internally, CeedMassOperator uses a CeedOperator object constructed based on
// an mfem::FiniteElementSpace. All libCEED objects use a Ceed device object
// constructed based on a command line argument.
//
// The mass matrix is inverted using a simple conjugate gradient algorithm
// corresponding to CEED BP1, see http://ceed.exascaleproject.org/bps. Arbitrary
// mesh and solution orders in 1D, 2D and 3D are supported from the same code.
//
// Build with:

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

/// Continuous function to project on the discrete FE space
double solution(const mfem::Vector &pt) {
  return pt.Norml2(); // distance to the origin
}

/// A structure used to pass additional data to f_build_mass
struct BuildContext { CeedInt dim, space_dim; };

/// libCEED Q-function for building quadrature data for a mass operator
static int f_build_mass(void *ctx, void *qdata, CeedInt Q,
                        const CeedScalar *const *u, CeedScalar *const *v) {
  // u[1] is Jacobians, size (Q x nc x dim) with column-major layout
  // u[4] is quadrature weights, size (Q)
  BuildContext *bc = (BuildContext*)ctx;
  CeedScalar *qd = (CeedScalar*)qdata;
  const CeedScalar *J = u[1], *qw = u[4];
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
static int f_apply_mass(void *ctx, void *qdata, CeedInt Q,
                        const CeedScalar *const *u, CeedScalar *const *v) {
  const CeedScalar *w = (const CeedScalar*)qdata;
  for (CeedInt i=0; i<Q; i++) v[0][i] = w[i] * u[0][i];
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
                              fes->GetNDofs(), CEED_MEM_HOST, CEED_COPY_VALUES,
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

    CeedQFunctionCreateInterior(ceed, 1, 1, sizeof(CeedScalar),
                                (CeedEvalMode)(CEED_EVAL_GRAD|CEED_EVAL_WEIGHT),
                                CEED_EVAL_NONE, f_build_mass,
                                __FILE__":f_build_mass", &build_qfunc);
    CeedQFunctionSetContext(build_qfunc, &build_ctx, sizeof(build_ctx));
    CeedOperatorCreate(ceed, mesh_restr, mesh_basis, build_qfunc, NULL, NULL,
                       &build_oper);
    CeedOperatorGetQData(build_oper, &qdata);
    CeedOperatorApply(build_oper, qdata, node_coords, NULL,
                      CEED_REQUEST_IMMEDIATE);

    CeedQFunctionCreateInterior(ceed, 1, 1, sizeof(CeedScalar),
                                CEED_EVAL_INTERP, CEED_EVAL_INTERP, f_apply_mass,
                                __FILE__":f_apply_mass", &apply_qfunc);
    CeedOperatorCreate(ceed, restr, basis, apply_qfunc, NULL, NULL, &oper);

    CeedVectorCreate(ceed, fes->GetNDofs(), &u);
    CeedVectorCreate(ceed, fes->GetNDofs(), &v);
  }

  /// Destructor
  ~CeedMassOperator() {
    CeedVectorDestroy(&v);
    CeedVectorDestroy(&u);
    CeedOperatorDestroy(&oper);
    CeedQFunctionDestroy(&apply_qfunc);
    // CeedVectorDestroy(&qdata); // qdata is owned by build_oper
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

  // 6. Construct a rhs vector using the linear form f(v) = (solution, v), where
  //    v is a test function.
  mfem::LinearForm b(fespace);
  mfem::FunctionCoefficient sol_coeff(solution);
  b.AddDomainIntegrator(new mfem::DomainLFIntegrator(sol_coeff));
  b.Assemble();

  // 7. Construct a CeedMassOperator utilizing the 'ceed' device and using the
  //    'fespace' object to extract data needed by the Ceed objects.
  CeedMassOperator mass(ceed, fespace);

  // 8. Solve the discrete system using the conjugate gradients (CG) method.
  mfem::CGSolver cg;
  cg.SetRelTol(1e-6);
  cg.SetMaxIter(100);
  cg.SetPrintLevel(3);
  cg.SetOperator(mass);

  mfem::GridFunction sol(fespace);
  sol = 0.0;
  cg.Mult(b, sol);
  //std::cout << "sol="<<sol<< std::endl;
 
  // 9. Compute and print the L2 projection error.
  std::cout << "L2 projection error: " << sol.ComputeL2Error(sol_coeff)
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
