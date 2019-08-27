//                         libCEED + MFEM Example: BP3
//
// This example illustrates a simple usage of libCEED with the MFEM (mfem.org)
// finite element library.
//
// The example reads a mesh from a file and solves a linear system with a
// diffusion stiffness matrix (with a prescribed analytic solution, provided by
// the function 'solution'). The diffusion matrix is expressed as a new class,
// CeedDiffusionOperator, derived from mfem::Operator. Internally,
// CeedDiffusionOperator uses a CeedOperator object constructed based on an
// mfem::FiniteElementSpace. All libCEED objects use a Ceed logical device
// object constructed based on a command line argument. (-ceed).
//
// The linear system is inverted using the conjugate gradients algorithm
// corresponding to CEED BP3, see http://ceed.exascaleproject.org/bps. Arbitrary
// mesh and solution orders in 1D, 2D and 3D are supported from the same code.
//
// Build with:
//
//     make bp3 [MFEM_DIR=</path/to/mfem>] [CEED_DIR=</path/to/libceed>]
//
// Sample runs:
//
//     bp3
//     bp3 -ceed /cpu/self
//     bp3 -m ../../../mfem/data/fichera.mesh -o 4
//     bp3 -m ../../../mfem/data/square-disc-nurbs.mesh -o 6
//     bp3 -m ../../../mfem/data/inline-segment.mesh -o 8

/// @file
/// MFEM diffusion operator based on libCEED

#include <ceed.h>
#include <mfem.hpp>
#include "bp3.hpp"

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

//TESTARGS -ceed {ceed_resource} -t -no-vis --size 2000
int main(int argc, char *argv[]) {
  // 1. Parse command-line options.
  const char *ceed_spec = "/cpu/self";
  #ifndef MFEM_DIR
  const char *mesh_file = "../../../mfem/data/star.mesh";
  #else
  const char *mesh_file = MFEM_DIR "/data/star.mesh";
  #endif
  int order = 2;
  bool visualization = true;
  bool test = false;
  double max_nnodes = 50000;

  mfem::OptionsParser args(argc, argv);
  args.AddOption(&ceed_spec, "-c", "-ceed", "Ceed specification.");
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree).");
  args.AddOption(&max_nnodes, "-s", "--size", "Maximum size (number of DoFs)");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Enable or disable GLVis visualization.");
  args.AddOption(&test, "-t", "--test", "-no-test",
                 "--no-test",
                 "Enable or disable test mode.");
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(std::cout);
    return 1;
  }
  if (!test) {
    args.PrintOptions(std::cout);
  }

  // 2. Initialize a Ceed device object using the given Ceed specification.
  Ceed ceed;
  CeedInit(ceed_spec, &ceed);

  // 3. Read the mesh from the given mesh file.
  mfem::Mesh *mesh = new mfem::Mesh(mesh_file, 1, 1);
  int dim = mesh->Dimension();

  // 4. Refine the mesh to increase the resolution. In this example we do
  //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
  //    largest number that gives a final system with no more than 50,000
  //    unknowns, approximately.
  {
    int ref_levels =
      (int)floor((log(max_nnodes/mesh->GetNE())-dim*log(order))/log(2.)/dim);
    for (int l = 0; l < ref_levels; l++) {
      mesh->UniformRefinement();
    }
  }
  if (mesh->GetNodalFESpace() == NULL) {
    mesh->SetCurvature(1, false, -1, mfem::Ordering::byNODES);
  }
  if (mesh->NURBSext) {
    mesh->SetCurvature(order, false, -1, mfem::Ordering::byNODES);
  }

  // 5. Define a finite element space on the mesh. Here we use continuous
  //    Lagrange finite elements of the specified order.
  MFEM_VERIFY(order > 0, "invalid order");
  mfem::FiniteElementCollection *fec = new mfem::H1_FECollection(order, dim);
  mfem::FiniteElementSpace *fespace = new mfem::FiniteElementSpace(mesh, fec);
  if (!test) {
    std::cout << "Number of finite element unknowns: "
              << fespace->GetTrueVSize() << std::endl;
  }

  mfem::FunctionCoefficient sol_coeff(solution);
  mfem::Array<int> ess_tdof_list;
  mfem::GridFunction sol(fespace);
  if (mesh->bdr_attributes.Size()) {
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

  // 7. Construct a CeedDiffusionOperator utilizing the 'ceed' device and using
  //    the 'fespace' object to extract data needed by the Ceed objects.
  CeedDiffusionOperator diff(ceed, fespace);

  mfem::Operator *D;
  mfem::Vector X, B;
  diff.FormLinearSystem(ess_tdof_list, sol, b, D, X, B);

  // 8. Solve the discrete system using the conjugate gradients (CG) method.
  mfem::CGSolver cg;
  cg.SetRelTol(1e-6);
  cg.SetMaxIter(1000);
  if (test) {
    cg.SetPrintLevel(0);
  } else {
    cg.SetPrintLevel(3);
  }
  cg.SetOperator(*D);

  cg.Mult(B, X);

  // 9. Compute and print the L2 norm of the error.
  if (!test) {
    std::cout << "L2 projection error: " << sol.ComputeL2Error(sol_coeff)
              << std::endl;
  } else {
    if (fabs(sol.ComputeL2Error(sol_coeff))>2e-3) {
      std::cout << "Error too large" << std::endl;
    }
  }

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
