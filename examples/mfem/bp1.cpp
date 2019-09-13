//                         libCEED + MFEM Example: BP1
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
// constructed based on a command line argument (-ceed).
//
// The mass matrix is inverted using a simple conjugate gradient algorithm
// corresponding to CEED BP1, see http://ceed.exascaleproject.org/bps. Arbitrary
// mesh and solution orders in 1D, 2D and 3D are supported from the same code.
//
// Build with:
//
//     make bp1 [MFEM_DIR=</path/to/mfem>] [CEED_DIR=</path/to/libceed>]
//
// Sample runs:
//
//     bp1
//     bp1 -ceed /cpu/self
//     bp1 -ceed /gpu/occa
//     bp1 -ceed /cpu/occa
//     bp1 -ceed /omp/occa
//     bp1 -ceed /ocl/occa
//     bp1 -m ../../../mfem/data/fichera.mesh
//     bp1 -m ../../../mfem/data/star.vtk -o 3
//     bp1 -m ../../../mfem/data/inline-segment.mesh -o 8

/// @file
/// MFEM mass operator based on libCEED

#include <ceed.h>
#include <mfem.hpp>
#include "bp1.hpp"

/// Continuous function to project on the discrete FE space
double solution(const mfem::Vector &pt) {
  return pt.Norml2(); // distance to the origin
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
  int order = 1;
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
  if (test) {
    cg.SetPrintLevel(0);
  } else {
    cg.SetPrintLevel(3);
  }
  cg.SetOperator(mass);

  mfem::GridFunction sol(fespace);
  sol = 0.0;
  cg.Mult(b, sol);

  // 9. Compute and print the L2 projection error.
  if (!test) {
    std::cout << "L2 projection error: " << sol.ComputeL2Error(sol_coeff)
              << std::endl;
  } else {
    if (fabs(sol.ComputeL2Error(sol_coeff))>2e-4) {
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
