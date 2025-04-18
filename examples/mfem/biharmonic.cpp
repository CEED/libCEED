#include <mfem.hpp>
#include <mfem/libceed/mfem-ceed.hpp>
#include <ceed.h>
#include <fstream>
#include <iostream>

using namespace mfem;
using namespace std;

// CEED QFunction to apply Laplacian operator
extern "C" CeedInt LaplacianApply(CeedInt Q,
                                   const CeedScalar *const *in,
                                   CeedScalar *const *out,
                                   void *ctx) {
  const CeedScalar *grad_u = in[1];
  CeedScalar *v = out[0];

  const int dim = 2; // 2D example
  for (CeedInt i = 0; i < Q; i++) {
    v[i] = 0.0;
    for (int d = 0; d < dim; d++) {
      v[i] += grad_u[i * dim + d];
    }
  }
  return 0;
}

// CEED QFunction to build operator coefficients
extern "C" CeedInt LaplacianBuild(CeedInt Q,const CeedScalar *const *in,CeedScalar *const *out, void *ctx) {
                                     CeedScalar *qdata = out[0];
                                     for (CeedInt i = 0; i < Q; i++) {
                                      qdata[i] = 1.0; // constant coefficient
                                      }
                                  return 0;
}

int main(int argc, char *argv[]) {
  // 1. Initialize MFEM and CEED
   Device device("cpu");
   cout << "Running on device: " << device.GetDeviceMemoryType() << endl;

    Ceed ceed;
    CeedInit("/cpu/self", &ceed);

  // 2. Mesh
  int nx = 8, ny = 8, order = 2;
  Mesh mesh = Mesh::MakeCartesian2D(nx, ny, Element::QUADRILATERAL, true);
  mesh.EnsureNodes();
  int dim = mesh.Dimension();

  // 3. Finite element space
  H1_FECollection fec(order, dim);
  FiniteElementSpace fespace(&mesh, &fec);
  cout << "Number of unknowns: " << fespace.GetTrueVSize() << endl;

  // 4. Define GridFunctions
  GridFunction rhs(&fespace);
  GridFunction w(&fespace);
  GridFunction u(&fespace);

  rhs.ProjectCoefficient(ConstantCoefficient(1.0)); // f = 1

  // 5. Define CEED QFunctions
  CeedQFunction qf_build, qf_apply;

  CeedQFunctionCreateInterior(ceed, 1, LaplacianBuild, __FILE__ ":LaplacianBuild", &qf_build);
  CeedQFunctionAddInput(qf_build, "dx", CEED_EVAL_GRAD, dim);
  CeedQFunctionAddOutput(qf_build, "qdata", CEED_EVAL_NONE, 1);

  CeedQFunctionCreateInterior(ceed, 1, LaplacianApply, __FILE__ ":LaplacianApply", &qf_apply);
  CeedQFunctionAddInput(qf_apply, "u", CEED_EVAL_INTERP, 1);
  CeedQFunctionAddInput(qf_apply, "grad_u", CEED_EVAL_GRAD, dim);
  CeedQFunctionAddInput(qf_apply, "qdata", CEED_EVAL_NONE, 1);
  CeedQFunctionAddOutput(qf_apply, "v", CEED_EVAL_INTERP, 1);

  // 6. Create CEED Operator
  mfem::ceed::CeedOperatorBuilder builder(&fespace, qf_apply, qf_build, ceed);
  CeedOperator op_laplace = builder.GetCeedOperator();

  // 7. Solver
  mfem::ceed::CeedSolver solver(op_laplace);

  // 8. Solve -Δ w = f
  cout << "Solving -Δ w = f..." << endl;
  solver.Solve(rhs, w);

  // 9. Solve -Δ u = w
  cout << "Solving -Δ u = w..." << endl;
  solver.Solve(w, u);

  // 10. Output
  ofstream ufile("u.gf"), wfile("w.gf");
  u.Save(ufile);
  w.Save(wfile);

  cout << "Solutions saved to 'u.gf' and 'w.gf'" << endl;

  // Cleanup
  CeedDestroy(&ceed);
  return 0;
}