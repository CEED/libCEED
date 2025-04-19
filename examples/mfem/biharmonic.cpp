// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

//                         libCEED + MFEM Example: BP6
//
// This example illustrates a simple usage of libCEED with the MFEM (mfem.org) finite element library.
//
// It solves the biharmonic equation \Nabla^2 u =f on Omega, u(\vec{x})=0 on ∂(Omega), du/d(\vec{x})*n=0 on ∂(Omega) by solving -\Nabla w=f, -\Nabla u=w
// 
// 
//


/// @file
/// MFEM mass operator based on libCEED
#include <mfem.hpp>
#include <mfem/libceed/mfem-ceed.hpp>
#include <ceed.h>
#include <ceed-backend.h>
#include <ceed-examples.h>  
#include <fstream>
#include <iostream>

using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MFEM and CEED
  Device device("cpu");
  std::cout << "\nRunning on device: " << device.GetDeviceMemoryTypeName() << "\n\n";

  Ceed ceed;
  CeedInit("/cpu/self", &ceed);

  // 2. Read mesh from file
  const char *mesh_file = "mesh.mesh";
  Mesh mesh(mesh_file, 1, 1, true); // generate_edges = 1, refine = 1, fix_orientation = true
  mesh.EnsureNodes();
  int dim = mesh.Dimension();

  
  // 3. Set up finite element space
  H1_FECollection fec(order, dim);
  FiniteElementSpace fespace(&mesh, &fec);

  std::cout << "Number of finite element unknowns: " << fespace.GetTrueVSize() << std::endl;

  // 4. Load source function f from file
  GridFunction rhs(&fespace);
  std::ifstream fin("f.gf");
  if (!fin)
  {
    std::cerr << "Error: Unable to open file 'f.gf'.\n";
    return 1;
  }
  rhs.Load(fin);

  // 5. set up intermediate grid functions
  GridFunction w(&fespace);
  GridFunction u(&fespace);

  // 6. Use CEED built-in QFunctions for Laplacian
  CeedQFunction qf_build, qf_apply;
  CeedQFunctionCreateInterior(ceed, 1, f_build_diff, "f_build_diff", &qf_build);
  CeedQFunctionAddInput(qf_build, "dx", CEED_EVAL_GRAD, dim);
  CeedQFunctionAddOutput(qf_build, "qdata", CEED_EVAL_NONE, dim*(dim+1)/2);

  CeedQFunctionCreateInterior(ceed, 1, f_apply_diff, "f_apply_diff", &qf_apply);
  CeedQFunctionAddInput(qf_apply, "u", CEED_EVAL_INTERP, 1);
  CeedQFunctionAddInput(qf_apply, "grad_u", CEED_EVAL_GRAD, dim);
  CeedQFunctionAddInput(qf_apply, "qdata", CEED_EVAL_NONE, dim*(dim+1)/2);
  CeedQFunctionAddOutput(qf_apply, "v", CEED_EVAL_INTERP, 1);

  // 7. Build CEED operator with MFEM wrapper
  mfem::ceed::CeedOperatorBuilder builder(&fespace, qf_apply, qf_build, ceed);
  CeedOperator ceed_op = builder.GetCeedOperator();

   // 8. Create CEED linear solver
  mfem::ceed::CeedSolver solver(ceed_op);

  // 9. Solve -Δ w = f and -Δ u = w
  std::cout << "Solving -Δ w = f...\n";
  solver.Solve(rhs, w);

 
  std::cout << "Solving -Δ u = w...\n";
  solver.Solve(w, u);

  // 10. Save solutions
  std::ofstream ufile("u.gf");
  u.Save(ufile);
  std::cout << "Solutions saved to 'u.gf' .\n";

  // 11. destroy
  CeedDestroy(&ceed);
  return 0;
}