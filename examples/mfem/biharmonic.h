#ifndef BIHARMONIC_H
#define BIHARMONIC_H

#include <mfem.hpp>
#include <ceed.h>
#include <string>

namespace biharmonic
{
   // Initialize the MFEM device (e.g., CPU, GPU).
   void InitializeDevice();

   // Initialize the libCEED context with the specified resource.
   void InitializeCeed(Ceed &ceed, const std::string &resource = "/cpu/self");

   // Load the mesh from a file and perform uniform refinements.
   mfem::Mesh *LoadMesh(const std::string &mesh_file, int ref_levels = 1);

   // Set up the finite element space with the given polynomial order.
   mfem::FiniteElementSpace *SetupFESpace(mfem::Mesh *mesh, int order);

   // Load the right-hand side function 'f' from a file into a GridFunction.
   mfem::GridFunction *LoadRHS(const std::string &rhs_file, mfem::FiniteElementSpace *fespace);

   // Set up the CEED QFunctions for the Laplacian operator.
   void SetupQFunctions(Ceed ceed, CeedQFunction &qf_build, CeedQFunction &qf_apply, int dim);

   // Build the CEED operator using the MFEM wrapper.
   CeedOperator BuildCeedOperator(mfem::FiniteElementSpace *fespace, CeedQFunction qf_apply, CeedQFunction qf_build, Ceed ceed);

   // Solve the equation using the CEED solver.
   void Solve(CeedOperator ceed_op, mfem::GridFunction &rhs, mfem::GridFunction &solution);

   // Save the solution to a file.
   void SaveSolution(const mfem::GridFunction &solution, const std::string &filename);
}

#endif // BIHARMONIC_H