#ifndef BIHARMONIC_HPP
#define BIHARMONIC_HPP

#include <mfem.hpp>
#include <ceed.h>

namespace biharmonic
{
   // Initializes the MFEM device.
   void InitializeDevice();

   // Initializes the libCEED context.
   void InitializeCeed(Ceed &ceed, const std::string &resource = "/cpu/self");

   // Loads the mesh from a file and prepares it.
   mfem::Mesh *LoadMesh(const std::string &mesh_file, int ref_levels);

   // Sets up the finite element space.
   mfem::FiniteElementSpace *SetupFESpace(mfem::Mesh *mesh, int order);

   // Loads the source function 'f' from a file.
   mfem::GridFunction *LoadRHS(const std::string &rhs_file, mfem::FiniteElementSpace *fespace);

   // Sets up the CEED QFunctions for the Laplacian.
   void SetupQFunctions(Ceed ceed, CeedQFunction &qf_build, CeedQFunction &qf_apply, int dim);

   // Builds the CEED operator using the MFEM wrapper.
   CeedOperator BuildCeedOperator(mfem::FiniteElementSpace *fespace, CeedQFunction qf_apply, CeedQFunction qf_build, Ceed ceed);

   // Solves the equation using the CEED solver.
   void Solve(CeedOperator ceed_op, mfem::GridFunction &rhs, mfem::GridFunction &solution);

   // Saves the solution to a file.
   void SaveSolution(const mfem::GridFunction &solution, const std::string &filename);
}

#endif // BIHARMONIC_HPP