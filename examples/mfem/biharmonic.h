#ifndef BIHARMONIC_H
#define BIHARMONIC_H

#include <mfem.hpp>
#include <ceed.h>
#include <string>

namespace biharmonic {
  void InitializeDevice();
  void InitializeCeed(Ceed &ceed, const std::string &resource = "/cpu/self");
  mfem::Mesh *LoadMesh(const std::string &mesh_file);
  mfem::FiniteElementSpace *SetupFESpace(mfem::Mesh *mesh, int order);
  mfem::GridFunction *LoadRHS(const std::string &rhs_file, mfem::FiniteElementSpace *fespace);
  void SetupQFunctions(Ceed ceed, CeedQFunction &qf_build, CeedQFunction &qf_apply, int dim);
  CeedOperator BuildCeedOperator(mfem::FiniteElementSpace *fespace, CeedQFunction qf_apply, CeedQFunction qf_build, Ceed ceed);
  void Solve(CeedOperator ceed_op, mfem::GridFunction &rhs, mfem::GridFunction &solution);
  void SaveSolution(const mfem::GridFunction &solution, const std::string &filename);
} // namespace biharmonic

#endif // BIHARMONIC_H
