/// @file
/// Test viewing of QFunction by name
/// \test Test viewing of QFunction by name
#include <ceed.h>

int main(int argc, char **argv) {
  Ceed          ceed;
  CeedQFunction qf_setup, qf_mass;

  CeedInit(argv[1], &ceed);

  CeedQFunctionCreateInteriorByName(ceed, "Mass1DBuild", &qf_setup);
  CeedQFunctionCreateInteriorByName(ceed, "MassApply", &qf_mass);

  CeedQFunctionView(qf_setup, stdout);
  CeedQFunctionView(qf_mass, stdout);

  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedDestroy(&ceed);
  return 0;
}
