#include <ceed.h>

int main(int argc, char **argv) {
  Ceed ceed;

  assert(argv[1]);
  CeedInit(argv[1], &ceed);
  CeedDestroy(&ceed);
  return 0;
}
