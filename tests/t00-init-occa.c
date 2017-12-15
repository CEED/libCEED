#include <ceed.h>

int main(int argc, char** argv) {
  Ceed ceed;

  CeedInit("/cpu/occa", &ceed);
  CeedDestroy(&ceed);
  return 0;
}
