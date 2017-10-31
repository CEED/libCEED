#include <feme.h>

int main(int argc, char **argv) {
  Feme feme;
  FemeVec x;

  FemeInit("/cpu/self", &feme);
  FemeVecCreate(feme, 10, &x);
  FemeVecDestroy(&x);
  FemeDestroy(&feme);
  return 0;
}
