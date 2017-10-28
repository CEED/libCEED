#include <feme.h>

int main(int argc, char **argv) {
  Feme feme;

  FemeInit("/cpu/self", &feme);
  FemeDestroy(&feme);
  return 0;
}
