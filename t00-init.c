#include <ceed.h>

int main(int argc, char **argv)
{
   Ceed ceed;

   CeedInit("/cpu/self", &ceed);
   CeedDestroy(&ceed);
   return 0;
}
