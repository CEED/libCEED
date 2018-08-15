c-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer x,n
      real*8 a(10)
      real*8 b(10)
      character arg*32

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)

      n=10

      call ceedvectorcreate(ceed,n,x,err)

      call ceedvectorgetarrayread(x,ceed_mem_host,a,err)
      call ceedvectorgetarrayread(x,ceed_mem_host,b,err)

      call ceedvectordestroy(x,err)
      call ceeddestroy(ceed,err)

      end
c-----------------------------------------------------------------------
