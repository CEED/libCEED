!-----------------------------------------------------------------------
      program test
      implicit none
      include 'ceed/fortran.h'

      integer ceed,err
      integer x,n
      real*8 a(10)
      real*8 b(10)
      integer*8 aoffset,boffset
      character arg*32

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)

      n=10

      call ceedvectorcreate(ceed,n,x,err)
      call ceedvectorsetvalue(x,0.0,err)

      call ceedvectorgetarrayread(x,ceed_mem_host,a,aoffset,err)
      call ceedvectorgetarrayread(x,ceed_mem_host,b,boffset,err)

      call ceedvectorrestorearrayread(x,a,aoffset,err)
      call ceedvectorrestorearrayread(x,b,boffset,err)

      call ceedvectordestroy(x,err)
      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
