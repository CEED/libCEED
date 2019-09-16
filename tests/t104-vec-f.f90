!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer x,n
      real*8 a(10)
      integer*8 aoffset
      character arg*32

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)

      n=10

      call ceedvectorcreate(ceed,n,x,err)

      call ceedvectorgetarray(x,ceed_mem_host,a,aoffset,err)

      call ceedvectordestroy(x,err)
! LCOV_EXCL_START
      call ceeddestroy(ceed,err)

      end
! LCOV_EXCL_STOP
!-----------------------------------------------------------------------
