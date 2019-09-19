!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer x,n
      integer*8 offset
      real*8 a(10)
      character arg*32

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)

      n=10

      call ceedvectorcreate(ceed,n,x,err)

      call ceedvectorrestorearray(x,a,offset,err)
! LCOV_EXCL_START
      call ceedvectordestroy(x,err)
      call ceeddestroy(ceed,err)

      end
! LCOV_EXCL_STOP
!-----------------------------------------------------------------------
