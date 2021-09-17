!-----------------------------------------------------------------------
      program test
      implicit none
      include 'ceed/fortran.h'

      integer ceed,err
      character arg*32

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)

      call ceedview(ceed,err)

      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
