!-----------------------------------------------------------------------
      program test
      implicit none
      include 'ceed/fortran.h'

      integer ceed,err
      integer isdeterministic
      character arg*32

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)
      call ceedisdeterministic(ceed,isdeterministic,err)
      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
