!-----------------------------------------------------------------------
      program test
      implicit none
      include 'ceed/fortran.h'

      integer ceed,err
      integer r

      integer ne
      parameter(ne=3)
      integer strides(3)

      character arg*32

      call getarg(1,arg)
      call ceedinit(trim(arg)//char(0),ceed,err)

      strides=[1,2,2]
      call ceedelemrestrictioncreateblockedstrided(ceed,ne,2,2,1,ne+1,&
     & strides,r,err)

      call ceedelemrestrictionview(r,err)

      call ceedelemrestrictiondestroy(r,err)
      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
