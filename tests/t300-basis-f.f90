!-----------------------------------------------------------------------
      program test
      implicit none
      include 'ceed/fortran.h'

      integer ceed,err
      integer b

      character arg*32

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)
      call ceedbasiscreatetensorh1lagrange(ceed,1,1,4,4,ceed_gauss_lobatto,b,&
     & err)
      call ceedbasisview(b,err)
      call ceedbasisdestroy(b,err)

      call ceedbasiscreatetensorh1lagrange(ceed,1,1,4,4,ceed_gauss,b,err)
      call ceedbasisview(b,err)
      call ceedbasisdestroy(b,err)
      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
