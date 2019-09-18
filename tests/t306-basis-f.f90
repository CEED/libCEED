!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer b
      integer p,q

      character arg*32

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)
      call ceedbasiscreatetensorh1lagrange(ceed,3,1,4,5,ceed_gauss_lobatto,b,&
     & err)

      call ceedbasisgetnumnodes(b,p,err)
      call ceedbasisgetnumquadraturepoints(b,q,err)


      if (p .NE. 64) then
! LCOV_EXCL_START
        write(*,*) 'Error ',p,' != 64 '
! LCOV_EXCL_STOP
      endif
      if (q .NE. 125) then
! LCOV_EXCL_START
        write(*,*) 'Error ',q,' != 125 '
! LCOV_EXCL_STOP
      endif

      call ceedbasisdestroy(b,err)
      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
