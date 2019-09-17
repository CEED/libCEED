!-----------------------------------------------------------------------
!
! Header with common subroutine
! 
      include 't320-basis-f.h'
!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer p,q,d
      parameter(p=6)
      parameter(q=4)
      parameter(d=2)

      real*8 qref(d*q)
      real*8 qweight(q)
      real*8 interp(p*q)
      real*8 grad(d*p*q)

      integer b

      character arg*32

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)

      call buildmats(qref,qweight,interp,grad)

      call ceedbasiscreateh1(ceed,ceed_triangle,1,p,q,interp,grad,qref,qweight,&
     & b,err)
      call ceedbasisview(b,err)

      call ceedbasisdestroy(b,err)
      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
