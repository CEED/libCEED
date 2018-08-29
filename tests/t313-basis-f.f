c-----------------------------------------------------------------------
c
c Header with common subroutine
c 
include 't310-basis-f.h'
c-----------------------------------------------------------------------
      subroutine feval(x1,x2,val)
      real*8 x1,x2,val

      val=x1*x1+x2*x2+x1*x2+1

      end
c-----------------------------------------------------------------------
      subroutine dfeval(x1,x2,val)
      real*8 x1,x2,val

      val=2*x1+x2

      end
c-----------------------------------------------------------------------
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
      real*8 xq(d*q)
      real*8 xr(d*p)
      real*8 input(p)
      real*8 output(d*q)
      real*8 val,diff
      real*8 x1,x2

      integer b

      character arg*32

      xq=(/2.d-1,6.d-1,1.d0/3.d0,2.d-1,2.d-1,2.d-1,
     $     1.d0/3.d0,6.d-1/)
      xr=(/0.d0,5.d-1,1.d0,0.d0,5.d-1,0.d0,0.d0,0.d0,
     $     0.d0,5.d-1,5.d-1,1.d0/)

      call getarg(1,arg)

      call buildmats(qref,qweight,interp,grad)

      call ceedinit(trim(arg)//char(0),ceed,err)
      call ceedbasiscreateh1(ceed,ceed_triangle,1,p,q,
     $  interp,grad,qref,qweight,b,err)

      do i=1,p
        x1=xr(0*p+i)
        x2=xr(1*p+i)
        call feval(x1,x2,val)
        input(i)=val
      enddo

      call ceedbasisapply(b,1,ceed_notranspose,ceed_eval_grad,
     $  input,output,err)

      do i=1,q
        x1=xq(0*q+i)
        x2=xq(1*q+i)
        call dfeval(x1,x2,val)
        diff=val-output(0*q+i)
        if (abs(diff)>1.0d-10) then
          write(*,'(A,I1,A,F12.8,A,F12.8)')
     $    '[',i,'] ',output(i),' != ',val
        endif
        call dfeval(x2,x1,val)
        diff=val-output(1*q+i)
        if (abs(diff)>1.0d-10) then
          write(*,'(A,I1,A,F12.8,A,F12.8)')
     $    '[',i,'] ',output(i),' != ',val
        endif
      enddo

      call ceedbasisdestroy(b,err)
      call ceeddestroy(ceed,err)

      end
c-----------------------------------------------------------------------
