!-----------------------------------------------------------------------
! 
! Header with common subroutine
! 
      include 't320-basis-f.h'
!-----------------------------------------------------------------------
      subroutine feval(x1,x2,val)
      real*8 x1,x2,val

      val=x1*x1+x2*x2+x1*x2+1

      end
!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer input,output
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
      real*8 iinput(p)
      real*8 ooutput(q)
      real*8 val,diff
      real*8 x1,x2
      integer*8 ioffset,ooffset

      integer b

      character arg*32

      xq=(/2.d-1,6.d-1,1.d0/3.d0,2.d-1,2.d-1,2.d-1,   1.d0/3.d0,6.d-1/)
      xr=(/0.d0,5.d-1,1.d0,0.d0,5.d-1,0.d0,0.d0,0.d0,   0.d0,5.d-1,5.d-1,1.d0/)

      call getarg(1,arg)

      call buildmats(qref,qweight,interp,grad)

      call ceedinit(trim(arg)//char(0),ceed,err)

      call ceedbasiscreateh1(ceed,ceed_triangle,1,p,q,interp,grad,qref,qweight,&
     & b,err)

      do i=1,p
        x1=xr(0*p+i)
        x2=xr(1*p+i)
        call feval(x1,x2,val)
        iinput(i)=val
      enddo

      call ceedvectorcreate(ceed,p,input,err)
      ioffset=0
      call ceedvectorsetarray(input,ceed_mem_host,ceed_use_pointer,iinput,&
     & ioffset,err)
      call ceedvectorcreate(ceed,q,output,err)
      call ceedvectorsetvalue(output,0.d0,err)

      call ceedbasisapply(b,1,ceed_notranspose,ceed_eval_interp,input,output,&
     & err)

      call ceedvectorgetarrayread(output,ceed_mem_host,ooutput,ooffset,err)
      do i=1,q
        x1=xq(0*q+i)
        x2=xq(1*q+i)
        call feval(x1,x2,val)
        diff=val-ooutput(i+ooffset)
        if (abs(diff)>1.0d-10) then
! LCOV_EXCL_START
          write(*,'(A,I1,A,F12.8,A,F12.8)')  '[',i,'] ',ooutput(i+ooffset),&
     &     ' != ',val
! LCOV_EXCL_STOP
        endif
      enddo
      call ceedvectorrestorearrayread(output,ooutput,ooffset,err)

      call ceedvectordestroy(input,err)
      call ceedvectordestroy(output,err)
      call ceedbasisdestroy(b,err)
      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
