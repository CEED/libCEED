!-----------------------------------------------------------------------
! 
! Header with common subroutine
! 
      include 't310-basis-f.h'
!-----------------------------------------------------------------------
      subroutine feval(x1,x2,val)
      real*8 x1,x2,val

      val=x1*x1+x2*x2+x1*x2+1

      end
!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer input,output,weights
      integer p,q,d
      parameter(p=6)
      parameter(q=4)
      parameter(d=2)

      real*8 qref(d*q)
      real*8 qweight(q)
      real*8 interp(p*q)
      real*8 grad(d*p*q)
      real*8 xr(d*p)
      real*8 iinput(p)
      real*8 ooutput(q)
      real*8 wweights(q)
      real*8 val,diff
      real*8 x1,x2
      integer*8 ioffset,offset1,offset2

      integer b

      character arg*32

      xr=(/0.0d0,5.0d-1,1.0d0,0.0d0,5.0d-1,0.0d0,0.0d0,0.0d0,0.0d0,5.0d-1,&
     &  5.0d-1,1.0d0/)

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
      call ceedvectorcreate(ceed,q,weights,err)
      call ceedvectorsetvalue(weights,0.d0,err)

      call ceedbasisapply(b,1,ceed_notranspose,ceed_eval_interp,input,output,&
     & err)
      call ceedbasisapply(b,1,ceed_notranspose,ceed_eval_weight,ceed_null,&
     & weights,err)

      call ceedvectorgetarrayread(output,ceed_mem_host,ooutput,offset1,err)
      call ceedvectorgetarrayread(weights,ceed_mem_host,wweights,offset2,err)
      val=0
      do i=1,q
        val=val+ooutput(i+offset1)*wweights(i+offset2)
      enddo
      call ceedvectorrestorearrayread(output,ooutput,offset1,err)
      call ceedvectorrestorearrayread(weights,wweights,offset2,err)

      diff=val-17.d0/24.d0
      if (abs(diff)>1.0d-10) then
! LCOV_EXCL_START
        write(*,'(A,I1,A,F12.8,A,F12.8)')'[',i,'] ',val,' != ',17.d0/24.d0
! LCOV_EXCL_STOP
      endif

      call ceedvectordestroy(input,err)
      call ceedvectordestroy(output,err)
      call ceedvectordestroy(weights,err)
      call ceedbasisdestroy(b,err)
      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
