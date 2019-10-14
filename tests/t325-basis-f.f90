!-----------------------------------------------------------------------
! 
! Header with common subroutine
! 
      include 't320-basis-f.h'
!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer input,output
      integer p,q,d,n
      parameter(p=6)
      parameter(q=4)
      parameter(d=2)
      parameter(n=3)

      real*8 qref(d*q)
      real*8 qweight(q)
      real*8 interp(p*q)
      real*8 grad(d*p*q)
      real*8 iinput(d*q*n)
      real*8 ooutput(p*n)
      real*8 colsum(p)
      integer*8 ioffset,ooffset

      integer b

      character arg*32

      call getarg(1,arg)

      call buildmats(qref,qweight,interp,grad)

      call ceedinit(trim(arg)//char(0),ceed,err)

      do i=1,p
      colsum(i)=0
      do j=0,q*d-1
        colsum(i)=colsum(i)+grad(i+j*p)
      enddo
      enddo

      call ceedbasiscreateh1(ceed,ceed_triangle,n,p,q,interp,grad,qref,&
     & qweight,b,err)

      call ceedvectorcreate(ceed,q*d*n,input,err)
      do i=0,d-1
      do j=0,n-1
      do k=1,q
        iinput(k+(j+i*n)*q)=j*1.d0
      enddo
      enddo
      enddo
      ioffset=0
      call ceedvectorsetarray(input,ceed_mem_host,ceed_copy_values,&
     & iinput,ioffset,err)
      call ceedvectorcreate(ceed,p*n,output,err)
      call ceedvectorsetvalue(output,0.d0,err)

      call ceedbasisapply(b,1,ceed_transpose,ceed_eval_grad,input,output,err)

      call ceedvectorgetarrayread(output,ceed_mem_host,ooutput,ooffset,err)
      do i=1,p
      do j=0,n-1
        if (abs(ooutput(i+j*p+ooffset)-j*colsum(i))>1.0d-14) then
! LCOV_EXCL_START
          write(*,'(A,I1,A,F12.8,A,F12.8)')  '[',i,'] ',ooutput(i+j*p+ooffset),&
     &     ' != ',j*colsum(i)
! LCOV_EXCL_STOP
        endif
      enddo
      enddo
      call ceedvectorrestorearrayread(output,ooutput,ooffset,err)

      call ceedvectordestroy(input,err)
      call ceedvectordestroy(output,err)
      call ceedbasisdestroy(b,err)
      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
