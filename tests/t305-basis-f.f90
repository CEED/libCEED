!-----------------------------------------------------------------------
      subroutine polyeval(x,n,p,uq)
      real*8 x,y
      integer n,i
      real*8 p(1)
      real*8 uq

      y=p(n) 

      do i=n-1,1,-1
        y=y*x+p(i)
      enddo

      uq=y

      end
!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer x,xq,u,uq,w
      integer bxl,bxg,bug
      integer i
      integer q
      parameter(q=6)

      integer plen
      parameter(plen=6)
      real*8 p(plen)
      real*8 pint(plen+1)
      real*8 xx(2)
      real*8 xxq(q)
      real*8 uuq(q)
      real*8 uu(q)
      real*8 ww(q)
      real*8 summ,error,p1,pm1
      integer*8 uoffset,xoffset,offset1,offset2

      character arg*32

      data p/1,2,3,4,5,6/
      data xx/-1,1/

      call getarg(1,arg)
      call ceedinit(trim(arg)//char(0),ceed,err)

      call ceedvectorcreate(ceed,2,x,err)
      xoffset=0
      call ceedvectorsetarray(x,ceed_mem_host,ceed_use_pointer,xx,xoffset,err)
      call ceedvectorcreate(ceed,q,xq,err)
      call ceedvectorsetvalue(xq,0.d0,err)
      call ceedvectorcreate(ceed,q,u,err)
      call ceedvectorcreate(ceed,q,uq,err)
      call ceedvectorsetvalue(uq,0.d0,err)
      call ceedvectorcreate(ceed,q,w,err)
      call ceedvectorsetvalue(w,0.d0,err)

      call ceedbasiscreatetensorh1lagrange(ceed,1,1,2,q,ceed_gauss_lobatto,bxl,&
     & err)

      call ceedbasisapply(bxl,1,ceed_notranspose,ceed_eval_interp,x,xq,err)

      call ceedvectorgetarrayread(xq,ceed_mem_host,xxq,offset1,err)
      do i=1,q
        call polyeval(xxq(i+offset1),plen,p,uu(i))
      enddo
      call ceedvectorrestorearrayread(xq,xxq,offset1,err)
      uoffset=0
      call ceedvectorsetarray(u,ceed_mem_host,ceed_use_pointer,uu,uoffset,err)

      call ceedbasiscreatetensorh1lagrange(ceed,1,1,2,q,ceed_gauss,bxg,err)
      call ceedbasiscreatetensorh1lagrange(ceed,1,1,q,q,ceed_gauss,bug,err)

      call ceedbasisapply(bxg,1,ceed_notranspose,ceed_eval_interp,x,xq,err)
      call ceedbasisapply(bug,1,ceed_notranspose,ceed_eval_interp,u,uq,err)
      call ceedbasisapply(bug,1,ceed_notranspose,ceed_eval_weight,ceed_null,w,&
     & err)

      call ceedvectorgetarrayread(w,ceed_mem_host,ww,offset1,err)
      call ceedvectorgetarrayread(uq,ceed_mem_host,uuq,offset2,err)
      summ=0.0
      do i=1,q
        summ=summ+ww(i+offset1)*uuq(i+offset2)
      enddo
      call ceedvectorrestorearrayread(w,ww,offset1,err)
      call ceedvectorrestorearrayread(uq,uuq,offset2,err)

      pint(1)=0.0
      do i=1,plen
        pint(i+1)=p(i)/i
      enddo

      call polyeval(1.0D0,plen,pint,p1)
      call polyeval(-1.0D0,plen,pint,pm1)
      error=summ-p1+pm1
      if (abs(error) > 1e-10) then
! LCOV_EXCL_START
        write(*,*) 'Error ',error,' sum ',summ,' exact ',p1-pm1
! LCOV_EXCL_STOP
      endif

      call ceedvectordestroy(x,err)
      call ceedvectordestroy(xq,err)
      call ceedvectordestroy(u,err)
      call ceedvectordestroy(uq,err)
      call ceedvectordestroy(w,err)
      call ceedbasisdestroy(bxl,err)
      call ceedbasisdestroy(bxg,err)
      call ceedbasisdestroy(bug,err)
      call ceeddestroy(ceed,err)
      end
!-----------------------------------------------------------------------
