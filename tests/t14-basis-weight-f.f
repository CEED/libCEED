c-----------------------------------------------------------------------
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
c-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer bxl,bxg,bug
      integer i
      integer q
      parameter(q=6)

      integer plen
      parameter(plen=6)
      real*8 p(plen)
      real*8 pint(plen+1)
      real*8 x(2)
      real*8 xq(q)
      real*8 uq(q)
      real*8 u(q)
      real*8 w(q)
      real*8 summ,error,p1,pm1

      character arg*32

      data p/1,2,3,4,5,6/
      data x/-1,1/

      call getarg(1,arg)
      call ceedinit(trim(arg)//char(0),ceed,err)

      call ceedbasiscreatetensorh1lagrange(ceed,1,1,2,q,
     $  ceed_gauss_lobatto,bxl,err)
      call ceedbasisapply(bxl,ceed_notranspose,ceed_eval_interp,
     $  x,xq,err)

      do i=1,q
        call polyeval(xq(i),plen,p,u(i))
      enddo

      call ceedbasiscreatetensorh1lagrange(ceed,1,1,2,q,ceed_gauss,
     $  bxg,err)
      call ceedbasiscreatetensorh1lagrange(ceed,1,1,q,q,ceed_gauss,
     $  bug,err)
      call ceedbasisapply(bxg,ceed_notranspose,ceed_eval_interp,
     $  x,xq,err)
      call ceedbasisapply(bug,ceed_notranspose,ceed_eval_interp,
     $  u,uq,err)
      call ceedbasisapply(bug,ceed_notranspose,ceed_eval_weight,
     $  %val(0),w,err)

      summ=0.0
      do i=1,q
        summ=summ+w(i)*uq(i)
      enddo

      pint(1)=0.0
      do i=1,plen
        pint(i+1)=p(i)/i
      enddo

      call polyeval(1.0D0,plen,pint,p1)
      call polyeval(-1.0D0,plen,pint,pm1)
      error=summ-p1+pm1
      if (abs(error) > 1e-10) then
        write(*,*) 'Error ',error,' sum ',summ,' exact ',p1-pm1
      endif

      call ceedbasisdestroy(bxl,err)
      call ceedbasisdestroy(bxg,err)
      call ceedbasisdestroy(bug,err)
      call ceeddestroy(ceed,err)
      end
c-----------------------------------------------------------------------
