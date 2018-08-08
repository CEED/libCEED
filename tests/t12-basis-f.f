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
      integer bxl,bul,bxg,bug
      integer i
      integer q
      parameter(q=6)

      real*8 p(6)
      real*8 x(2)
      real*8 xq(q)
      real*8 uq(q)
      real*8 u(q)
      real*8 px

      character arg*32

      data p/1,2,3,4,5,6/
      data x/-1,1/

      call getarg(1,arg)
      call ceedinit(trim(arg)//char(0),ceed,err)

      call ceedbasiscreatetensorh1lagrange(ceed,1,1,2,q,
     $  ceed_gauss_lobatto,bxl,err)
      call ceedbasiscreatetensorh1lagrange(ceed,1,1,q,q,
     $  ceed_gauss_lobatto,bul,err)
      call ceedbasisapply(bxl,1,ceed_notranspose,ceed_eval_interp,
     $  x,xq,err)

      do i=1,q
        call polyeval(xq(i),6,p,uq(i))
      enddo

      call ceedbasisapply(bul,1,ceed_transpose,ceed_eval_interp,uq,u,
     $  err)

      call ceedbasiscreatetensorh1lagrange(ceed,1,1,2,q,ceed_gauss,
     $  bxg,err)
      call ceedbasiscreatetensorh1lagrange(ceed,1,1,q,q,ceed_gauss,
     $  bug,err)

      do i=1,q
        call polyeval(xq(i),6,p,px)
        if (abs(uq(i)-px) > 1e-14) then
          write(*,*) uq(i),' not eqaul to ',px,'=p(',xq(i),')'
        endif
      enddo

      call ceedbasisdestroy(bxl,err)
      call ceedbasisdestroy(bul,err)
      call ceedbasisdestroy(bxg,err)
      call ceedbasisdestroy(bug,err)
      call ceeddestroy(ceed,err)
      end
c-----------------------------------------------------------------------
