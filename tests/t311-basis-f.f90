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
      integer x,xq,u,uq
      integer bxl,bul,bxg,bug
      integer i
      integer q
      parameter(q=6)

      real*8 p(6)
      real*8 xx(2)
      real*8 xxq(q)
      real*8 uuq(q)
      real*8 px
      integer*8 uqoffset,xoffset,offset1,offset2

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
      call ceedvectorsetvalue(u,0.d0,err)
      call ceedvectorcreate(ceed,q,uq,err)
      call ceedvectorsetvalue(uq,0.d0,err)

      call ceedbasiscreatetensorh1lagrange(ceed,1,1,2,q,ceed_gauss_lobatto,&
     & bxl,err)
      call ceedbasiscreatetensorh1lagrange(ceed,1,1,q,q,ceed_gauss_lobatto,&
     & bul,err)

      call ceedbasisapply(bxl,1,ceed_notranspose,ceed_eval_interp,x,xq,err)

      call ceedvectorgetarrayread(xq,ceed_mem_host,xxq,offset1,err)
      do i=1,q
        call polyeval(xxq(i+offset1),6,p,uuq(i))
      enddo
      call ceedvectorrestorearrayread(xq,xxq,offset1,err)
      uqoffset=0
      call ceedvectorsetarray(uq,ceed_mem_host,ceed_use_pointer,uuq,uqoffset,&
     & err)

      call ceedbasisapply(bul,1,ceed_transpose,ceed_eval_interp,uq,u,err)

      call ceedbasiscreatetensorh1lagrange(ceed,1,1,2,q,ceed_gauss,bxg,err)
      call ceedbasiscreatetensorh1lagrange(ceed,1,1,q,q,ceed_gauss,bug,err)

      call ceedbasisapply(bxg,1,ceed_notranspose,ceed_eval_interp,x,xq,err)
      call ceedbasisapply(bug,1,ceed_notranspose,ceed_eval_interp,u,uq,err)

      call ceedvectorgetarrayread(xq,ceed_mem_host,xxq,offset1,err)
      call ceedvectorgetarrayread(uq,ceed_mem_host,uuq,offset2,err)
      do i=1,q
        call polyeval(xxq(i+offset1),6,p,px)
        if (abs(uuq(i+offset2)-px) > 1e-14) then
! LCOV_EXCL_START
          write(*,*) uuq(i+offset2),' not eqaul to ',px,'=p(',xxq(i+offset1),')'
! LCOV_EXCL_STOP
        endif
      enddo
      call ceedvectorrestorearrayread(xq,xxq,offset1,err)
      call ceedvectorrestorearrayread(uq,uuq,offest2,err)

      call ceedvectordestroy(x,err)
      call ceedvectordestroy(xq,err)
      call ceedvectordestroy(u,err)
      call ceedvectordestroy(uq,err)
      call ceedbasisdestroy(bxl,err)
      call ceedbasisdestroy(bul,err)
      call ceedbasisdestroy(bxg,err)
      call ceedbasisdestroy(bug,err)
      call ceeddestroy(ceed,err)
      end
!-----------------------------------------------------------------------
