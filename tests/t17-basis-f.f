c-----------------------------------------------------------------------
      subroutine eval(dimn,x,rslt)
      integer dimn
      real*8 x(3)
      real*8 rslt

      rslt=tanh(x(1)+0.1)
      if (dimn>1) then
        rslt=rslt+atan(x(2)+0.2)
      endif
      if (dimn>2) then 
        rslt=rslt+exp(-(x(3)+0.3)*(x(3)+0.3))
      endif

      end
c-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer bxl,bug
      integer dimn,d
      integer i
      integer p
      integer q
      parameter(p=8)
      parameter(q=10)
      integer maxdim
      parameter(maxdim=3)
      integer qdimnmax
      parameter(qdimnmax=q**maxdim)
      integer pdimnmax
      parameter(pdimnmax=p**maxdim)
      integer xdimmax
      parameter(xdimmax=2**maxdim)
      integer pdimn,qdimn,xdim

      real*8 x(xdimmax*maxdim)
      real*8 xx(maxdim)
      real*8 xq(qdimnmax*maxdim)
      real*8 uq(qdimnmax)
      real*8 u(pdimnmax)
      real*8 ones(qdimnmax*maxdim)
      real*8 gtposeones(pdimnmax)
      real*8 sum1
      real*8 sum2
      integer dimxqdimn

      character arg*32

      call getarg(1,arg)
      call ceedinit(trim(arg)//char(0),ceed,err)

      do dimn=3,3
        qdimn=q**dimn
        pdimn=p**dimn
        xdim=2**dimn
        dimxqdimn=dimn*qdimn
        sum1=0
        sum2=0

        do i=1,dimxqdimn
          ones(i)=1
        enddo

        do d=0,dimn-1
          do i=1,xdim
            if ((mod(i-1,2**(dimn-d))/(2**(dimn-d-1))).ne.0) then
              x(d*xdim+i)=1
            else
              x(d*xdim+i)=-1
            endif
          enddo
        enddo

        call ceedbasiscreatetensorh1lagrange(ceed,dimn,dimn,2,p,
     $    ceed_gauss_lobatto,bxl,err)
        call ceedbasisapply(bxl,ceed_notranspose,ceed_eval_interp,
     $    x,xq,err)

        do i=1,pdimn
          do d=0,dimn-1
            xx(d+1)=xq(d*pdimn+i)
          enddo
          call eval(dimn,xx,u(i))
        enddo

        call ceedbasiscreatetensorh1lagrange(ceed,dimn,1,p,q,
     $    ceed_gauss,bug,err)

        call ceedbasisapply(bug,ceed_notranspose,ceed_eval_grad,
     $    u,uq,err)
        call ceedbasisapply(bug,ceed_transpose,ceed_eval_grad,
     $    ones,gtposeones,err)

        write(*,*) 'Sum 1:'
        do i=1,pdimn
          sum1=sum1+gtposeones(i)*u(i)
          write(*,*) sum1
        enddo
        write(*,*) 'Sum 2:'
        do i=1,dimxqdimn
          sum2=sum2+uq(i)
          write(*,*) sum2
        enddo
        if(dabs(sum1-sum2) > 1.0D-10) then
          write(*,'(A,I1,A,F12.6,A,F12.6)')'[',dimn,'] Error: ',sum1,
     $      ' != ',sum2
        endif

        call ceedbasisdestroy(bxl,err)
        call ceedbasisdestroy(bug,err)
      enddo

      call ceeddestroy(ceed,err)
      end
c-----------------------------------------------------------------------
