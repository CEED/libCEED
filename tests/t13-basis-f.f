c-----------------------------------------------------------------------
      subroutine eval(dimn,x,rslt)
      integer dimn
      real*8 x(1)
      real*8 rslt
      real*8 center

      integer d

      rslt=1
      center=0.1

      do d=1,dimn
        rslt=rslt*tanh(x(d)-center)
        center=center+0.1
      enddo

      end
c-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer bxl,bul,bxg,bug
      integer dimn,d
      integer i
      integer q
      parameter(q=10)
      integer maxdim
      parameter(maxdim=3)
      integer qdimmax
      parameter(qdimmax=q**maxdim)
      integer xdimmax
      parameter(xdimmax=2**maxdim)
      integer qdim,xdim

      real*8 x(xdimmax*maxdim)
      real*8 xx(maxdim)
      real*8 xq(qdimmax*maxdim)
      real*8 uq(qdimmax)
      real*8 u(qdimmax)
      real*8 fx

      character arg*32

      call getarg(1,arg)
      call ceedinit(trim(arg)//char(0),ceed,err)

      do dimn=1,3
        qdim=q**dimn
        xdim=2**dimn

        do d=0,dimn-1
          do i=1,xdim
            if ((mod(i-1,2**(dimn-d))/(2**(dimn-d-1))).ne.0) then
              x(d*xdim+i)=1
            else
              x(d*xdim+i)=-1
            endif
          enddo
        enddo

c        do i=1,dimn*xdim
c          write(*,*) 'x:',x(i)
c        enddo
        call ceedbasiscreatetensorh1lagrange(ceed,dimn,dimn,2,q,
     $    ceed_gauss_lobatto,bxl,err)
        call ceedbasiscreatetensorh1lagrange(ceed,dimn,1,q,q,
     $    ceed_gauss_lobatto,bul,err)
        call ceedbasisapply(bxl,1,ceed_notranspose,ceed_eval_interp,
     $    x,xq,err)

        do i=1,qdim
          do d=0,dimn-1
            xx(d+1)=xq(d*qdim+i)
          enddo
          call eval(dimn,xx,uq(i))
        enddo

        call ceedbasisapply(bul,1,ceed_transpose,ceed_eval_interp,
     $    uq,u,err)

        call ceedbasiscreatetensorh1lagrange(ceed,dimn,dimn,2,q,
     $    ceed_gauss,bxg,err)
        call ceedbasiscreatetensorh1lagrange(ceed,dimn,1,q,q,
     $    ceed_gauss,bug,err)

        call ceedbasisapply(bxg,1,ceed_notranspose,ceed_eval_interp,
     $    x,xq,err)
        call ceedbasisapply(bug,1,ceed_notranspose,ceed_eval_interp,
     $    u,uq,err)

        do i=1,qdim
          do d=0,dimn-1
            xx(d+1)=xq(d*qdim+i)
          enddo
          call eval(dimn,xx,fx)

          if(dabs(uq(i)-fx) > 1.0D-4) then
            write(*,*) 'Error: Not close enough'
          endif
        enddo

        call ceedbasisdestroy(bxl,err)
        call ceedbasisdestroy(bul,err)
        call ceedbasisdestroy(bxg,err)
        call ceedbasisdestroy(bug,err)
      enddo

      call ceeddestroy(ceed,err)
      end
c-----------------------------------------------------------------------
