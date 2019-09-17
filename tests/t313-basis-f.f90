!-----------------------------------------------------------------------
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
!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer x,xq,u,uq
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

      real*8 xx(xdimmax*maxdim)
      real*8 xxx(maxdim)
      real*8 xxq(qdimmax*maxdim)
      real*8 uuq(qdimmax)
      real*8 fx
      integer*8 uqoffset,xoffset,offset1,offset2

      character arg*32

      call getarg(1,arg)
      call ceedinit(trim(arg)//char(0),ceed,err)

      do dimn=1,maxdim
        qdim=q**dimn
        xdim=2**dimn

        do d=0,dimn-1
          do i=1,xdim
            if ((mod(i-1,2**(dimn-d))/(2**(dimn-d-1))).ne.0) then
              xx(d*xdim+i)=1
            else
              xx(d*xdim+i)=-1
            endif
          enddo
        enddo

        call ceedvectorcreate(ceed,xdim*dimn,x,err)
        xoffset=0
        call ceedvectorsetarray(x,ceed_mem_host,ceed_use_pointer,xx,xoffset,err)
        call ceedvectorcreate(ceed,qdim*dimn,xq,err)
        call ceedvectorsetvalue(xq,0.d0,err)
        call ceedvectorcreate(ceed,qdim,u,err)
        call ceedvectorsetvalue(u,0.d0,err)
        call ceedvectorcreate(ceed,qdim,uq,err)

        call ceedbasiscreatetensorh1lagrange(ceed,dimn,dimn,2,q,&
     &   ceed_gauss_lobatto,bxl,err)
        call ceedbasiscreatetensorh1lagrange(ceed,dimn,1,q,q,&
     &   ceed_gauss_lobatto,bul,err)

        call ceedbasisapply(bxl,1,ceed_notranspose,ceed_eval_interp,x,xq,err)

        call ceedvectorgetarrayread(xq,ceed_mem_host,xxq,offset1,err)
        do i=1,qdim
          do d=0,dimn-1
            xxx(d+1)=xxq(d*qdim+i+offset1)
          enddo
          call eval(dimn,xxx,uuq(i))
        enddo
        call ceedvectorrestorearrayread(xq,xxq,offset1,err)
        uqoffset=0
        call ceedvectorsetarray(uq,ceed_mem_host,ceed_use_pointer,uuq,uqoffset,&
     &   err)

        call ceedbasisapply(bul,1,ceed_transpose,ceed_eval_interp,uq,u,err)

        call ceedbasiscreatetensorh1lagrange(ceed,dimn,dimn,2,q,ceed_gauss,bxg,&
     &   err)
        call ceedbasiscreatetensorh1lagrange(ceed,dimn,1,q,q,ceed_gauss,bug,err)

        call ceedbasisapply(bxg,1,ceed_notranspose,ceed_eval_interp,x,xq,err)
        call ceedbasisapply(bug,1,ceed_notranspose,ceed_eval_interp,u,uq,err)

        call ceedvectorgetarrayread(xq,ceed_mem_host,xxq,offset1,err)
        call ceedvectorgetarrayread(uq,ceed_mem_host,uuq,offset2,err)
        do i=1,qdim
          do d=0,dimn-1
            xxx(d+1)=xxq(d*qdim+i+offset1)
          enddo
          call eval(dimn,xxx,fx)

          if(dabs(uuq(i+offset2)-fx) > 1.0D-4) then
! LCOV_EXCL_START
          write(*,*) uuq(i+offset2),' not equal to ',fx,dimn,i
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
      enddo

      call ceeddestroy(ceed,err)
      end
!-----------------------------------------------------------------------
