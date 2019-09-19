!-----------------------------------------------------------------------
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
!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer x,xq,u,uq,ones,gtposeones
      integer bxl,bug
      integer dimn,d
      integer i
      integer p
      integer q
      parameter(p=8)
      parameter(q=8)
      integer maxdim
      parameter(maxdim=3)
      integer qdimnmax
      parameter(qdimnmax=q**maxdim)
      integer pdimnmax
      parameter(pdimnmax=p**maxdim)
      integer xdimmax
      parameter(xdimmax=2**maxdim)
      integer pdimn,qdimn,xdim

      real*8 xx(xdimmax*maxdim)
      real*8 xxx(maxdim)
      real*8 xxq(pdimnmax*maxdim)
      real*8 uuq(qdimnmax*maxdim)
      real*8 uu(pdimnmax)
      real*8 ggtposeones(pdimnmax)
      real*8 sum1
      real*8 sum2
      integer dimxqdimn
      integer*8 uoffset,xoffset,offset1,offset2,offset3

      character arg*32

      call getarg(1,arg)
      call ceedinit(trim(arg)//char(0),ceed,err)

      do dimn=1,maxdim
        qdimn=q**dimn
        pdimn=p**dimn
        xdim=2**dimn
        dimxqdimn=dimn*qdimn
        sum1=0
        sum2=0

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
        call ceedvectorcreate(ceed,pdimn*dimn,xq,err)
        call ceedvectorsetvalue(xq,0.d0,err)
        call ceedvectorcreate(ceed,pdimn,u,err)
        call ceedvectorcreate(ceed,qdimn*dimn,uq,err)
        call ceedvectorsetvalue(uq,0.d0,err)
        call ceedvectorcreate(ceed,qdimn*dimn,ones,err)
        call ceedvectorsetvalue(ones,1.d0,err)
        call ceedvectorcreate(ceed,pdimn,gtposeones,err)
        call ceedvectorsetvalue(gtposeones,0.d0,err)

        call ceedbasiscreatetensorh1lagrange(ceed,dimn,dimn,2,p,&
     &   ceed_gauss_lobatto,bxl,err)
        call ceedbasisapply(bxl,1,ceed_notranspose,ceed_eval_interp,x,xq,err)

        call ceedvectorgetarrayread(xq,ceed_mem_host,xxq,offset1,err)
        do i=1,pdimn
          do d=0,dimn-1
            xxx(d+1)=xxq(d*pdimn+i+offset1)
          enddo
          call eval(dimn,xxx,uu(i))
        enddo
        call ceedvectorrestorearrayread(xq,xxq,offset1,err)
        uoffset=0
        call ceedvectorsetarray(u,ceed_mem_host,ceed_use_pointer,uu,uoffset,err)

        call ceedbasiscreatetensorh1lagrange(ceed,dimn,1,p,q,&
     &    ceed_gauss_lobatto,bug,err)

        call ceedbasisapply(bug,1,ceed_notranspose,ceed_eval_grad,u,uq,err)
        call ceedbasisapply(bug,1,ceed_transpose,ceed_eval_grad,ones,&
     &   gtposeones,err)

        call ceedvectorgetarrayread(gtposeones,ceed_mem_host,ggtposeones,&
     &   offset1,err)
        call ceedvectorgetarrayread(u,ceed_mem_host,uu,offset2,err)
        call ceedvectorgetarrayread(uq,ceed_mem_host,uuq,offset3,err)
        do i=1,pdimn
          sum1=sum1+ggtposeones(i+offset1)*uu(i+offset2)
        enddo
        do i=1,dimxqdimn
          sum2=sum2+uuq(i+offset3)
        enddo
        call ceedvectorrestorearrayread(gtposeones,ggtposeones,offset1,err)
        call ceedvectorrestorearrayread(u,uu,offset2,err)
        call ceedvectorrestorearrayread(uq,uuq,offset3,err)
        if(abs(sum1-sum2) > 1.0D-10) then
! LCOV_EXCL_START
          write(*,'(A,I1,A,F12.6,A,F12.6)')'[',dimn,'] Error: ',sum1,' != ',&
     &     sum2
! LCOV_EXCL_STOP
        endif

        call ceedvectordestroy(x,err)
        call ceedvectordestroy(xq,err)
        call ceedvectordestroy(u,err)
        call ceedvectordestroy(uq,err)
        call ceedvectordestroy(ones,err)
        call ceedvectordestroy(gtposeones,err)
        call ceedbasisdestroy(bxl,err)
        call ceedbasisdestroy(bug,err)
      enddo

      call ceeddestroy(ceed,err)
      end
!-----------------------------------------------------------------------
