!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer b,i,u,v

      integer dimn,p1d,q1d,length
      integer*8 offset
      parameter(dimn   = 2)
      parameter(p1d    = 4)
      parameter(q1d    = 4)
      parameter(length = q1d**dimn)

      real*8 uu(length)
      real*8 vv(length)

      character arg*32

      call getarg(1,arg)
      call ceedinit(trim(arg)//char(0),ceed,err)

      call ceedvectorcreate(ceed,length,u,err)
      call ceedvectorcreate(ceed,length,v,err)

      do i=1,length
        uu(i)=1.0
      enddo
      call ceedvectorsetarray(u,ceed_mem_host,ceed_use_pointer,uu,err)

      call ceedbasiscreatetensorh1lagrange(ceed,dimn,1,p1d,q1d,&
     & ceed_gauss_lobatto,b,err)

      call ceedbasisapply(b,1,ceed_notranspose,ceed_eval_interp,u,v,err)

      call ceedvectorgetarrayread(v,ceed_mem_host,vv,offset,err)
      do i=1,length
        if (abs(vv(i+offset)-1.) > 1.D-15) then
          write(*,*) 'v(',i,'=',vv(i+offset),' not eqaul to 1.0'
        endif
      enddo
      call ceedvectorrestorearrayread(v,vv,offset,err)

      call ceedbasisdestroy(b,err)
      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
