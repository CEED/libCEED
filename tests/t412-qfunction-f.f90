!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer u,v
      integer qf
      integer q,s,i
      parameter(q=8)
      parameter(s=3)
      real*8 uu(q*s)
      real*8 vv(q*s)
      character arg*32
      integer*8 uoffset,voffset

      call getarg(1,arg)
      call ceedinit(trim(arg)//char(0),ceed,err)

      call ceedqfunctioncreateidentity(ceed,s,qf,err)

      do i=0,q*s-1
        uu(i+1)=i*i
      enddo

      call ceedvectorcreate(ceed,q*s,u,err)
      uoffset=0
      call ceedvectorsetarray(u,ceed_mem_host,ceed_use_pointer,uu,uoffset,err)
      call ceedvectorcreate(ceed,q*s,v,err)
      call ceedvectorsetvalue(v,0.d0,err)

      call ceedqfunctionapply(qf,q,u,ceed_null,ceed_null,ceed_null,&
             &ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,&
             &ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,&
             &v,ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,&
             &ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,&
             &ceed_null,ceed_null,ceed_null,ceed_null,err)

      call ceedvectorgetarrayread(v,ceed_mem_host,vv,voffset,err)
      do i=1,q*s
        if (abs(vv(i+voffset)-(i-1)*(i-1)) > 1.0D-14) then
! LCOV_EXCL_START
          write(*,*) 'v(i)=',vv(i+voffset),', u(i)=',(i-1)*(i-1)
! LCOV_EXCL_STOP
        endif
      enddo
      call ceedvectorrestorearrayread(v,vv,voffset,err)

      call ceedvectordestroy(u,err)
      call ceedvectordestroy(v,err)
      call ceedqfunctiondestroy(qf,err)
      call ceeddestroy(ceed,err)
      end
!-----------------------------------------------------------------------
