!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer u,v
      integer qf
      integer q,i
      parameter(q=8)
      real*8 uu(q)
      real*8 vv(q)
      character arg*32
      integer*8 uoffset,voffset

      call getarg(1,arg)
      call ceedinit(trim(arg)//char(0),ceed,err)

      call ceedqfunctioncreateidentity(ceed,1,qf,err)

      do i=0,q-1
        uu(i+1)=i*i
      enddo

      call ceedvectorcreate(ceed,q,u,err)
      uoffset=0
      call ceedvectorsetarray(u,ceed_mem_host,ceed_use_pointer,uu,uoffset,err)
      call ceedvectorcreate(ceed,q,v,err)
      call ceedvectorsetvalue(v,0.d0,err)

      call ceedqfunctionapply(qf,q,u,ceed_null,ceed_null,ceed_null,&
             &ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,&
             &ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,&
             &v,ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,&
             &ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,&
             &ceed_null,ceed_null,ceed_null,ceed_null,err)

      call ceedvectorgetarrayread(v,ceed_mem_host,vv,voffset,err)
      do i=1,q
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
