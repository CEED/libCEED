!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer qdata,j,w,u,v
      integer qf_setup,qf_mass
      integer q,i
      parameter(q=8)
      real*8 jj(q)
      real*8 ww(q)
      real*8 uu(q)
      real*8 vv(q)
      real*8 vvv(q)
      real*8 x
      character arg*32
      integer*8 joffset,uoffset,voffset,woffset

      call getarg(1,arg)
      call ceedinit(trim(arg)//char(0),ceed,err)

      call ceedqfunctioncreateinteriorbyname(ceed,'Mass1DBuild',qf_setup,err)
      call ceedqfunctioncreateinteriorbyname(ceed,'MassApply',qf_mass,err)

      do i=0,q-1
        jj(i+1)=1
        x=2.0*i/(q-1)-1
        ww(i+1)=1-x*x
        uu(i+1)=2+3*x+5*x*x
        vvv(i+1)=ww(i+1)*uu(i+1)
      enddo

      call ceedvectorcreate(ceed,q,j,err)
      joffset=0
      call ceedvectorsetarray(j,ceed_mem_host,ceed_use_pointer,jj,joffset,err)
      call ceedvectorcreate(ceed,q,w,err)
      woffset=0
      call ceedvectorsetarray(w,ceed_mem_host,ceed_use_pointer,ww,woffset,err)
      call ceedvectorcreate(ceed,q,u,err)
      uoffset=0
      call ceedvectorsetarray(u,ceed_mem_host,ceed_use_pointer,uu,uoffset,err)
      call ceedvectorcreate(ceed,q,v,err)
      call ceedvectorsetvalue(v,0.d0,err)
      call ceedvectorcreate(ceed,q,qdata,err)
      call ceedvectorsetvalue(qdata,0.d0,err)

      call ceedqfunctionapply(qf_setup,q,j,w,ceed_null,ceed_null,&
             &ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,&
             &ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,&
             &qdata,ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,&
             &ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,&
             &ceed_null,ceed_null,ceed_null,ceed_null,err)

      call ceedqfunctionapply(qf_mass,q,u,qdata,ceed_null,ceed_null,&
             &ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,&
             &ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,&
             &v,ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,&
             &ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,&
             &ceed_null,ceed_null,ceed_null,ceed_null,err)

      call ceedvectorgetarrayread(v,ceed_mem_host,vv,voffset,err)
      do i=1,q
        if (abs(vv(i+voffset)-vvv(i)) > 1.0D-14) then
! LCOV_EXCL_START
          write(*,*) 'v(i)=',vv(i+voffset),', vv(i)=',vvv(i)
! LCOV_EXCL_STOP
        endif
      enddo
      call ceedvectorrestorearrayread(v,vv,voffset,err)

      call ceedvectordestroy(u,err)
      call ceedvectordestroy(v,err)
      call ceedvectordestroy(w,err)
      call ceedvectordestroy(qdata,err)
      call ceedqfunctiondestroy(qf_setup,err)
      call ceedqfunctiondestroy(qf_mass,err)
      call ceeddestroy(ceed,err)
      end
!-----------------------------------------------------------------------
