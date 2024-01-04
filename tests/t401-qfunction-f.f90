!-----------------------------------------------------------------------
!
! Header with QFunctions
!
      include 't401-qfunction-f.h'
!-----------------------------------------------------------------------
      program test
      implicit none
      include 'ceed/fortran.h'

      integer ceed,err
      integer qdata,w,u,v
      integer qf_setup,qf_mass
      integer ctx
      integer q,i
      parameter(q=8)
      real*8 ww(q)
      real*8 uu(q)
      real*8 vv(q)
      real*8 vvv(q)
      integer ctxsize
      parameter(ctxsize=5)
      real*8 ctxdata(5)
      real*8 x
      character arg*32
      integer*8 uoffset,voffset,woffset,coffset

      external setup,mass

      ctxdata=(/1.d0,2.d0,3.d0,4.d0,5.d0/)

      call getarg(1,arg)
      call ceedinit(trim(arg)//char(0),ceed,err)

      call ceedqfunctioncreateinterior(ceed,1,setup,&
     &SOURCE_DIR&
     &//'t401-qfunction.h:setup'//char(0),qf_setup,err)
      call ceedqfunctionaddinput(qf_setup,'w', 1,ceed_eval_weight,err)
      call ceedqfunctionaddoutput(qf_setup,'qdata',1,ceed_eval_none,err)

      call ceedqfunctioncreateinterior(ceed,1,mass,&
     &SOURCE_DIR&
     &//'t401-qfunction.h:mass'//char(0),qf_mass,err)
      call ceedqfunctionaddinput(qf_mass,'qdata',1,ceed_eval_none,err)
      call ceedqfunctionaddinput(qf_mass,'u',1,ceed_eval_interp,err)
      call ceedqfunctionaddoutput(qf_mass,'v',1,ceed_eval_interp,err)

      call ceedqfunctioncontextcreate(ceed,ctx,err)
      coffset=0
      call ceedqfunctioncontextsetdata(ctx,ceed_mem_host,ceed_use_pointer,ctxsize,&
     & ctxdata,coffset,err)
      call ceedqfunctionsetcontext(qf_mass,ctx,err)

      do i=0,q-1
        x=2.0*i/(q-1)-1
        ww(i+1)=1-x*x
        uu(i+1)=2+3*x+5*x*x
        vvv(i+1)=ww(i+1)*uu(i+1)
      enddo

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

      call ceedqfunctionapply(qf_setup,q,w,ceed_null,ceed_null,ceed_null,&
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
        if (abs(vv(i+voffset)-ctxdata(5)*vvv(i)) > 1.0D-14) then
! LCOV_EXCL_START
          write(*,*) 'v(i)=',vv(i+voffset),', 5*vv(i)=',ctxdata(5)*vvv(i)
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
      call ceedqfunctioncontextdestroy(ctx,err)
      call ceeddestroy(ceed,err)
      end
!-----------------------------------------------------------------------
