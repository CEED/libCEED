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
      integer qf_setup,qf_mass
      integer ctx
      integer ctxsize
      parameter(ctxsize=5)
      real*8 ctxdata(5)

      character arg*32
      integer*8 coffset

! LCOV_EXCL_START
      external setup,mass
! LCOV_EXCL_STOP

      ctxdata=(/1.d0,2.d0,3.d0,4.d0,5.d0/)

      call getarg(1,arg)
      call ceedinit(trim(arg)//char(0),ceed,err)

      call ceedqfunctioncreateinterior(ceed,1,setup,&
     &SOURCE_DIR&
     &//'t400-qfunction.h:setup'//char(0),qf_setup,err)
      call ceedqfunctionaddinput(qf_setup,'w', 1,ceed_eval_weight,err)
      call ceedqfunctionaddoutput(qf_setup,'qdata',1,ceed_eval_none,err)

      call ceedqfunctioncreateinterior(ceed,1,mass,&
     &SOURCE_DIR&
     &//'t400-qfunction.h:mass'//char(0),qf_mass,err)
      call ceedqfunctionaddinput(qf_mass,'qdata',1,ceed_eval_none,err)
      call ceedqfunctionaddinput(qf_mass,'u',1,ceed_eval_interp,err)
      call ceedqfunctionaddoutput(qf_mass,'v',1,ceed_eval_interp,err)

      call ceedqfunctionview(qf_setup,err)
      call ceedqfunctionview(qf_mass,err)

      call ceedqfunctioncontextcreate(ceed,ctx,err)
      coffset=0
      call ceedqfunctioncontextsetdata(ctx,ceed_mem_host,ceed_use_pointer,ctxsize,&
     & ctxdata,coffset,err)
      call ceedqfunctioncontextview(ctx,err)

      call ceedqfunctiondestroy(qf_setup,err)
      call ceedqfunctiondestroy(qf_mass,err)
      call ceeddestroy(ceed,err)
      end
!-----------------------------------------------------------------------
