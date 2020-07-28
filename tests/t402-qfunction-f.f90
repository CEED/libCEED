!-----------------------------------------------------------------------
!
! Header with QFunctions
! 
      include 't401-qfunction-f.h'
!-----------------------------------------------------------------------
      program test
      implicit none
      include 'ceedf.h'

      integer ceed,err
      integer qf_setup,qf_mass
      character arg*32

! LCOV_EXCL_START
      external setup,mass
! LCOV_EXCL_STOP

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

      call ceedqfunctiondestroy(qf_setup,err)
      call ceedqfunctiondestroy(qf_mass,err)
      call ceeddestroy(ceed,err)
      end
!-----------------------------------------------------------------------
