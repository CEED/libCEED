!-----------------------------------------------------------------------
      subroutine setup(ctx,q,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,&
&           u15,u16,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,ierr)
! LCOV_EXCL_START
      real*8 ctx(1)
      real*8 u1(8)
      real*8 v1(8)
      integer q,ierr

      do i=1,q
        v1(i)=u1(i)
      enddo

      ierr=0
      end
! LCOV_EXCL_STOP
!-----------------------------------------------------------------------
      subroutine mass(ctx,q,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,&
&           u15,u16,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,ierr)
! LCOV_EXCL_START
      real*8 ctx(1)
      real*8 u1(8)
      real*8 u2(8)
      real*8 v1(8)
      integer q,ierr

      do i=1,q
        v1(i)=u1(i)*u2(i)
      enddo

      ierr=0
      end
! LCOV_EXCL_STOP
!-----------------------------------------------------------------------
      program test

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
