c-----------------------------------------------------------------------
      subroutine setup(ctx,qdata,q,u,v,ierr)
      real*8 ctx(1)
      real*8 qdata(1)
      real*8 u(1)
      real*8 v(1)
      integer q,ierr

      do i=1,q
        qdata(i)=u(i)
      enddo

      ierr=0
      end
c-----------------------------------------------------------------------
      subroutine mass(ctx,qdata,q,u,v,ierr)
      real*8 ctx(1)
      real*8 qdata(1)
      real*8 u(1)
      real*8 v(1)
      integer q,ierr

      do i=1,q
        v(i)=qdata(i)*u(i)
      enddo

      ierr=0
      end
c-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer qf_setup,qf_mass
      integer q,i
      parameter(q=8)
      real*8 qdata(q)
      real*8 w(q)
      real*8 u(q)
      real*8 v(q)
      real*8 vv(q)
      real*8 x
      character arg*32

      external setup,mass

      call getarg(1,arg)
      call ceedinit(trim(arg)//char(0),ceed,err)

      call ceedqfunctioncreateinterior(ceed,1,1,8,ceed_eval_weight,
     $  ceed_eval_none,setup,'t20-qfunction-f.f:setup',qf_setup,err)
      call ceedqfunctioncreateinterior(ceed,1,1,8,ceed_eval_interp,
     $  ceed_eval_interp,mass,'t20-qfunction-f.f:mass',qf_mass,err)

      do i=0,q-1
        x=2.0*i/(q-1)-1
        w(i+1)=1-x*x
        u(i+1)=2+3*x+5*x*x
        v(i+1)=w(i+1)*u(i+1)
      enddo

      call ceedqfunctionapply(qf_setup,qdata,q,w,%val(0),err)
      call ceedqfunctionapply(qf_mass,qdata,q,u,vv,err)

      do i=1,q
        if (abs(v(i)-vv(i)) > 1.0D-15) then
          write(*,*) 'v(i)=',v(i),', vv(i)=',vv(i)
        endif
      enddo

      call ceedqfunctiondestroy(qf_setup,err)
      call ceedqfunctiondestroy(qf_mass,err)
      call ceeddestroy(ceed,err)
      end
c-----------------------------------------------------------------------
