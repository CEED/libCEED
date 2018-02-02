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
      integer erestrictx,erestrictu
      integer bx,bu
      integer qf_setup,qf_mass
      integer op_setup,op_mass
      integer qdata,x,u,v
      integer nelem,p,q
      parameter(nelem=5)
      parameter(p=5)
      parameter(q=8)
      integer nx,nu
      parameter(nx=nelem+1)
      parameter(nu=nelem*(p-1)+1)
      integer indx(nelem*2)
      integer indu(nelem*p)
      real*8 arrx(Nx)
      character arg*32

      external setup,mass

      call getarg(1,arg)
      call ceedinit(trim(arg)//char(0),ceed,err)

c      call ceedqfunctioncreateinterior(ceed,1,1,8,ceed_eval_weight,
c     $  ceed_eval_none,setup,'t20-qfunction-f.f:setup',qf_setup,err)
c      call ceedqfunctioncreateinterior(ceed,1,1,8,ceed_eval_interp,
c     $  ceed_eval_interp,mass,'t20-qfunction-f.f:mass',qf_mass,err)
c
c      do i=0,q-1
c        x=2.0*i/(q-1)-1
c        w(i+1)=1-x*x
c        u(i+1)=2+3*x+5*x*x
c        v(i+1)=w(i+1)*u(i+1)
c      enddo
c
c      call ceedqfunctionapply(qf_setup,qdata,q,w,%val(0),err)
c      call ceedqfunctionapply(qf_mass,qdata,q,u,vv,err)
c
c      do i=1,q
c        if (abs(v(i)-vv(i)) > 1.0D-15) then
c          write(*,*) 'v(i)=',v(i),', vv(i)=',vv(i)
c        endif
c      enddo
c
c      call ceedqfunctiondestroy(qf_setup,err)
c      call ceedqfunctiondestroy(qf_mass,err)
c      call ceeddestroy(ceed,err) 
      end
c-----------------------------------------------------------------------
