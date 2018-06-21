c-----------------------------------------------------------------------
      subroutine setup(ctx,q,u1,u2,u3,u4,u5,u6,u7,
     $  u8,u9,u10,u11,u12,u13,u14,u15,u16,v1,v2,v3,v4,v5,v6,v7,v8,
     $  v9,v10,v11,v12,v13,v14,v15,v16,ierr)
      real*8 ctx(1)
      real*8 u1(8)
      real*8 v1(8)
      integer q,ierr

      do i=1,q
        v1(i)=u1(i)
      enddo

      ierr=0
      end
c-----------------------------------------------------------------------
      subroutine mass(ctx,q,u1,u2,u3,u4,u5,u6,u7,
     $  u8,u9,u10,u11,u12,u13,u14,u15,u16,v1,v2,v3,v4,v5,v6,v7,v8,
     $  v9,v10,v11,v12,v13,v14,v15,v16,ierr)
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

      call ceedqfunctioncreateinterior(ceed,1,setup, 
c     __FILE__ should not be more than the 72 characters, -ffree-line-length-none ?
     $__FILE__ 
     $     //':setup'//char(0),qf_setup,err)
      call ceedqfunctionaddinput(qf_setup,'w', 1,ceed_eval_interp,err)
      call ceedqfunctionaddoutput(qf_setup,'qdata',1,ceed_eval_interp,
     $  err)

      call ceedqfunctioncreateinterior(ceed,1,mass,
     $__FILE__ 
     $  //':mass'//char(0),qf_mass,err)
      call ceedqfunctionaddinput(qf_mass,'qdata',1,ceed_eval_interp,err)
      call ceedqfunctionaddinput(qf_mass,'u',1,ceed_eval_interp,err)
      call ceedqfunctionaddoutput(qf_mass,'v',1,ceed_eval_interp,err)

      do i=0,q-1
        x=2.0*i/(q-1)-1
        w(i+1)=1-x*x
        u(i+1)=2+3*x+5*x*x
        v(i+1)=w(i+1)*u(i+1)
      enddo

      call ceedqfunctionapply(qf_setup,q,w,ceed_null,ceed_null,
     $  ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,
     $  ceed_null,ceed_null,ceed_null,ceed_null,
     $  ceed_null,ceed_null,ceed_null,ceed_null,
     $  qdata,ceed_null,ceed_null,ceed_null,
     $  ceed_null,ceed_null,ceed_null,ceed_null,
     $  ceed_null,ceed_null,ceed_null,ceed_null,
     $  ceed_null,ceed_null,ceed_null,ceed_null,err)
      call ceedqfunctionapply(qf_mass,q,u,qdata,ceed_null,
     $  ceed_null,ceed_null,ceed_null,ceed_null,ceed_null,
     $  ceed_null,ceed_null,ceed_null,ceed_null,
     $  ceed_null,ceed_null,ceed_null,ceed_null,
     $  vv,ceed_null,ceed_null,ceed_null,
     $  ceed_null,ceed_null,ceed_null,ceed_null,
     $  ceed_null,ceed_null,ceed_null,ceed_null,
     $  ceed_null,ceed_null,ceed_null,ceed_null,err)

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
