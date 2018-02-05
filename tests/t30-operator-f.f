c-----------------------------------------------------------------------
      subroutine setup(ctx,qdata,q,u,v,ierr)
      real*8 ctx
      real*8 qdata(1)
      real*8 u(1)
      real*8 v(1)
      integer q,ierr

      do i=1,q
        qdata(i)=u(i)*u(i+q*1*1)
      enddo

      ierr=0
      end
c-----------------------------------------------------------------------
      subroutine mass(ctx,qdata,q,u,v,ierr)
      real*8 ctx
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

      integer ceed,err,i,j
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
      real*8 arrx(nx)
      character arg*32

      external setup,mass

      call getarg(1,arg)
      call ceedinit(trim(arg)//char(0),ceed,err)

      do i=0,nx-1
        arrx(i+1)=i/(nx-1)
      enddo
      do i=0,nelem-1
        indx(2*i+1)=i
        indx(2*i+2)=i+1
      enddo

      call ceedelemrestrictioncreate(ceed,nelem,2,nx,ceed_mem_host,
     $  ceed_use_pointer,indx,erestrictx,err)

      do i=0,nelem-1
        do j=0,p-1
          indu(p*i+j+1)=i*(p-1)+j
        enddo
      enddo

      call ceedelemrestrictioncreate(ceed,nelem,p,nu,ceed_mem_host,
     $  ceed_use_pointer,indu,erestrictu,err)

      call ceedbasiscreatetensorh1lagrange(ceed,1,1,2,q,ceed_gauss,
     $  bx,err)
      call ceedbasiscreatetensorh1lagrange(ceed,1,1,p,q,ceed_gauss,
     $  bu,err)

      call ceedqfunctioncreateinterior(ceed,1,1,8,
     $  ior(ceed_eval_grad,ceed_eval_weight),ceed_eval_none,setup,
     $  't30-operator-f.f:setup',qf_setup,err)
      call ceedqfunctioncreateinterior(ceed,1,1,8,ceed_eval_interp,
     $  ceed_eval_interp,mass,'t30-operator-f.f:mass',qf_mass,err)

      call ceedoperatorcreate(ceed,erestrictx,bx,qf_setup,
     $  %val(0),%val(0),op_setup,err)
      call ceedoperatorcreate(ceed,erestrictu,bu,qf_mass,
     $  %val(0),%val(0),op_mass,err)

      call ceedvectorcreate(ceed,nx,x,err)
      call ceedvectorsetarray(x,ceed_mem_host,ceed_use_pointer,arrx,err)
      call ceedoperatorgetqdata(op_setup,qdata,err)
      call ceedoperatorapply(op_setup,qdata,x,%val(0),
     $  ceed_request_immediate,err)

      call ceedvectorcreate(ceed,nu,u,err)
      call ceedvectorcreate(ceed,nu,v,err)
      call ceedoperatorapply(op_mass,qdata,u,v,
     $  ceed_request_immediate,err)

      call ceedvectordestroy(x,err)
      call ceedvectordestroy(u,err)
      call ceedvectordestroy(v,err)
      call ceedoperatordestroy(op_mass,err)
      call ceedoperatordestroy(op_setup,err)
      call ceedqfunctiondestroy(qf_mass,err)
      call ceedqfunctiondestroy(qf_setup,err)
      call ceedbasisdestroy(bu,err)
      call ceedbasisdestroy(bx,err)
      call ceedelemrestrictiondestroy(erestrictu,err)
      call ceedelemrestrictiondestroy(erestrictx,err)
      call ceeddestroy(ceed,err)
      end
c-----------------------------------------------------------------------
