c-----------------------------------------------------------------------
      subroutine setup(ctx,q,u1,u2,u3,u4,u5,u6,u7,
     $  u8,u9,u10,u11,u12,u13,u14,u15,u16,v1,v2,v3,v4,v5,v6,v7,v8,
     $  v9,v10,v11,v12,v13,v14,v15,v16,ierr)
      real*8 ctx
      real*8 u1(1)
      real*8 u2(1)
      real*8 v1(1)
      integer q,ierr

      do i=1,q
        v1(i)=u1(i)*u2(i)
      enddo

      ierr=0
      end
c-----------------------------------------------------------------------
      subroutine mass(ctx,q,u1,u2,u3,u4,u5,u6,u7,
     $  u8,u9,u10,u11,u12,u13,u14,u15,u16,v1,v2,v3,v4,v5,v6,v7,v8,
     $  v9,v10,v11,v12,v13,v14,v15,v16,ierr)
      real*8 ctx
      real*8 u1(1)
      real*8 u2(1)
      real*8 v1(1)
      integer q,ierr

      do i=1,q
        v1(i)=u2(i)*u1(i)
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

      call ceedelemrestrictioncreate(ceed,nelem,2,nx,1,ceed_mem_host,
     $  ceed_use_pointer,indx,erestrictx,err)

      do i=0,nelem-1
        do j=0,p-1
          indu(p*i+j+1)=i*(p-1)+j
        enddo
      enddo

      call ceedelemrestrictioncreate(ceed,nelem,p,nu,1,ceed_mem_host,
     $  ceed_use_pointer,indu,erestrictu,err)

      call ceedbasiscreatetensorh1lagrange(ceed,1,1,2,q,ceed_gauss,
     $  bx,err)
      call ceedbasiscreatetensorh1lagrange(ceed,1,1,p,q,ceed_gauss,
     $  bu,err)

      call ceedqfunctioncreateinterior(ceed,1,setup,
c     __FILE__ should not be more than the 72 characters, -ffree-line-length-none ?
     $__FILE__ 
     $     //':setup'//char(0),qf_setup,err)
c     $  't30-operator-f.f:setup',qf_setup,err)
      call ceedqfunctionaddinput(qf_setup,'_weight',1,
     $  ceed_eval_weight,err)
      call ceedqfunctionaddinput(qf_setup,'x',1,ceed_eval_grad,err)
      call ceedqfunctionaddoutput(qf_setup,'rho',1,
     $  ceed_eval_none,err)

      call ceedqfunctioncreateinterior(ceed,1,mass,
     $__FILE__ 
     $     //':mass'//char(0),qf_mass,err)
c     $  't30-operator-f.f:mass',qf_mass,err)
      call ceedqfunctionaddinput(qf_mass,'rho',1,ceed_eval_none,err)
      call ceedqfunctionaddinput(qf_mass,'u',1,ceed_eval_interp,err)
      call ceedqfunctionaddoutput(qf_mass,'v',1,ceed_eval_interp,err)

      call ceedoperatorcreate(ceed,qf_setup,ceed_null,ceed_null,
     $  op_setup,err)
      call ceedoperatorcreate(ceed,qf_mass,ceed_null,ceed_null,
     $  op_mass,err)

      call ceedvectorcreate(ceed,nx,x,err)
      call ceedvectorsetarray(x,ceed_mem_host,ceed_use_pointer,arrx,err)
      call ceedvectorcreate(ceed,nelem*q,qdata,err)

      call ceedoperatorsetfield(op_setup,'_weight',
     $  ceed_restriction_identity,bx,ceed_vector_none,err)
      call ceedoperatorsetfield(op_setup,'x',erestrictx,bx,
     $  ceed_vector_active,err)
      call ceedoperatorsetfield(op_setup,'rho',
     $  ceed_restriction_identity,ceed_basis_colocated,
     $  ceed_vector_active,err)
      call ceedoperatorsetfield(op_mass,'rho',
     $  ceed_restriction_identity,ceed_basis_colocated,
     $  qdata,err)
      call ceedoperatorsetfield(op_mass,'u',erestrictu,bu,
     $  ceed_vector_active,err)
      call ceedoperatorsetfield(op_mass,'v',erestrictu,bu,
     $  ceed_vector_active,err)

      call ceedoperatorapply(op_setup,x,qdata,
     $  ceed_request_immediate,err)

      call ceedvectorcreate(ceed,nu,u,err)
      call ceedvectorcreate(ceed,nu,v,err)
      call ceedoperatorapply(op_mass,u,v,ceed_request_immediate,err)

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
