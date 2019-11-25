!-----------------------------------------------------------------------
      subroutine setup(ctx,q,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,&
&           u15,u16,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,ierr)
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
!-----------------------------------------------------------------------
      subroutine mass(ctx,q,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,&
&           u15,u16,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,ierr)
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
!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err,i,j
      integer erestrictx,erestrictu,erestrictxi,erestrictui
      integer bx,bu
      integer qf_setup,qf_mass
      integer op_setup,op_mass
      integer qdata,x,u,v
      integer nelem,p,q
      parameter(nelem=15)
      parameter(p=5)
      parameter(q=8)
      integer nx,nu
      parameter(nx=nelem+1)
      parameter(nu=nelem*(p-1)+1)
      integer indx(nelem*2)
      integer indu(nelem*p)
      real*8 arrx(nx)
      integer*8 voffset,xoffset

      real*8 hv(nu)
      real*8 total

      character arg*32

      external setup,mass

      call getarg(1,arg)
      call ceedinit(trim(arg)//char(0),ceed,err)

      do i=0,nx-1
        arrx(i+1)=i/(nx-1.d0)
      enddo
      do i=0,nelem-1
        indx(2*i+1)=i
        indx(2*i+2)=i+1
      enddo

      call ceedelemrestrictioncreate(ceed,nelem,2,nx,1,ceed_mem_host,&
     & ceed_use_pointer,indx,erestrictx,err)
      call ceedelemrestrictioncreateidentity(ceed,nelem,2,2*nelem,1,&
     & erestrictxi,err)

      do i=0,nelem-1
        do j=0,p-1
          indu(p*i+j+1)=i*(p-1)+j
        enddo
      enddo

      call ceedelemrestrictioncreate(ceed,nelem,p,nu,1,ceed_mem_host,&
     & ceed_use_pointer,indu,erestrictu,err)
      call ceedelemrestrictioncreateidentity(ceed,nelem,q,q*nelem,1,&
     & erestrictui,err)

      call ceedbasiscreatetensorh1lagrange(ceed,1,1,2,q,ceed_gauss,bx,err)
      call ceedbasiscreatetensorh1lagrange(ceed,1,1,p,q,ceed_gauss,bu,err)

      call ceedqfunctioncreateinterior(ceed,1,setup,&
     &SOURCE_DIR&
     &//'t500-operator.h:setup'//char(0),qf_setup,err)
      call ceedqfunctionaddinput(qf_setup,'_weight',1,ceed_eval_weight,err)
      call ceedqfunctionaddinput(qf_setup,'dx',1,ceed_eval_grad,err)
      call ceedqfunctionaddoutput(qf_setup,'rho',1,ceed_eval_none,err)

      call ceedqfunctioncreateinterior(ceed,1,mass,&
     &SOURCE_DIR&
     &//'t500-operator.h:mass'//char(0),qf_mass,err)
      call ceedqfunctionaddinput(qf_mass,'rho',1,ceed_eval_none,err)
      call ceedqfunctionaddinput(qf_mass,'u',1,ceed_eval_interp,err)
      call ceedqfunctionaddoutput(qf_mass,'v',1,ceed_eval_interp,err)

      call ceedoperatorcreate(ceed,qf_setup,ceed_qfunction_none,&
     & ceed_qfunction_none,op_setup,err)
      call ceedoperatorcreate(ceed,qf_mass,ceed_qfunction_none,&
     & ceed_qfunction_none,op_mass,err)

      call ceedvectorcreate(ceed,nx,x,err)
      xoffset=0
      call ceedvectorsetarray(x,ceed_mem_host,ceed_use_pointer,arrx,xoffset,err)
      call ceedvectorcreate(ceed,nelem*q,qdata,err)

      call ceedoperatorsetfield(op_setup,'_weight',erestrictxi,&
     & ceed_notranspose,bx,ceed_vector_none,err)
      call ceedoperatorsetfield(op_setup,'dx',erestrictx,&
     & ceed_notranspose,bx,ceed_vector_active,err)
      call ceedoperatorsetfield(op_setup,'rho',erestrictui,&
     & ceed_notranspose,ceed_basis_collocated,ceed_vector_active,err)
      call ceedoperatorsetfield(op_mass,'rho',erestrictui,&
     & ceed_notranspose,ceed_basis_collocated,qdata,err)
      call ceedoperatorsetfield(op_mass,'u',erestrictu,&
     & ceed_notranspose,bu,ceed_vector_active,err)
      call ceedoperatorsetfield(op_mass,'v',erestrictu,&
     & ceed_notranspose,bu,ceed_vector_active,err)

      call ceedoperatorapply(op_setup,x,qdata,ceed_request_immediate,err)

      call ceedvectorcreate(ceed,nu,u,err)
      call ceedvectorsetvalue(u,1.d0,err)
      call ceedvectorcreate(ceed,nu,v,err)
      call ceedvectorsetvalue(v,0.d0,err)

      call ceedoperatorapplyadd(op_mass,u,v,ceed_request_immediate,err)

      call ceedvectorgetarrayread(v,ceed_mem_host,hv,voffset,err)
      total=0.
      do i=1,nu
        total=total+hv(voffset+i)
      enddo
      if (abs(total-1.)>1.0d-10) then
! LCOV_EXCL_START
        write(*,*) 'Computed Area: ',total,' != True Area: 1.0'
! LCOV_EXCL_STOP
      endif
      call ceedvectorrestorearrayread(v,hv,voffset,err)

      call ceedvectorsetvalue(v,1.d0,err)
      call ceedoperatorapplyadd(op_mass,u,v,ceed_request_immediate,err)

      call ceedvectorgetarrayread(v,ceed_mem_host,hv,voffset,err)
      total=-nu
      do i=1,nu
        total=total+hv(voffset+i)
      enddo
      if (abs(total-1.)>1.0d-10) then
! LCOV_EXCL_START
        write(*,*) 'Computed Area: ',total,' != True Area: 1.0'
! LCOV_EXCL_STOP
      endif
      call ceedvectorrestorearrayread(v,hv,voffset,err)

      call ceedvectordestroy(qdata,err)
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
      call ceedelemrestrictiondestroy(erestrictui,err)
      call ceedelemrestrictiondestroy(erestrictxi,err)
      call ceeddestroy(ceed,err)
      end
!-----------------------------------------------------------------------
