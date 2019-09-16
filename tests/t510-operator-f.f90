!-----------------------------------------------------------------------
! 
! Header with common subroutine
! 
      include 't320-basis-f.h'
!-----------------------------------------------------------------------
      subroutine setup(ctx,q,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,&
&           u15,u16,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,ierr)
      real*8 ctx
      real*8 u1(1)
      real*8 u2(1)
      real*8 v1(1)
      integer q,ierr

      do i=1,q
        v1(i)=u1(i)*(u2(i+q*0)*u2(i+q*3)-u2(i+q*1)*u2(i+q*2))
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

      integer ceed,err,i
      integer erestrictx,erestrictu,erestrictxi,erestrictui
      integer bx,bu
      integer qf_setup,qf_mass
      integer op_setup,op_mass
      integer qdata,x,u,v
      integer nelem,p,q,d
      integer val,row,col,offset
      parameter(nelem=12)
      parameter(p=6)
      parameter(q=4)
      parameter(d=2)
      integer ndofs,nqpts,nx,ny
      parameter(nx=3)
      parameter(ny=2)
      parameter(ndofs=(nx*2+1)*(ny*2+1))
      parameter(nqpts=nelem*q)
      integer indx(nelem*p)
      real*8 arrx(d*ndofs)
      integer*8 voffset,xoffset

      real*8 qref(d*q)
      real*8 qweight(q)
      real*8 interp(p*q)
      real*8 grad(d*p*q)

      real*8 hv(ndofs)

      character arg*32

      external setup,mass

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)

      do i=0,ndofs-1
        arrx(i+1)=mod(i,(nx*2+1))
        arrx(i+1)=arrx(i+1)*(1.d0/(nx*2.d0))
        val=(i/(nx*2+1))
        arrx(i+1+ndofs)=val*(1.d0/(ny*2.d0))
      enddo
      do i=0,5
        col=mod(i,nx)
        row=i/nx
        offset=col*2+row*(nx*2+1)*2

        indx(i*2*p+1)=2+offset
        indx(i*2*p+2)=9+offset
        indx(i*2*p+3)=16+offset
        indx(i*2*p+4)=1+offset
        indx(i*2*p+5)=8+offset
        indx(i*2*p+6)=0+offset

        indx(i*2*p+7)=14+offset
        indx(i*2*p+8)=7+offset
        indx(i*2*p+9)=0+offset
        indx(i*2*p+10)=15+offset
        indx(i*2*p+11)=8+offset
        indx(i*2*p+12)=16+offset
      enddo

      call ceedelemrestrictioncreate(ceed,nelem,p,ndofs,d,ceed_mem_host,&
     & ceed_use_pointer,indx,erestrictx,err)
      call ceedelemrestrictioncreateidentity(ceed,nelem,p,nelem*p,d,&
     & erestrictxi,err)

      call ceedelemrestrictioncreate(ceed,nelem,p,ndofs,1,ceed_mem_host,&
     & ceed_use_pointer,indx,erestrictu,err)
      call ceedelemrestrictioncreateidentity(ceed,nelem,q,nqpts,1,&
     & erestrictui,err)

      call buildmats(qref,qweight,interp,grad)
      call ceedbasiscreateh1(ceed,ceed_triangle,d,p,q,interp,grad,qref,qweight,&
     & bx,err)
      call buildmats(qref,qweight,interp,grad)
      call ceedbasiscreateh1(ceed,ceed_triangle,1,p,q,interp,grad,qref,qweight,&
     & bu,err)

      call ceedqfunctioncreateinterior(ceed,1,setup,&
     &SOURCE_DIR&
     &//'t510-operator.h:setup'//char(0),qf_setup,err)
      call ceedqfunctionaddinput(qf_setup,'_weight',1,ceed_eval_weight,err)
      call ceedqfunctionaddinput(qf_setup,'x',d*d,ceed_eval_grad,err)
      call ceedqfunctionaddoutput(qf_setup,'rho',1,ceed_eval_none,err)

      call ceedqfunctioncreateinterior(ceed,1,mass,&
     &SOURCE_DIR&
     &//'t510-operator.h:mass'//char(0),qf_mass,err)
      call ceedqfunctionaddinput(qf_mass,'rho',1,ceed_eval_none,err)
      call ceedqfunctionaddinput(qf_mass,'u',1,ceed_eval_interp,err)
      call ceedqfunctionaddoutput(qf_mass,'v',1,ceed_eval_interp,err)

      call ceedoperatorcreate(ceed,qf_setup,ceed_null,ceed_null,op_setup,err)
      call ceedoperatorcreate(ceed,qf_mass,ceed_null,ceed_null,op_mass,err)

      call ceedvectorcreate(ceed,d*ndofs,x,err)
      xoffset=0
      call ceedvectorsetarray(x,ceed_mem_host,ceed_use_pointer,arrx,xoffset,err)
      call ceedvectorcreate(ceed,nqpts,qdata,err)

      call ceedoperatorsetfield(op_setup,'_weight',erestrictxi,&
     & ceed_notranspose,bx,ceed_vector_none,err)
      call ceedoperatorsetfield(op_setup,'x',erestrictx,&
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

      call ceedvectorcreate(ceed,ndofs,u,err)
      call ceedvectorsetvalue(u,0.d0,err)
      call ceedvectorcreate(ceed,ndofs,v,err)
      call ceedoperatorapply(op_mass,u,v,ceed_request_immediate,err)

      call ceedvectorgetarrayread(v,ceed_mem_host,hv,voffset,err)
      do i=1,ndofs
        if (abs(hv(voffset+i))>1.0d-10) then
! LCOV_EXCL_START
          write(*,*) '[',i,'] v ',hv(voffset+i),' != 0.0'
! LCOV_EXCL_STOP
        endif
      enddo
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
