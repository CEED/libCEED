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

      integer ceed,err,i,j,k
      integer erestrictx,erestrictu,erestrictxi,erestrictui,erestrictlini
      integer bx,bu
      integer qf_setup,qf_mass
      integer op_setup,op_mass
      integer qdata,x,a,u,v
      integer nelem,p,q,d
      integer row,col,offset
      parameter(nelem=6)
      parameter(p=3)
      parameter(q=4)
      parameter(d=2)
      integer ndofs,nqpts,nx,ny
      parameter(nx=3)
      parameter(ny=2)
      parameter(ndofs=(nx*2+1)*(ny*2+1))
      parameter(nqpts=nelem*q*q)
      integer indx(nelem*p*p)
      real*8 arrx(d*ndofs),aa(nqpts),qq(nqpts),vv(ndofs)
      integer*8 xoffset,aoffset,qoffset,voffset
      real*8 total

      character arg*32

      external setup,mass

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)

! DoF Coordinates
      do i=0,nx*2
        do j=0,ny*2
          arrx(i+j*(nx*2+1)+0*ndofs+1)=1.d0*i/(2*nx)
          arrx(i+j*(nx*2+1)+1*ndofs+1)=1.d0*j/(2*ny)
        enddo
      enddo
      call ceedvectorcreate(ceed,d*ndofs,x,err)
      xoffset=0
      call ceedvectorsetarray(x,ceed_mem_host,ceed_use_pointer,arrx,xoffset,err)

! Qdata Vector
      call ceedvectorcreate(ceed,nqpts,qdata,err)

! Element Setup
      do i=0,nelem-1
        col=mod(i,nx)
        row=i/nx
        offset=col*(p-1)+row*(nx*2+1)*(p-1)
        do j=0,p-1
          do k=0,p-1
            indx(p*(p*i+k)+j+1)=offset+k*(nx*2+1)+j
          enddo
        enddo
      enddo

! Restrictions
      call ceedelemrestrictioncreate(ceed,nelem,p*p,ndofs,d,&
     & ceed_mem_host,ceed_use_pointer,indx,erestrictx,err)
      call ceedelemrestrictioncreateidentity(ceed,nelem,p*p,&
     & nelem*p*p,d,erestrictxi,err)

      call ceedelemrestrictioncreate(ceed,nelem,p*p,ndofs,1,&
     & ceed_mem_host,ceed_use_pointer,indx,erestrictu,err)
      call ceedelemrestrictioncreateidentity(ceed,nelem,q*q,nqpts,&
     & 1,erestrictui,err)

! Bases
      call ceedbasiscreatetensorh1lagrange(ceed,d,d,p,q,ceed_gauss,&
     & bx,err)
      call ceedbasiscreatetensorh1lagrange(ceed,d,1,p,q,ceed_gauss,&
     & bu,err)

! QFunctions
! -- Setup 
      call ceedqfunctioncreateinterior(ceed,1,setup,&
     &SOURCE_DIR&
     &//'t510-operator.h:setup'//char(0),qf_setup,err)
      call ceedqfunctionaddinput(qf_setup,'_weight',1,ceed_eval_weight,err)
      call ceedqfunctionaddinput(qf_setup,'dx',d*d,ceed_eval_grad,err)
      call ceedqfunctionaddoutput(qf_setup,'rho',1,ceed_eval_none,err)
! -- Mass
      call ceedqfunctioncreateinterior(ceed,1,mass,&
     &SOURCE_DIR&
     &//'t510-operator.h:mass'//char(0),qf_mass,err)
      call ceedqfunctionaddinput(qf_mass,'rho',1,ceed_eval_none,err)
      call ceedqfunctionaddinput(qf_mass,'u',1,ceed_eval_interp,err)
      call ceedqfunctionaddoutput(qf_mass,'v',1,ceed_eval_interp,err)

! Operators
! -- Setup 
      call ceedoperatorcreate(ceed,qf_setup,ceed_qfunction_none,&
     & ceed_qfunction_none,op_setup,err)
      call ceedoperatorsetfield(op_setup,'_weight',erestrictxi,&
     & ceed_notranspose,bx,ceed_vector_none,err)
      call ceedoperatorsetfield(op_setup,'dx',erestrictx,&
     & ceed_notranspose,bx,ceed_vector_active,err)
      call ceedoperatorsetfield(op_setup,'rho',erestrictui,&
     & ceed_notranspose,ceed_basis_collocated,ceed_vector_active,err)
! -- Mass
      call ceedoperatorcreate(ceed,qf_mass,ceed_qfunction_none,&
     & ceed_qfunction_none,op_mass,err)
      call ceedoperatorsetfield(op_mass,'rho',erestrictui,&
     & ceed_notranspose,ceed_basis_collocated,qdata,err)
      call ceedoperatorsetfield(op_mass,'u',erestrictu,&
     & ceed_notranspose,bu,ceed_vector_active,err)
      call ceedoperatorsetfield(op_mass,'v',erestrictu,&
     & ceed_notranspose,bu,ceed_vector_active,err)

! Apply Setup Operator
      call ceedoperatorapply(op_setup,x,qdata,ceed_request_immediate,err)

! Assemble QFunction
      call ceedoperatorassemblelinearqfunction(op_mass,a,erestrictlini,&
     & ceed_request_immediate,err)

! Check Output
      call ceedvectorgetarrayread(a,ceed_mem_host,aa,aoffset,err)
      call ceedvectorgetarrayread(qdata,ceed_mem_host,qq,qoffset,err)
      do i=1,nqpts
        if (abs(qq(qoffset+i)-aa(aoffset+i))>1.0d-9) then
! LCOV_EXCL_START
          write(*,*) 'Error: A[',i,'] = ',aa(aoffset+i),' != ',&
     &      qq(qoffset+i)
! LCOV_EXCL_STOP
        endif
      enddo
      call ceedvectorrestorearrayread(a,aa,aoffset,err)
      call ceedvectorrestorearrayread(qdata,qq,qoffset,err)

! Apply original Mass Operator
      call ceedvectorcreate(ceed,ndofs,u,err)
      call ceedvectorsetvalue(u,1.d0,err)
      call ceedvectorcreate(ceed,ndofs,v,err)
      call ceedvectorsetvalue(v,0.d0,err)
      call ceedoperatorapply(op_mass,u,v,ceed_request_immediate,err)

! Check Output
      call ceedvectorgetarrayread(v,ceed_mem_host,vv,voffset,err)
      total=0.
      do i=1,ndofs
        total=total+vv(voffset+i)
      enddo
      if (abs(total-1.)>1.0d-14) then
! LCOV_EXCL_START
        write(*,*) 'Error: True operator computed area = ',total,' != 1.0'
! LCOV_EXCL_STOP
      endif
      call ceedvectorrestorearrayread(v,vv,voffset,err)

! Switch to new qdata
      call ceedvectorgetarrayread(a,ceed_mem_host,aa,aoffset,err)
      call ceedvectorsetarray(qdata,ceed_mem_host,ceed_copy_values,aa,&
     & aoffset,err)
      call ceedvectorrestorearrayread(a,aa,aoffset,err)

! Apply new Mass Operator
      call ceedvectorsetvalue(v,0.d0,err)
      call ceedoperatorapply(op_mass,u,v,ceed_request_immediate,err)

! Check Output
      call ceedvectorgetarrayread(v,ceed_mem_host,vv,voffset,err)
      total=0.
      do i=1,ndofs
        total=total+vv(voffset+i)
      enddo
      if (abs(total-1.)>1.0d-10) then
! LCOV_EXCL_START
        write(*,*) 'Error: Assembled operator computed area = ',total,' != 1.0'
! LCOV_EXCL_STOP
      endif
      call ceedvectorrestorearrayread(v,vv,voffset,err)

! Cleanup
      call ceedqfunctiondestroy(qf_setup,err)
      call ceedqfunctiondestroy(qf_mass,err)
      call ceedoperatordestroy(op_setup,err)
      call ceedoperatordestroy(op_mass,err)
      call ceedelemrestrictiondestroy(erestrictu,err)
      call ceedelemrestrictiondestroy(erestrictx,err)
      call ceedelemrestrictiondestroy(erestrictui,err)
      call ceedelemrestrictiondestroy(erestrictxi,err)
      call ceedelemrestrictiondestroy(erestrictlini,err)
      call ceedbasisdestroy(bu,err)
      call ceedbasisdestroy(bx,err)
      call ceedvectordestroy(x,err)
      call ceedvectordestroy(a,err)
      call ceedvectordestroy(u,err)
      call ceedvectordestroy(v,err)
      call ceedvectordestroy(qdata,err)
      call ceeddestroy(ceed,err)
      end
!-----------------------------------------------------------------------
