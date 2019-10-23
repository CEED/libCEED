!-----------------------------------------------------------------------
      subroutine setup_mass(ctx,q,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,&
&           u14,u15,u16,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,&
&           v16,ierr)
      real*8 ctx
      real*8 u1(1)
      real*8 u2(1)
      real*8 v1(1)
      integer q,ierr

      do i=1,q
        v1(i)=u2(i)*(u1(i+q*0)*u1(i+q*3)-u1(i+q*1)*u1(i+q*2))
      enddo

      ierr=0
      end
!-----------------------------------------------------------------------
      subroutine apply(ctx,q,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,&
&           u15,u16,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,ierr)
      real*8 ctx
      real*8 u1(1)
      real*8 u2(1)
      real*8 v1(1)
      integer q,ierr

      do i=1,q
!       mass
        v1(i) = u1(i)*u2(i)
      enddo

      ierr=0
      end
!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err,i
      integer erestrictxi,erestrictui,erestrictqi
      integer bx,bu
      integer qf_setup_mass,qf_apply
      integer op_setup_mass,op_apply,op_inv
      integer qdata_mass,x,u,v
      integer nelem,p,q,d
      parameter(nelem=1)
      parameter(p=4)
      parameter(q=5)
      parameter(d=2)
      integer ndofs,nqpts
      parameter(ndofs=p*p)
      parameter(nqpts=nelem*q*q)
      real*8 arrx(d*nelem*2*2),uu(ndofs)
      integer*8 xoffset,uoffset

      character arg*32

      external setup_mass,apply

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)

! DoF Coordinates
      do i=0,1
      do j=0,1
        arrx(i+1+j*2+0*4)=i
        arrx(i+1+j*2+1*4)=j
      enddo
      enddo
      call ceedvectorcreate(ceed,d*nelem*2*2,x,err)
      xoffset=0
      call ceedvectorsetarray(x,ceed_mem_host,ceed_use_pointer,arrx,xoffset,err)

! Qdata Vector
      call ceedvectorcreate(ceed,nqpts,qdata_mass,err)

! Restrictions
      call ceedelemrestrictioncreateidentity(ceed,nelem,2*2,nelem*2*2,d,&
     & erestrictxi,err)

      call ceedelemrestrictioncreateidentity(ceed,nelem,p*p,ndofs,1,&
     & erestrictui,err)

      call ceedelemrestrictioncreateidentity(ceed,nelem,q*q,nqpts,1,&
     & erestrictqi,err)

! Bases
      call ceedbasiscreatetensorh1lagrange(ceed,d,d,2,q,ceed_gauss,bx,err)
      call ceedbasiscreatetensorh1lagrange(ceed,d,1,p,q,ceed_gauss,bu,err)

! QFunction - setup mass
      call ceedqfunctioncreateinterior(ceed,1,setup_mass,&
     &SOURCE_DIR&
     &//'t540-operator.h:setup_mass'//char(0),qf_setup_mass,err)
      call ceedqfunctionaddinput(qf_setup_mass,'dx',d*d,ceed_eval_grad,err)
      call ceedqfunctionaddinput(qf_setup_mass,'_weight',1,ceed_eval_weight,err)
      call ceedqfunctionaddoutput(qf_setup_mass,'qdata',1,ceed_eval_none,err)

! Operator - setup mass
      call ceedoperatorcreate(ceed,qf_setup_mass,ceed_qfunction_none,&
     & ceed_qfunction_none,op_setup_mass,err)
      call ceedoperatorsetfield(op_setup_mass,'dx',erestrictxi,&
     & ceed_notranspose,bx,ceed_vector_active,err)
      call ceedoperatorsetfield(op_setup_mass,'_weight',erestrictxi,&
     & ceed_notranspose,bx,ceed_vector_none,err)
      call ceedoperatorsetfield(op_setup_mass,'qdata',erestrictqi,&
     & ceed_notranspose,ceed_basis_collocated,ceed_vector_active,err)

! Apply Setup Operators
      call ceedoperatorapply(op_setup_mass,x,qdata_mass,&
     & ceed_request_immediate,err)

! QFunction - apply
      call ceedqfunctioncreateinterior(ceed,1,apply,&
     &SOURCE_DIR&
     &//'t540-operator.h:apply'//char(0),qf_apply,err)
      call ceedqfunctionaddinput(qf_apply,'u',1,ceed_eval_interp,err)
      call ceedqfunctionaddinput(qf_apply,'qdata_mass',1,ceed_eval_none,err)
      call ceedqfunctionaddoutput(qf_apply,'v',1,ceed_eval_interp,err)

! Operator - apply
      call ceedoperatorcreate(ceed,qf_apply,ceed_qfunction_none,&
     & ceed_qfunction_none,op_apply,err)
      call ceedoperatorsetfield(op_apply,'u',erestrictui,&
     & ceed_notranspose,bu,ceed_vector_active,err)
      call ceedoperatorsetfield(op_apply,'qdata_mass',erestrictqi,&
     & ceed_notranspose,ceed_basis_collocated,qdata_mass,err)
      call ceedoperatorsetfield(op_apply,'v',erestrictui,&
     & ceed_notranspose,bu,ceed_vector_active,err)

! Apply original operator
      call ceedvectorcreate(ceed,ndofs,u,err)
      call ceedvectorsetvalue(u,1.d0,err)
      call ceedvectorcreate(ceed,ndofs,v,err)
      call ceedvectorsetvalue(v,0.d0,err)
      call ceedoperatorapply(op_apply,u,v,ceed_request_immediate,err)

! Create FDM element inverse
      call ceedoperatorcreatefdmelementinverse(op_apply,op_inv,&
     & ceed_request_immediate,err)

! Apply FDM element inverse
      call ceedoperatorapply(op_inv,v,u,ceed_request_immediate,err)

! Check Output
      call ceedvectorgetarrayread(u,ceed_mem_host,uu,uoffset,err)
      do i=1,ndofs
        if (abs(uu(uoffset+i)-1.0)>1.0d-14) then
! LCOV_EXCL_START
          write(*,*) '[',i,'] Error in inverse: ',uu(uoffset+i),' != 1.0'
! LCOV_EXCL_STOP
        endif
      enddo
      call ceedvectorrestorearrayread(u,uu,uoffset,err)

! Cleanup
      call ceedqfunctiondestroy(qf_setup_mass,err)
      call ceedqfunctiondestroy(qf_apply,err)
      call ceedoperatordestroy(op_setup_mass,err)
      call ceedoperatordestroy(op_apply,err)
      call ceedoperatordestroy(op_inv,err)
      call ceedelemrestrictiondestroy(erestrictxi,err)
      call ceedelemrestrictiondestroy(erestrictui,err)
      call ceedelemrestrictiondestroy(erestrictqi,err)
      call ceedbasisdestroy(bu,err)
      call ceedbasisdestroy(bx,err)
      call ceedvectordestroy(x,err)
      call ceedvectordestroy(u,err)
      call ceedvectordestroy(v,err)
      call ceedvectordestroy(qdata_mass,err)
      call ceeddestroy(ceed,err)
      end
!-----------------------------------------------------------------------
