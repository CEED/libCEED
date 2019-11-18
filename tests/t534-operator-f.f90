!-----------------------------------------------------------------------
      subroutine setup(ctx,q,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,&
&           u15,u16,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,ierr)
      real*8 ctx
      real*8 u1(1)
      real*8 u2(1)
      real*8 v1(1)
      real*8 w
      integer q,ierr

      do i=1,q
        w=u2(i)/(u1(i+q*0)*u1(i+q*3)-u1(i+q*1)*u1(i+q*2))
        v1(i+q*0)=w*(u1(i+q*2)*u1(i+q*2)+u1(i+q*3)*u1(i+q*3))
        v1(i+q*1)=-w*(u1(i+q*0)*u1(i+q*2)+u1(i+q*2)*u1(i+q*3))
        v1(i+q*2)=w*(u1(i+q*0)*u1(i+q*0)+u1(i+q*1)*u1(i+q*1))
      enddo

      ierr=0
      end
!-----------------------------------------------------------------------
      subroutine diff(ctx,q,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,&
&           u15,u16,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,ierr)
      real*8 ctx
      real*8 u1(1)
      real*8 u2(1)
      real*8 v1(1)
      real*8 du0,du1
      integer q,ierr

      do i=1,q
        du0=u1(i+q*0)
        du1=u1(i+q*1)
        v1(i+q*0)=u2(i+q*0)*du0+u2(i+q*2)*du1
        v1(i+q*1)=u2(i+q*2)*du0+u2(i+q*1)*du1
      enddo

      ierr=0
      end
!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err,i,j,k
      integer erestrictx,erestrictu,erestrictxi,erestrictui,erestrictqi
      integer bx,bu
      integer qf_setup,qf_diff
      integer op_setup,op_diff
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
      real*8 arrx(d*ndofs),aa(nqpts),uu(ndofs),vv(ndofs),atrue(ndofs)
      integer*8 xoffset,aoffset,uoffset,voffset

      character arg*32

      external setup,diff,diff_lin

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
      call ceedvectorcreate(ceed,nqpts*d*(d+1)/2,qdata,err)

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

      call ceedelemrestrictioncreateidentity(ceed,nelem,q*q,nqpts,&
     & d*(d+1)/2,erestrictqi,err)

! Bases
      call ceedbasiscreatetensorh1lagrange(ceed,d,d,p,q,ceed_gauss,&
     & bx,err)
      call ceedbasiscreatetensorh1lagrange(ceed,d,1,p,q,ceed_gauss,&
     & bu,err)

! QFunction - setup 
      call ceedqfunctioncreateinterior(ceed,1,setup,&
     &SOURCE_DIR&
     &//'t531-operator.h:setup'//char(0),qf_setup,err)
      call ceedqfunctionaddinput(qf_setup,'dx',d*d,ceed_eval_grad,err)
      call ceedqfunctionaddinput(qf_setup,'_weight',1,ceed_eval_weight,err)
      call ceedqfunctionaddoutput(qf_setup,'qdata',d*(d+1)/2,ceed_eval_none,err)

! Operator - setup 
      call ceedoperatorcreate(ceed,qf_setup,ceed_qfunction_none,&
     & ceed_qfunction_none,op_setup,err)
      call ceedoperatorsetfield(op_setup,'dx',erestrictx,&
     & ceed_notranspose,bx,ceed_vector_active,err)
      call ceedoperatorsetfield(op_setup,'_weight',erestrictxi,&
     & ceed_notranspose,bx,ceed_vector_none,err)
      call ceedoperatorsetfield(op_setup,'qdata',erestrictqi,&
     & ceed_notranspose,ceed_basis_collocated,ceed_vector_active,err)

! Apply Setup Operator
      call ceedoperatorapply(op_setup,x,qdata,ceed_request_immediate,err)

! QFunction - apply
      call ceedqfunctioncreateinterior(ceed,1,diff,&
     &SOURCE_DIR&
     &//'t531-operator.h:diff'//char(0),qf_diff,err)
      call ceedqfunctionaddinput(qf_diff,'du',d,ceed_eval_grad,err)
      call ceedqfunctionaddinput(qf_diff,'qdata',d*(d+1)/2,ceed_eval_none,err)
      call ceedqfunctionaddoutput(qf_diff,'dv',d,ceed_eval_grad,err)

! Operator - apply
      call ceedoperatorcreate(ceed,qf_diff,ceed_qfunction_none,&
     & ceed_qfunction_none,op_diff,err)
      call ceedoperatorsetfield(op_diff,'du',erestrictu,&
     & ceed_notranspose,bu,ceed_vector_active,err)
      call ceedoperatorsetfield(op_diff,'qdata',erestrictqi,&
     & ceed_notranspose,ceed_basis_collocated,qdata,err)
      call ceedoperatorsetfield(op_diff,'dv',erestrictu,&
     & ceed_notranspose,bu,ceed_vector_active,err)

! Assemble Diagonal
      call ceedoperatorassemblelineardiagonal(op_diff,a,&
     & ceed_request_immediate,err)

! Manually assemble diagonal
      call ceedvectorcreate(ceed,ndofs,u,err)
      call ceedvectorsetvalue(u,0.d0,err)
      call ceedvectorcreate(ceed,ndofs,v,err)
      do i=1,ndofs
        call ceedvectorgetarray(u,ceed_mem_host,uu,uoffset,err)
        uu(i+uoffset)=1.d0
        if (i>1) then
          uu(i-1+uoffset)=0.d0
        endif
        call ceedvectorrestorearray(u,uu,uoffset,err)

        call ceedoperatorapply(op_diff,u,v,ceed_request_immediate,err)

        call ceedvectorgetarrayread(v,ceed_mem_host,vv,voffset,err)
        atrue(i)=vv(voffset+i)
        call ceedvectorrestorearrayread(v,vv,voffset,err)
      enddo

! Check Output
      call ceedvectorgetarrayread(a,ceed_mem_host,aa,aoffset,err)
      do i=1,ndofs
        if (abs(aa(aoffset+i)-atrue(i))>1.0d-14) then
! LCOV_EXCL_START
          write(*,*) '[',i,'] Error in assembly: ',aa(aoffset+i),' != ',&
     &      atrue(i)
! LCOV_EXCL_STOP
        endif
      enddo
      call ceedvectorrestorearrayread(a,aa,aoffset,err)

! Cleanup
      call ceedqfunctiondestroy(qf_setup,err)
      call ceedqfunctiondestroy(qf_diff,err)
      call ceedoperatordestroy(op_setup,err)
      call ceedoperatordestroy(op_diff,err)
      call ceedelemrestrictiondestroy(erestrictu,err)
      call ceedelemrestrictiondestroy(erestrictx,err)
      call ceedelemrestrictiondestroy(erestrictxi,err)
      call ceedelemrestrictiondestroy(erestrictui,err)
      call ceedelemrestrictiondestroy(erestrictqi,err)
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
