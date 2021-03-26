!-----------------------------------------------------------------------
!
! Header with QFunctions
! 
      include 't531-operator-f.h'
!-----------------------------------------------------------------------
      program test
      implicit none
      include 'ceed/fortran.h'

      integer ceed,err,i,j,k
      integer stridesu(3),stridesqd(3)
      integer erestrictx,erestrictu,erestrictui
      integer erestrictqi,erestrictlini
      integer bx,bu
      integer qf_setup,qf_diff,qf_diff_lin
      integer op_setup,op_diff,op_diff_lin
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
      real*8 arrx(d*ndofs),vv(ndofs)
      integer*8 xoffset,voffset

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
      call ceedelemrestrictioncreate(ceed,nelem,p*p,d,ndofs,d*ndofs,&
     & ceed_mem_host,ceed_use_pointer,indx,erestrictx,err)

      call ceedelemrestrictioncreate(ceed,nelem,p*p,1,1,ndofs,&
     & ceed_mem_host,ceed_use_pointer,indx,erestrictu,err)
      stridesu=[1,q*q,q*q]
      call ceedelemrestrictioncreatestrided(ceed,nelem,q*q,1,nqpts,&
     & stridesu,erestrictui,err)

      stridesqd=[1,q*q,q*q*d*(d+1)/2]
      call ceedelemrestrictioncreatestrided(ceed,nelem,q*q,d*(d+1)/2,&
     & d*(d+1)/2*nqpts,stridesqd,erestrictqi,err)

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
     & bx,ceed_vector_active,err)
      call ceedoperatorsetfield(op_setup,'_weight',ceed_elemrestriction_none,&
     & bx,ceed_vector_none,err)
      call ceedoperatorsetfield(op_setup,'qdata',erestrictqi,&
     & ceed_basis_collocated,ceed_vector_active,err)

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
     & bu,ceed_vector_active,err)
      call ceedoperatorsetfield(op_diff,'qdata',erestrictqi,&
     & ceed_basis_collocated,qdata,err)
      call ceedoperatorsetfield(op_diff,'dv',erestrictu,&
     & bu,ceed_vector_active,err)

! Apply original Poisson Operator
      call ceedvectorcreate(ceed,ndofs,u,err)
      call ceedvectorsetvalue(u,1.d0,err)
      call ceedvectorcreate(ceed,ndofs,v,err)
      call ceedvectorsetvalue(v,0.d0,err)
      call ceedoperatorapply(op_diff,u,v,ceed_request_immediate,err)

! Check Output
      call ceedvectorgetarrayread(v,ceed_mem_host,vv,voffset,err)
      do i=1,ndofs
      if (abs(vv(voffset+i))>1.0d-14) then
! LCOV_EXCL_START
        write(*,*) 'Error: Operator computed v[i] = ',vv(voffset+i),' != 0.0'
! LCOV_EXCL_STOP
      endif
      enddo
      call ceedvectorrestorearrayread(v,vv,voffset,err)

! Assemble QFunction
      call ceedoperatorlinearassembleqfunction(op_diff,a,erestrictlini,&
     & ceed_request_immediate,err)

! QFunction - apply linearized
      call ceedqfunctioncreateinterior(ceed,1,diff_lin,&
     &SOURCE_DIR&
     &//'t531-operator.h:diff_lin'//char(0),qf_diff_lin,err)
      call ceedqfunctionaddinput(qf_diff_lin,'du',d,ceed_eval_grad,err)
      call ceedqfunctionaddinput(qf_diff_lin,'qdata',d*d,ceed_eval_none,err)
      call ceedqfunctionaddoutput(qf_diff_lin,'dv',d,ceed_eval_grad,err)

! Operator - apply linearized
      call ceedoperatorcreate(ceed,qf_diff_lin,ceed_qfunction_none,&
     & ceed_qfunction_none,op_diff_lin,err)
      call ceedoperatorsetfield(op_diff_lin,'du',erestrictu,&
     & bu,ceed_vector_active,err)
      call ceedoperatorsetfield(op_diff_lin,'qdata',erestrictlini,&
     & ceed_basis_collocated,a,err)
      call ceedoperatorsetfield(op_diff_lin,'dv',erestrictu,&
     & bu,ceed_vector_active,err)

! Apply linearized Poisson Operator
      call ceedvectorsetvalue(v,0.d0,err)
      call ceedoperatorapply(op_diff_lin,u,v,ceed_request_immediate,err)

! Check Output
      call ceedvectorgetarrayread(v,ceed_mem_host,vv,voffset,err)
      do i=1,ndofs
      if (abs(vv(voffset+i))>1.0d-14) then
! LCOV_EXCL_START
        write(*,*) 'Error: Linearized operator computed v[i] = ',vv(voffset+i),&
     &   ' != 0.0'
! LCOV_EXCL_STOP
      endif
      enddo
      call ceedvectorrestorearrayread(v,vv,voffset,err)

! Cleanup
      call ceedqfunctiondestroy(qf_setup,err)
      call ceedqfunctiondestroy(qf_diff,err)
      call ceedqfunctiondestroy(qf_diff_lin,err)
      call ceedoperatordestroy(op_setup,err)
      call ceedoperatordestroy(op_diff,err)
      call ceedoperatordestroy(op_diff_lin,err)
      call ceedelemrestrictiondestroy(erestrictu,err)
      call ceedelemrestrictiondestroy(erestrictx,err)
      call ceedelemrestrictiondestroy(erestrictui,err)
      call ceedelemrestrictiondestroy(erestrictqi,err)
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
