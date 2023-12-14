!-----------------------------------------------------------------------
!
! Header with QFunctions
!
      include 't502-operator-f.h'
!-----------------------------------------------------------------------
      program test
      implicit none
      include 'ceed/fortran.h'

      integer ceed,err,i,j
      integer stridesu_small(3),stridesu_large(3)
      integer erestrictx,erestrictu
      integer erestrictui_small,erestrictui_large
      integer bx_small,bu_small,bx_large,bu_large
      integer qf_setup,qf_mass
      integer op_setup_small,op_mass_small,op_setup_large,op_mass_large
      integer qdata_small,qdata_large,x,u,v
      integer nelem,p,q,scale
      parameter(nelem=15)
      parameter(p=5)
      parameter(q=8)
      parameter(scale=3)
      integer nx,nu
      parameter(nx=nelem+1)
      parameter(nu=nelem*(p-1)+1)
      integer indx(nelem*2)
      integer indu(nelem*p)
      real*8 arrx(nx)
      integer*8 voffset,xoffset

      real*8 hu(nu*2),hv(nu*2)
      real*8 total1,total2

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

      call ceedelemrestrictioncreate(ceed,nelem,2,1,1,nx,ceed_mem_host,&
     & ceed_use_pointer,indx,erestrictx,err)

      do i=0,nelem-1
        do j=0,p-1
          indu(p*i+j+1)=2*(i*(p-1)+j)
        enddo
      enddo

      call ceedelemrestrictioncreate(ceed,nelem,p,2,1,2*nu,ceed_mem_host,&
     & ceed_use_pointer,indu,erestrictu,err)
      stridesu_small=[1,q,q]
      call ceedelemrestrictioncreatestrided(ceed,nelem,q,1,q*nelem,&
     & stridesu_small,erestrictui_small,err)
      stridesu_large=[1,q*scale,q*scale]
      call ceedelemrestrictioncreatestrided(ceed,nelem,q*scale,1,&
     & q*nelem*scale,stridesu_large,erestrictui_large,err)

      call ceedbasiscreatetensorh1lagrange(ceed,1,1,2,q,ceed_gauss,bx_small,err)
      call ceedbasiscreatetensorh1lagrange(ceed,1,2,p,q,ceed_gauss,bu_small,err)
      call ceedbasiscreatetensorh1lagrange(ceed,1,1,2,q*scale,&
     & ceed_gauss,bx_large,err)
      call ceedbasiscreatetensorh1lagrange(ceed,1,2,p,q*scale,&
     & ceed_gauss,bu_large,err)

! Common QFunctions

      call ceedqfunctioncreateinterior(ceed,1,setup,&
     &SOURCE_DIR&
     &//'t502-operator.h:setup'//char(0),qf_setup,err)
      call ceedqfunctionaddinput(qf_setup,'weight',1,ceed_eval_weight,err)
      call ceedqfunctionaddinput(qf_setup,'x',1,ceed_eval_grad,err)
      call ceedqfunctionaddoutput(qf_setup,'rho',1,ceed_eval_none,err)

      call ceedqfunctioncreateinterior(ceed,1,mass,&
     &SOURCE_DIR&
     &//'t502-operator.h:mass'//char(0),qf_mass,err)
      call ceedqfunctionaddinput(qf_mass,'rho',1,ceed_eval_none,err)
      call ceedqfunctionaddinput(qf_mass,'u',2,ceed_eval_interp,err)
      call ceedqfunctionaddoutput(qf_mass,'v',2,ceed_eval_interp,err)

      call ceedvectorcreate(ceed,nx,x,err)
      xoffset=0
      call ceedvectorsetarray(x,ceed_mem_host,ceed_use_pointer,arrx,xoffset,err)

! Small operator

      call ceedoperatorcreate(ceed,qf_setup,ceed_qfunction_none,&
     & ceed_qfunction_none,op_setup_small,err)
      call ceedoperatorcreate(ceed,qf_mass,ceed_qfunction_none,&
     & ceed_qfunction_none,op_mass_small,err)

      call ceedvectorcreate(ceed,nelem*q,qdata_small,err)

      call ceedoperatorsetfield(op_setup_small,'weight',&
     & ceed_elemrestriction_none,bx_small,ceed_vector_none,err)
      call ceedoperatorsetfield(op_setup_small,'x',erestrictx,&
     & bx_small,ceed_vector_active,err)
      call ceedoperatorsetfield(op_setup_small,'rho',erestrictui_small,&
     ceed_basis_none,ceed_vector_active,err)
      call ceedoperatorsetfield(op_mass_small,'rho',erestrictui_small,&
     ceed_basis_none,qdata_small,err)
      call ceedoperatorsetfield(op_mass_small,'u',erestrictu,&
     & bu_small,ceed_vector_active,err)
      call ceedoperatorsetfield(op_mass_small,'v',erestrictu,&
     & bu_small,ceed_vector_active,err)

! Large operator

      call ceedoperatorcreate(ceed,qf_setup,ceed_qfunction_none,&
     & ceed_qfunction_none,op_setup_large,err)
      call ceedoperatorcreate(ceed,qf_mass,ceed_qfunction_none,&
     & ceed_qfunction_none,op_mass_large,err)

      call ceedvectorcreate(ceed,nelem*q*scale,qdata_large,err)

      call ceedoperatorsetfield(op_setup_large,'weight',&
     & ceed_elemrestriction_none,bx_large,ceed_vector_none,err)
      call ceedoperatorsetfield(op_setup_large,'x',erestrictx,&
     & bx_large,ceed_vector_active,err)
      call ceedoperatorsetfield(op_setup_large,'rho',erestrictui_large,&
     ceed_basis_none,ceed_vector_active,err)
      call ceedoperatorsetfield(op_mass_large,'rho',erestrictui_large,&
     ceed_basis_none,qdata_large,err)
      call ceedoperatorsetfield(op_mass_large,'u',erestrictu,&
     & bu_large,ceed_vector_active,err)
      call ceedoperatorsetfield(op_mass_large,'v',erestrictu,&
     & bu_large,ceed_vector_active,err)

! Setup U, V

      call ceedvectorcreate(ceed,2*nu,u,err)
      call ceedvectorgetarraywrite(u,ceed_mem_host,hu,voffset,err)
      do i=1,nu
        hu(voffset+2*i-1)=1.
        hu(voffset+2*i)=2.
      enddo
      call ceedvectorrestorearray(u,hu,voffset,err)
      call ceedvectorcreate(ceed,2*nu,v,err)

! Small apply

      call ceedoperatorapply(op_setup_small,x,qdata_small,&
     & ceed_request_immediate,err)
      call ceedoperatorapply(op_mass_small,u,v,ceed_request_immediate,err)

      call ceedvectorgetarrayread(v,ceed_mem_host,hv,voffset,err)
      total1=0.
      total2=0.
      do i=1,nu
        total1=total1+hv(voffset+2*i-1)
        total2=total2+hv(voffset+2*i)
      enddo
      if (abs(total1-1.)>1.0d-10) then
! LCOV_EXCL_START
        write(*,*) 'Computed Area: ',total1,' != True Area: 1.0'
! LCOV_EXCL_STOP
      endif
      if (abs(total2-2.)>1.0d-10) then
! LCOV_EXCL_START
        write(*,*) 'Computed Area: ',total2,' != True Area: 2.0'
! LCOV_EXCL_STOP
      endif
      call ceedvectorrestorearrayread(v,hv,voffset,err)

! Large apply

      call ceedoperatorapply(op_setup_large,x,qdata_large,&
     & ceed_request_immediate,err)
      call ceedoperatorapply(op_mass_large,u,v,ceed_request_immediate,err)

      call ceedvectorgetarrayread(v,ceed_mem_host,hv,voffset,err)
      total1=0.
      total2=0.
      do i=1,nu
        total1=total1+hv(voffset+2*i-1)
        total2=total2+hv(voffset+2*i)
      enddo
      if (abs(total1-1.)>1.0d-10) then
! LCOV_EXCL_START
        write(*,*) 'Computed Area: ',total1,' != True Area: 1.0'
! LCOV_EXCL_STOP
      endif
      if (abs(total2-2.)>1.0d-10) then
! LCOV_EXCL_START
        write(*,*) 'Computed Area: ',total2,' != True Area: 2.0'
! LCOV_EXCL_STOP
      endif
      call ceedvectorrestorearrayread(v,hv,voffset,err)

      call ceedvectordestroy(qdata_small,err)
      call ceedvectordestroy(qdata_large,err)
      call ceedvectordestroy(x,err)
      call ceedvectordestroy(u,err)
      call ceedvectordestroy(v,err)
      call ceedoperatordestroy(op_mass_small,err)
      call ceedoperatordestroy(op_setup_small,err)
      call ceedoperatordestroy(op_mass_large,err)
      call ceedoperatordestroy(op_setup_large,err)
      call ceedqfunctiondestroy(qf_mass,err)
      call ceedqfunctiondestroy(qf_setup,err)
      call ceedbasisdestroy(bu_small,err)
      call ceedbasisdestroy(bx_small,err)
      call ceedbasisdestroy(bu_large,err)
      call ceedbasisdestroy(bx_large,err)
      call ceedelemrestrictiondestroy(erestrictu,err)
      call ceedelemrestrictiondestroy(erestrictx,err)
      call ceedelemrestrictiondestroy(erestrictui_small,err)
      call ceedelemrestrictiondestroy(erestrictui_large,err)
      call ceeddestroy(ceed,err)
      end
!-----------------------------------------------------------------------

