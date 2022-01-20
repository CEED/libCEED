!-----------------------------------------------------------------------
!
! Header with QFunctions
! 
      include 't500-operator-f.h'
!-----------------------------------------------------------------------
      program test
      implicit none
      include 'ceed/fortran.h'

      integer ceed,err,i,j
      integer stridesu(3)
      integer erestrictx,erestrictu,erestrictui
      integer bx,bu
      integer qf_setup,qf_mass
      integer op_setup,op_mass
      integer x,qdata
      integer nelem,p,q
      parameter(nelem=15)
      parameter(p=5)
      parameter(q=8)
      integer nx,nu
      parameter(nx=nelem+1)
      parameter(nu=nelem*(p-1)+1)
      integer indx(nelem*2)
      integer indu(nelem*p)

      character arg*32

! LCOV_EXCL_START
      external setup,mass
! LCOV_EXCL_STOP

      call getarg(1,arg)
      call ceedinit(trim(arg)//char(0),ceed,err)

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
      stridesu=[1,q,q]
      call ceedelemrestrictioncreatestrided(ceed,nelem,q,1,q*nelem,stridesu,&
     & erestrictui,err)

      call ceedbasiscreatetensorh1lagrange(ceed,1,1,2,q,ceed_gauss,bx,err)
      call ceedbasiscreatetensorh1lagrange(ceed,1,2,p,q,ceed_gauss,bu,err)

      call ceedqfunctioncreateinterior(ceed,1,setup,&
     &SOURCE_DIR&
     &//'t500-operator.h:setup'//char(0),qf_setup,err)
      call ceedqfunctionaddinput(qf_setup,'weight',1,ceed_eval_weight,err)
      call ceedqfunctionaddinput(qf_setup,'dx',1,ceed_eval_grad,err)
      call ceedqfunctionaddoutput(qf_setup,'rho',1,ceed_eval_none,err)

      call ceedqfunctioncreateinterior(ceed,1,mass,&
     &SOURCE_DIR&
     &//'t500-operator.h:mass'//char(0),qf_mass,err)
      call ceedqfunctionaddinput(qf_mass,'rho',1,ceed_eval_none,err)
      call ceedqfunctionaddinput(qf_mass,'u',2,ceed_eval_interp,err)
      call ceedqfunctionaddoutput(qf_mass,'v',2,ceed_eval_interp,err)

      call ceedoperatorcreate(ceed,qf_setup,ceed_qfunction_none,&
     & ceed_qfunction_none,op_setup,err)
      call ceedoperatorcreate(ceed,qf_mass,ceed_qfunction_none,&
     & ceed_qfunction_none,op_mass,err)

      call ceedvectorcreate(ceed,nx,x,err)
      call ceedvectorcreate(ceed,nelem*q,qdata,err)

      call ceedoperatorsetfield(op_setup,'weight',ceed_elemrestriction_none,&
     & bx,ceed_vector_none,err)
      call ceedoperatorsetfield(op_setup,'dx',erestrictx,bx,&
     & ceed_vector_active,err)
      call ceedoperatorsetfield(op_setup,'rho',erestrictui,&
     & ceed_basis_collocated,ceed_vector_active,err)
      call ceedoperatorsetfield(op_mass,'rho',erestrictui,&
     & ceed_basis_collocated,qdata,err)
      call ceedoperatorsetfield(op_mass,'u',erestrictu,bu,&
     & ceed_vector_active,err)
      call ceedoperatorsetfield(op_mass,'v',erestrictu,bu,&
     & ceed_vector_active,err)

      call ceedoperatorview(op_setup,err)
      call ceedoperatorview(op_mass,err)

      call ceedvectordestroy(qdata,err)
      call ceedoperatordestroy(op_mass,err)
      call ceedoperatordestroy(op_setup,err)
      call ceedqfunctiondestroy(qf_mass,err)
      call ceedqfunctiondestroy(qf_setup,err)
      call ceedbasisdestroy(bu,err)
      call ceedbasisdestroy(bx,err)
      call ceedelemrestrictiondestroy(erestrictu,err)
      call ceedelemrestrictiondestroy(erestrictx,err)
      call ceedelemrestrictiondestroy(erestrictui,err)
      call ceeddestroy(ceed,err)
      end
!-----------------------------------------------------------------------
