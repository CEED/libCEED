!-----------------------------------------------------------------------
!
! Header with QFunctions
! 
      include 't510-operator-f.h'
!-----------------------------------------------------------------------
      program test
      implicit none
      include 'ceed/fortran.h'

      integer ceed,err,i,j,k
      integer stridesu(3)
      integer erestrictx,erestrictu,erestrictui
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
      real*8 arrx(d*ndofs),aa(nqpts),uu(ndofs),vv(ndofs),atrue(ndofs)
      integer*8 xoffset,aoffset,uoffset,voffset

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
      call ceedelemrestrictioncreate(ceed,nelem,p*p,d,ndofs,d*ndofs,&
     & ceed_mem_host,ceed_use_pointer,indx,erestrictx,err)

      call ceedelemrestrictioncreate(ceed,nelem,p*p,1,1,ndofs,&
     & ceed_mem_host,ceed_use_pointer,indx,erestrictu,err)
      stridesu=[1,q*q,q*q]
      call ceedelemrestrictioncreatestrided(ceed,nelem,q*q,1,nqpts,&
     & stridesu,erestrictui,err)

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
      call ceedoperatorsetfield(op_setup,'_weight',ceed_elemrestriction_none,&
     & bx,ceed_vector_none,err)
      call ceedoperatorsetfield(op_setup,'dx',erestrictx,&
     & bx,ceed_vector_active,err)
      call ceedoperatorsetfield(op_setup,'rho',erestrictui,&
     & ceed_basis_collocated,ceed_vector_active,err)
! -- Mass
      call ceedoperatorcreate(ceed,qf_mass,ceed_qfunction_none,&
     & ceed_qfunction_none,op_mass,err)
      call ceedoperatorsetfield(op_mass,'rho',erestrictui,&
     & ceed_basis_collocated,qdata,err)
      call ceedoperatorsetfield(op_mass,'u',erestrictu,&
     & bu,ceed_vector_active,err)
      call ceedoperatorsetfield(op_mass,'v',erestrictu,&
     & bu,ceed_vector_active,err)

! Apply Setup Operator
      call ceedoperatorapply(op_setup,x,qdata,ceed_request_immediate,err)

! Assemble Diagonal
      call ceedvectorcreate(ceed,ndofs,a,err)
      call ceedoperatorlinearassemblediagonal(op_mass,a,&
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

        call ceedoperatorapply(op_mass,u,v,ceed_request_immediate,err)

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
      call ceedqfunctiondestroy(qf_mass,err)
      call ceedoperatordestroy(op_setup,err)
      call ceedoperatordestroy(op_mass,err)
      call ceedelemrestrictiondestroy(erestrictu,err)
      call ceedelemrestrictiondestroy(erestrictx,err)
      call ceedelemrestrictiondestroy(erestrictui,err)
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
