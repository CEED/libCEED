!-----------------------------------------------------------------------
!
! Header with QFunctions
! 
      include 't502-operator-f.h'
!-----------------------------------------------------------------------
      program test

      include 'ceed/fortran.h'

      integer ceed,err,i,j
      integer stridesu(3)
      integer erestrictx,erestrictui
      integer erestrictucoarse,erestrictufine
      integer bx,bucoarse,bufine
      integer qf_setup,qf_mass
      integer op_setup,op_masscoarse,op_massfine
      integer op_prolong,op_restrict
      integer qdata,x,ucoarse,ufine,vcoarse,vfine,pmultfine
      integer nelem,pcoarse,pfine,q,ncomp
      parameter(ncomp=2)
      parameter(nelem=15)
      parameter(pcoarse=3)
      parameter(pfine=5)
      parameter(q=8)
      integer nx,nucoarse,nufine
      parameter(nx=nelem+1)
      parameter(nucoarse=nelem*(pcoarse-1)+1)
      parameter(nufine=nelem*(pfine-1)+1)
      integer indx(nelem*2)
      integer inducoarse(nelem*pcoarse)
      integer indufine(nelem*pfine)
      real*8 arrx(nx)
      integer*8 voffset,xoffset
      real*8 val

      real*8 hv(nufine*ncomp)
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
      call ceedelemrestrictioncreate(ceed,nelem,2,1,1,nx,ceed_mem_host,&
     & ceed_use_pointer,indx,erestrictx,err)

      do i=0,nelem-1
        do j=0,pcoarse-1
          inducoarse(pcoarse*i+j+1)=i*(pcoarse-1)+j
        enddo
      enddo
      call ceedelemrestrictioncreate(ceed,nelem,pcoarse,ncomp,nucoarse,&
     & ncomp*nucoarse,ceed_mem_host,ceed_use_pointer,inducoarse,&
     & erestrictucoarse,err)

      do i=0,nelem-1
        do j=0,pfine-1
          indufine(pfine*i+j+1)=i*(pfine-1)+j
        enddo
      enddo
      call ceedelemrestrictioncreate(ceed,nelem,pfine,ncomp,nufine,&
     & ncomp*nufine,ceed_mem_host,ceed_use_pointer,indufine,&
     & erestrictufine,err)

     stridesu=[1,q,q]
      call ceedelemrestrictioncreatestrided(ceed,nelem,q,1,q*nelem,stridesu,&
     & erestrictui,err)

      call ceedbasiscreatetensorh1lagrange(ceed,1,1,2,q,ceed_gauss,bx,err)
      call ceedbasiscreatetensorh1lagrange(ceed,1,ncomp,pfine,q,ceed_gauss,&
     & bufine,err)
      call ceedbasiscreatetensorh1lagrange(ceed,1,ncomp,pcoarse,q,ceed_gauss,&
     & bucoarse,err)

     call ceedqfunctioncreateinterior(ceed,1,setup,&
    &SOURCE_DIR&
    &//'t502-operator.h:setup'//char(0),qf_setup,err)
     call ceedqfunctionaddinput(qf_setup,'weight',1,ceed_eval_weight,err)
     call ceedqfunctionaddinput(qf_setup,'dx',1,ceed_eval_grad,err)
     call ceedqfunctionaddoutput(qf_setup,'qdata',1,ceed_eval_none,err)

     call ceedqfunctioncreateinterior(ceed,1,mass,&
    &SOURCE_DIR&
    &//'t502-operator.h:mass'//char(0),qf_mass,err)
     call ceedqfunctionaddinput(qf_mass,'qdata',1,ceed_eval_none,err)
     call ceedqfunctionaddinput(qf_mass,'u',ncomp,ceed_eval_interp,err)
     call ceedqfunctionaddoutput(qf_mass,'v',ncomp,ceed_eval_interp,err)

      call ceedoperatorcreate(ceed,qf_setup,ceed_qfunction_none,&
     & ceed_qfunction_none,op_setup,err)
      call ceedoperatorcreate(ceed,qf_mass,ceed_qfunction_none,&
     & ceed_qfunction_none,op_massfine,err)

      call ceedvectorcreate(ceed,nx,x,err)
      xoffset=0
      call ceedvectorsetarray(x,ceed_mem_host,ceed_use_pointer,arrx,xoffset,err)
      call ceedvectorcreate(ceed,nelem*q,qdata,err)

      call ceedoperatorsetfield(op_setup,'weight',ceed_elemrestriction_none,&
     & bx,ceed_vector_none,err)
      call ceedoperatorsetfield(op_setup,'dx',erestrictx,bx,&
     & ceed_vector_active,err)
      call ceedoperatorsetfield(op_setup,'qdata',erestrictui,&
     & ceed_basis_collocated,ceed_vector_active,err)
      call ceedoperatorsetfield(op_massfine,'qdata',erestrictui,&
     & ceed_basis_collocated,qdata,err)
      call ceedoperatorsetfield(op_massfine,'u',erestrictufine,bufine,&
     & ceed_vector_active,err)
      call ceedoperatorsetfield(op_massfine,'v',erestrictufine,bufine,&
     & ceed_vector_active,err)

      call ceedoperatorapply(op_setup,x,qdata,ceed_request_immediate,err)

! Create multigrid level
      call ceedvectorcreate(ceed,ncomp*nufine,pmultfine,err)
      val=1.0
      call ceedvectorsetvalue(pmultfine,val,err)
      call ceedoperatormultigridlevelcreate(op_massfine,pmultfine,&
     & erestrictucoarse,bucoarse,op_masscoarse,op_prolong,op_restrict,err)

! Coarse problem
      call ceedvectorcreate(ceed,ncomp*nucoarse,ucoarse,err)
      val=1.0
      call ceedvectorsetvalue(ucoarse,val,err)
      call ceedvectorcreate(ceed,ncomp*nucoarse,vcoarse,err)
      call ceedoperatorapply(op_masscoarse,ucoarse,vcoarse,&
     & ceed_request_immediate,err)

! Check output
      call ceedvectorgetarrayread(vcoarse,ceed_mem_host,hv,voffset,err)
      total=0.
      do i=1,nucoarse*ncomp
        total=total+hv(voffset+i)
      enddo
      if (abs(total-2.)>1.0d-10) then
! LCOV_EXCL_START
        write(*,*) 'Computed Area: ',total,' != True Area: 1.0'
! LCOV_EXCL_STOP
      endif
      call ceedvectorrestorearrayread(vcoarse,hv,voffset,err)

! Prolong coarse u
      call ceedvectorcreate(ceed,ncomp*nufine,ufine,err)
      call ceedoperatorapply(op_prolong,ucoarse,ufine,&
     & ceed_request_immediate,err)

! Fine problem
      call ceedvectorcreate(ceed,ncomp*nufine,vfine,err)
      call ceedoperatorapply(op_massfine,ufine,vfine,&
     & ceed_request_immediate,err)

! Check output
      call ceedvectorgetarrayread(vfine,ceed_mem_host,hv,voffset,err)
      total=0.
      do i=1,nufine*ncomp
        total=total+hv(voffset+i)
      enddo
      if (abs(total-2.)>1.0d-10) then
! LCOV_EXCL_START
        write(*,*) 'Computed Area Fine Grid: ',total,' != True Area: 1.0'
! LCOV_EXCL_STOP
      endif
      call ceedvectorrestorearrayread(vfine,hv,voffset,err)

! Restrict state to coarse grid
      call ceedoperatorapply(op_restrict,vfine,vcoarse,&
     & ceed_request_immediate,err)

! Check output
      call ceedvectorgetarrayread(vcoarse,ceed_mem_host,hv,voffset,err)
      total=0.
      do i=1,nucoarse*ncomp
        total=total+hv(voffset+i)
      enddo
      if (abs(total-2.)>1.0d-10) then
! LCOV_EXCL_START
        write(*,*) 'Computed Area Coarse Grid: ',total,' != True Area: 1.0'
! LCOV_EXCL_STOP
      endif
      call ceedvectorrestorearrayread(vcoarse,hv,voffset,err)

      call ceedvectordestroy(qdata,err)
      call ceedvectordestroy(x,err)
      call ceedvectordestroy(ucoarse,err)
      call ceedvectordestroy(ufine,err)
      call ceedvectordestroy(vcoarse,err)
      call ceedvectordestroy(vfine,err)
      call ceedvectordestroy(pmultfine,err)
      call ceedoperatordestroy(op_masscoarse,err)
      call ceedoperatordestroy(op_massfine,err)
      call ceedoperatordestroy(op_prolong,err)
      call ceedoperatordestroy(op_restrict,err)
      call ceedoperatordestroy(op_setup,err)
      call ceedqfunctiondestroy(qf_mass,err)
      call ceedqfunctiondestroy(qf_setup,err)
      call ceedbasisdestroy(bucoarse,err)
      call ceedbasisdestroy(bufine,err)
      call ceedbasisdestroy(bx,err)
      call ceedelemrestrictiondestroy(erestrictucoarse,err)
      call ceedelemrestrictiondestroy(erestrictufine,err)
      call ceedelemrestrictiondestroy(erestrictx,err)
      call ceedelemrestrictiondestroy(erestrictui,err)
      call ceeddestroy(ceed,err)
      end
!-----------------------------------------------------------------------
