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

      integer ceed,err,i,j,k
      integer erestrictxtet,erestrictutet,erestrictxitet,erestrictuitet,&
&             erestrictxhex,erestrictuhex,erestrictxihex,erestrictuihex
      integer bxtet,butet,bxhex,buhex
      integer qf_setuptet,qf_masstet,qf_setuphex,qf_masshex
      integer op_setuptet,op_masstet,op_setuphex,op_masshex,op_setup,op_mass
      integer qdatatet,qdatahex,x,u,v
      integer nelemtet,nelemhex,ptet,phex,qtet,qhex,d
      integer row,col,offset
      parameter(nelemtet=6)
      parameter(ptet=6)
      parameter(qtet=4)
      parameter(nelemhex=6)
      parameter(phex=3)
      parameter(qhex=4)
      parameter(d=2)
      integer ndofs,nqptstet,nqptshex,nqpts,nx,ny,nxtet,nytet,nxhex
      parameter(nx=3)
      parameter(ny=3)
      parameter(nxtet=3)
      parameter(nytet=1)
      parameter(nxhex=3)
      parameter(ndofs=(nx*2+1)*(ny*2+1))
      parameter(nqptstet=nelemtet*qtet)
      parameter(nqptshex=nelemhex*qhex*qhex)
      parameter(nqpts=nqptstet+nqptshex)
      integer indxtet(nelemtet*ptet),indxhex(nelemhex*phex*phex)
      real*8 arrx(d*ndofs)
      integer*8 voffset,xoffset

      real*8 qref(d*qtet)
      real*8 qweight(qtet)
      real*8 interp(ptet*qtet)
      real*8 grad(d*ptet*qtet)

      real*8 hv(ndofs)
      real*8 total

      character arg*32

      external setup,mass

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)

! DoF Coordinates
      do i=0,ny*2
        do j=0,nx*2
          arrx(i+j*(ny*2+1)+0*ndofs+1)=1.d0*i/(2*ny)
          arrx(i+j*(ny*2+1)+1*ndofs+1)=1.d0*j/(2*nx)
        enddo
      enddo

      call ceedvectorcreate(ceed,d*ndofs,x,err)
      xoffset=0
      call ceedvectorsetarray(x,ceed_mem_host,ceed_use_pointer,arrx,xoffset,err)

! Qdata Vectors
      call ceedvectorcreate(ceed,nqptstet,qdatatet,err)
      call ceedvectorcreate(ceed,nqptshex,qdatahex,err)

! Tet Elements
      do i=0,2
        col=mod(i,nx)
        row=i/nx
        offset=col*2+row*(nx*2+1)*2

        indxtet(i*2*ptet+1)=2+offset
        indxtet(i*2*ptet+2)=9+offset
        indxtet(i*2*ptet+3)=16+offset
        indxtet(i*2*ptet+4)=1+offset
        indxtet(i*2*ptet+5)=8+offset
        indxtet(i*2*ptet+6)=0+offset

        indxtet(i*2*ptet+7)=14+offset
        indxtet(i*2*ptet+8)=7+offset
        indxtet(i*2*ptet+9)=0+offset
        indxtet(i*2*ptet+10)=15+offset
        indxtet(i*2*ptet+11)=8+offset
        indxtet(i*2*ptet+12)=16+offset
      enddo

! -- Restrictions
      call ceedelemrestrictioncreate(ceed,nelemtet,ptet,ndofs,d,ceed_mem_host,&
     & ceed_use_pointer,indxtet,erestrictxtet,err)
      call ceedelemrestrictioncreateidentity(ceed,nelemtet,ptet,nelemtet*ptet,&
     & d,erestrictxitet,err)

      call ceedelemrestrictioncreate(ceed,nelemtet,ptet,ndofs,1,ceed_mem_host,&
     & ceed_use_pointer,indxtet,erestrictutet,err)
      call ceedelemrestrictioncreateidentity(ceed,nelemtet,qtet,nqptstet,1,&
     & erestrictuitet,err)

! -- Bases
      call buildmats(qref,qweight,interp,grad)
      call ceedbasiscreateh1(ceed,ceed_triangle,d,ptet,qtet,interp,grad,qref,&
     & qweight,bxtet,err)
      call buildmats(qref,qweight,interp,grad)
      call ceedbasiscreateh1(ceed,ceed_triangle,1,ptet,qtet,interp,grad,qref,&
     & qweight,butet,err)

! -- QFunctions
      call ceedqfunctioncreateinterior(ceed,1,setup,&
     &SOURCE_DIR&
     &//'t510-operator.h:setup'//char(0),qf_setuptet,err)
      call ceedqfunctionaddinput(qf_setuptet,'_weight',1,ceed_eval_weight,err)
      call ceedqfunctionaddinput(qf_setuptet,'dx',d*d,ceed_eval_grad,err)
      call ceedqfunctionaddoutput(qf_setuptet,'rho',1,ceed_eval_none,err)

      call ceedqfunctioncreateinterior(ceed,1,mass,&
     &SOURCE_DIR&
     &//'t510-operator.h:mass'//char(0),qf_masstet,err)
      call ceedqfunctionaddinput(qf_masstet,'rho',1,ceed_eval_none,err)
      call ceedqfunctionaddinput(qf_masstet,'u',1,ceed_eval_interp,err)
      call ceedqfunctionaddoutput(qf_masstet,'v',1,ceed_eval_interp,err)

! -- Operators
! ---- Setup Tet
      call ceedoperatorcreate(ceed,qf_setuptet,ceed_qfunction_none,&
     & ceed_qfunction_none,op_setuptet,err)
      call ceedoperatorsetfield(op_setuptet,'_weight',erestrictxitet,&
     & ceed_notranspose,bxtet,ceed_vector_none,err)
      call ceedoperatorsetfield(op_setuptet,'dx',erestrictxtet,&
     & ceed_notranspose,bxtet,ceed_vector_active,err)
      call ceedoperatorsetfield(op_setuptet,'rho',erestrictuitet,&
     & ceed_notranspose,ceed_basis_collocated,qdatatet,err)
! ---- Mass Tet
      call ceedoperatorcreate(ceed,qf_masstet,ceed_qfunction_none,&
     & ceed_qfunction_none,op_masstet,err)
      call ceedoperatorsetfield(op_masstet,'rho',erestrictuitet,&
     & ceed_notranspose,ceed_basis_collocated,qdatatet,err)
      call ceedoperatorsetfield(op_masstet,'u',erestrictutet,&
     & ceed_notranspose,butet,ceed_vector_active,err)
      call ceedoperatorsetfield(op_masstet,'v',erestrictutet,&
     & ceed_notranspose,butet,ceed_vector_active,err)

! Hex Elements
      do i=0,nelemhex-1
        col=mod(i,nx)
        row=i/nx
        offset=(nxtet*2+1)*(nytet*2)*(1+row)+col*2
        do j=0,phex-1
          do k=0,phex-1
            indxhex(phex*(phex*i+k)+j+1)=offset+k*(nxhex*2+1)+j
          enddo
        enddo
      enddo

! -- Restrictions
      call ceedelemrestrictioncreate(ceed,nelemhex,phex*phex,ndofs,d,&
     & ceed_mem_host,ceed_use_pointer,indxhex,erestrictxhex,err)
      call ceedelemrestrictioncreateidentity(ceed,nelemhex,phex*phex,&
     & nelemhex*phex*phex,d,erestrictxihex,err)

      call ceedelemrestrictioncreate(ceed,nelemhex,phex*phex,ndofs,1,&
     & ceed_mem_host,ceed_use_pointer,indxhex,erestrictuhex,err)
      call ceedelemrestrictioncreateidentity(ceed,nelemhex,qhex*qhex,nqptshex,&
     & 1,erestrictuihex,err)

! -- Bases
      call ceedbasiscreatetensorh1lagrange(ceed,d,d,phex,qhex,ceed_gauss,&
     & bxhex,err)
      call ceedbasiscreatetensorh1lagrange(ceed,d,1,phex,qhex,ceed_gauss,&
     & buhex,err)

! -- QFunctions
      call ceedqfunctioncreateinterior(ceed,1,setup,&
     &SOURCE_DIR&
     &//'t510-operator.h:setup'//char(0),qf_setuphex,err)
      call ceedqfunctionaddinput(qf_setuphex,'_weight',1,ceed_eval_weight,err)
      call ceedqfunctionaddinput(qf_setuphex,'dx',d*d,ceed_eval_grad,err)
      call ceedqfunctionaddoutput(qf_setuphex,'rho',1,ceed_eval_none,err)

      call ceedqfunctioncreateinterior(ceed,1,mass,&
     &SOURCE_DIR&
     &//'t510-operator.h:mass'//char(0),qf_masshex,err)
      call ceedqfunctionaddinput(qf_masshex,'rho',1,ceed_eval_none,err)
      call ceedqfunctionaddinput(qf_masshex,'u',1,ceed_eval_interp,err)
      call ceedqfunctionaddoutput(qf_masshex,'v',1,ceed_eval_interp,err)

! -- Operators
! ---- Setup Hex
      call ceedoperatorcreate(ceed,qf_setuphex,ceed_qfunction_none,&
     & ceed_qfunction_none,op_setuphex,&
     & err)
      call ceedoperatorsetfield(op_setuphex,'_weight',erestrictxihex,&
     & ceed_notranspose,bxhex,ceed_vector_none,err)
      call ceedoperatorsetfield(op_setuphex,'dx',erestrictxhex,&
     & ceed_notranspose,bxhex,ceed_vector_active,err)
      call ceedoperatorsetfield(op_setuphex,'rho',erestrictuihex,&
     & ceed_notranspose,ceed_basis_collocated,qdatahex,err)
! ---- Mass Hex
      call ceedoperatorcreate(ceed,qf_masshex,ceed_qfunction_none,&
     & ceed_qfunction_none,op_masshex,&
     & err)
      call ceedoperatorsetfield(op_masshex,'rho',erestrictuihex,&
     & ceed_notranspose,ceed_basis_collocated,qdatahex,err)
      call ceedoperatorsetfield(op_masshex,'u',erestrictuhex,&
     & ceed_notranspose,buhex,ceed_vector_active,err)
      call ceedoperatorsetfield(op_masshex,'v',erestrictuhex,&
     & ceed_notranspose,buhex,ceed_vector_active,err)

! Composite Operators
      call ceedcompositeoperatorcreate(ceed,op_setup,err)
      call ceedcompositeoperatoraddsub(op_setup,op_setuptet,err)
      call ceedcompositeoperatoraddsub(op_setup,op_setuphex,err)

      call ceedcompositeoperatorcreate(ceed,op_mass,err)
      call ceedcompositeoperatoraddsub(op_mass,op_masstet,err)
      call ceedcompositeoperatoraddsub(op_mass,op_masshex,err)

! Apply Setup Operator
      call ceedoperatorapply(op_setup,x,ceed_vector_none,&
     & ceed_request_immediate,err)

! Apply Mass Operator
      call ceedvectorcreate(ceed,ndofs,u,err)
      call ceedvectorsetvalue(u,1.d0,err)
      call ceedvectorcreate(ceed,ndofs,v,err)
      call ceedvectorsetvalue(v,0.d0,err)

      call ceedoperatorapplyadd(op_mass,u,v,ceed_request_immediate,err)

! Check Output
      call ceedvectorgetarrayread(v,ceed_mem_host,hv,voffset,err)
      total=0.
      do i=1,ndofs
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

! Check Output
      call ceedvectorgetarrayread(v,ceed_mem_host,hv,voffset,err)
      total=-ndofs
      do i=1,ndofs
        total=total+hv(voffset+i)
      enddo
      if (abs(total-1.)>1.0d-10) then
! LCOV_EXCL_START
        write(*,*) 'Computed Area: ',total,' != True Area: 1.0'
! LCOV_EXCL_STOP
      endif
      call ceedvectorrestorearrayread(v,hv,voffset,err)

! Cleanup
      call ceedqfunctiondestroy(qf_setuptet,err)
      call ceedqfunctiondestroy(qf_masstet,err)
      call ceedoperatordestroy(op_setuptet,err)
      call ceedoperatordestroy(op_masstet,err)
      call ceedqfunctiondestroy(qf_setuphex,err)
      call ceedqfunctiondestroy(qf_masshex,err)
      call ceedoperatordestroy(op_setuphex,err)
      call ceedoperatordestroy(op_masshex,err)
      call ceedoperatordestroy(op_setup,err)
      call ceedoperatordestroy(op_mass,err)
      call ceedelemrestrictiondestroy(erestrictutet,err)
      call ceedelemrestrictiondestroy(erestrictxtet,err)
      call ceedelemrestrictiondestroy(erestrictuitet,err)
      call ceedelemrestrictiondestroy(erestrictxitet,err)
      call ceedelemrestrictiondestroy(erestrictuhex,err)
      call ceedelemrestrictiondestroy(erestrictxhex,err)
      call ceedelemrestrictiondestroy(erestrictuihex,err)
      call ceedelemrestrictiondestroy(erestrictxihex,err)
      call ceedbasisdestroy(butet,err)
      call ceedbasisdestroy(bxtet,err)
      call ceedbasisdestroy(buhex,err)
      call ceedbasisdestroy(bxhex,err)
      call ceedvectordestroy(x,err)
      call ceedvectordestroy(u,err)
      call ceedvectordestroy(v,err)
      call ceedvectordestroy(qdatatet,err)
      call ceedvectordestroy(qdatahex,err)
      call ceeddestroy(ceed,err)
      end
!-----------------------------------------------------------------------
