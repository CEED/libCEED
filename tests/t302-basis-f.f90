!-----------------------------------------------------------------------
      program test
      implicit none
      include 'ceed/fortran.h'

      integer ceed,err,i,j
      integer b,p
      parameter(p=4)
      real*8 collograd1d(36),x2(6)
      real*8 grad1d(16),qref(6)
      integer*8 gradoffset,qoffset
      real*8 sum

      character arg*32

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)

!     Already collocated, GetCollocatedGrad will return grad1d
      call ceedbasiscreatetensorh1lagrange(ceed,1,1,p,p,ceed_gauss_lobatto,b,&
     & err)
      call ceedbasisgetcollocatedgrad(b,collograd1d,err)
      call ceedbasisgetgrad1d(b,grad1d,gradoffset,err)
      do i=0,p-1
        do j=1,p
          if (abs(collograd1d(j+p*i)-grad1d(j+p*i+gradoffset))>1.0D-13) then
! LCOV_EXCL_START
            write(*,*) 'Error in collocated gradient ',collograd1d(j+p*i),' != ',&
     &       grad1d(j+p*i+gradoffset)
! LCOV_EXCL_STOP
          endif
        enddo
      enddo
      call ceedbasisdestroy(b,err)

!     Q = P, not already collocated
      call ceedbasiscreatetensorh1lagrange(ceed,1,1,p,p,ceed_gauss,b,err)
      call ceedbasisgetcollocatedgrad(b,collograd1d,err)

      call ceedbasisgetqref(b,qref,qoffset,err)
      do i=1,p
        x2(i)=qref(i+qoffset)*qref(i+qoffset)
      enddo

      do i=0,p-1
        sum=0
        do j=1,p
            sum=sum+collograd1d(j+p*i)*x2(j)
        enddo
        if (abs(sum-2*qref(i+1+qoffset))>1.0D-13) then
! LCOV_EXCL_START
            write(*,*) 'Error in collocated gradient ',sum,' != ',&
            &       2*qref(i+1+qoffset)
! LCOV_EXCL_STOP
        endif
      enddo
      call ceedbasisdestroy(b,err)

!     Q = P + 2, not already collocated
      call ceedbasiscreatetensorh1lagrange(ceed,1,1,p,p+2,ceed_gauss,b,err)
      call ceedbasisgetcollocatedgrad(b,collograd1d,err)

      call ceedbasisgetqref(b,qref,qoffset,err)
      do i=1,p+2
        x2(i)=qref(i+qoffset)*qref(i+qoffset)
      enddo

      do i=0,p+1
        sum=0
        do j=1,p+2
            sum=sum+collograd1d(j+(p+2)*i)*x2(j)
        enddo
        if (abs(sum-2*qref(i+1+qoffset))>1.0D-13) then
! LCOV_EXCL_START
            write(*,*) 'Error in collocated gradient ',sum,' != ',&
     &       2*qref(i+1+qoffset)
! LCOV_EXCL_STOP
        endif
      enddo
      call ceedbasisdestroy(b,err)

      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------