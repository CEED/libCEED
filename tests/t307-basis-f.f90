!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer b
      real*8 colograd1d(16), colograd1d2(36)

      character arg*32

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)

!     Already collocated, GetCollocatedGrad will return grad1d
      call ceedbasiscreatetensorh1lagrange(ceed,1,1,4,4,ceed_gauss_lobatto,b,&
     & err)
      call ceedbasisgetcollocatedgrad(b,colograd1d,err)
      call ceedbasisview(b,err)
      do i=1,16
        if (abs(colograd1d(i))<1.0D-14) then
          colograd1d(i) = 0
        endif
      enddo
      do i=0,3
        write(*,'(A,I1,A,F12.8,F12.8,F12.8,F12.8,F12.8,F12.8)')&
     &   'colograd[',i,']:',(colograd1d(j+4*i),j=1,4)
      call flush(6)
      enddo
      call ceedbasisdestroy(b,err)

!     Q = P, not already collocated
      call ceedbasiscreatetensorh1lagrange(ceed,1,1,4,4,ceed_gauss,b,err)
      call ceedbasisgetcollocatedgrad(b,colograd1d,err)
      call ceedbasisview(b,err)
      do i=1,16
        if (abs(colograd1d(i))<1.0D-14) then
! LCOV_EXCL_START
          colograd1d(i) = 0
! LCOV_EXCL_STOP
        endif
      enddo
      do i=0,3
        write(*,'(A,I1,A,F12.8,F12.8,F12.8,F12.8,F12.8,F12.8)')&
     &   'colograd[',i,']:',(colograd1d(j+4*i),j=1,4)
      call flush(6)
      enddo
      call ceedbasisdestroy(b,err)

!     Q = P + 2, not already collocated
      call ceedbasiscreatetensorh1lagrange(ceed,1,1,4,6,ceed_gauss,b,err)
      call ceedbasisgetcollocatedgrad(b,colograd1d2,err)
      call ceedbasisview(b,err)
      do i=1,36
        if (abs(colograd1d2(i))<1.0D-14) then
! LCOV_EXCL_START
          colograd1d2(i) = 0
! LCOV_EXCL_STOP
        endif
      enddo
      do i=0,5
        write(*,'(A,I1,A,F12.8,F12.8,F12.8,F12.8,F12.8,F12.8)')&
     &   'colograd[',i,']:',(colograd1d2(j+6*i),j=1,6)
      call flush(6)
      enddo
      call ceedbasisdestroy(b,err)

      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
