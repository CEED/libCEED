!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      real*8 qr(12), tau(3)

      character arg*32

      qr = (/ 1, -1, 4, 1, 4, -2, 1, 4, 2, 1, -1, 0 /)

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)
      call ceedqrfactorization(ceed,qr,tau,4,3,err);
      do i=1,12
        if (abs(qr(i))<1.0D-14) then
! LCOV_EXCL_START
          qr(i) = 0
! LCOV_EXCL_STOP
        endif
        write(*,'(A,F12.8)') '',qr(i)
      enddo
      do i=1,3
        if (abs(tau(i))<1.0D-14) then
! LCOV_EXCL_START
          tau(i) = 0
! LCOV_EXCL_STOP
        endif
        write(*,'(A,F12.8)') '',tau(i)
      enddo

      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
