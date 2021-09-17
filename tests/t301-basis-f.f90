!-----------------------------------------------------------------------
      program test
      implicit none
      include 'ceed/fortran.h'

      integer ceed,err,i,j
      real*8 a(12),qr(12),a_qr(12),tau(3)

      character arg*32

      A = (/ 1, -1, 4, 1, 4, -2, 1, 4, 2, 1, -1, 0 /)
      qr = (/ 1, -1, 4, 1, 4, -2, 1, 4, 2, 1, -1, 0 /)
      a_qr = (/ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 /)

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)
      call ceedqrfactorization(ceed,qr,tau,4,3,err)
      do i=1,3
        do j=i,3
          a_qr((i-1)*3+j)=qr((i-1)*3+j)
        enddo
      enddo
      call ceedhouseholderapplyq(a_qr,qr,tau,ceed_notranspose,4,3,3,3,1,err)

      do i=1,12
        if (abs(a(i)-a_qr(i))>1.0D-14) then
! LCOV_EXCL_START
        write(*,*) 'Error in QR factorization a_qr(',i,') = ',a_qr(i),&
    &    ' != a(',i,') = ',a(i)
! LCOV_EXCL_STOP
      endif
      enddo

      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
