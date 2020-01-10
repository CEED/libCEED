!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      real*8 a(16), lambda(4)

      character arg*32

      a = (/ 0.2, 0.0745355993, -0.0745355993, 0.0333333333,&
     &       0.0745355993, 1., 0.1666666667, -0.0745355993,&
     &      -0.0745355993, 0.1666666667, 1., 0.0745355993,&
     &      0.0333333333, -0.0745355993, 0.0745355993, 0.2 /)

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)
      call ceedsymmetricschurdecomposition(ceed,a,lambda,4,err);
      write (*,*) 'Q:'
      do i=0,3
        do j=1,4
          if (abs(a(i*4+j))<1.0D-14) then
! LCOV_EXCL_START
            a(i*4+j) = 0
! LCOV_EXCL_STOP
          endif
        enddo
        write(*,'(A,F12.8,F12.8,F12.8,F12.8)') '',&
     &   a(i*4+1),a(i*4+2),a(i*4+3),a(i*4+4)
      enddo
      write (*,*) 'lambda:'
      do i=1,4
        if (abs(lambda(i))<1.0D-14) then
! LCOV_EXCL_START
          lambda(i) = 0
! LCOV_EXCL_STOP
        endif
        write(*,'(A,F12.8)') '',lambda(i)
      enddo

      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
