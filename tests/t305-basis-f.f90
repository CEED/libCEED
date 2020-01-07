!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      real*8 m(16), k(16), x(16), lambda(4), xxt(16)

      character arg*32

      m = (/ 0.2, 0.0745355993, -0.0745355993, 0.0333333333,&
     &       0.0745355993, 1., 0.1666666667, -0.0745355993,&
     &      -0.0745355993, 0.1666666667, 1., 0.0745355993,&
     &      0.0333333333, -0.0745355993, 0.0745355993, 0.2 /)
      k = (/ 3.0333333333, -3.4148928136, 0.4982261470, -0.1166666667,&
     &      -3.4148928136, 5.8333333333, -2.9166666667, 0.4982261470,&
     &       0.4982261470, -2.9166666667, 5.8333333333, -3.4148928136,&
     &      -0.1166666667, 0.4982261470, -3.4148928136, 3.0333333333 /)

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)
      call ceedsimultaneousdiagonalization(ceed,k,m,x,lambda,4,err);

      do i=0,3
        do j=0,3
          xxt(j+i*4+1) = 0
          do kk=0,3
            xxt(j+i*4+1) = xxt(j+i*4+1)+x(kk+i*4+1)*x(kk+j*4+1)
          enddo
        enddo
      enddo

      write (*,*) 'x x^T:'
      do i=0,3
        do j=1,4
          if (abs(x(i*4+j))<1.0D-14) then
! LCOV_EXCL_START
            x(i*4+j) = 0
! LCOV_EXCL_STOP
          endif
        enddo
        write(*,'(A,F12.8,F12.8,F12.8,F12.8)') '',&
     &   xxt(i*4+1),xxt(i*4+2),xxt(i*4+3),xxt(i*4+4)
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
