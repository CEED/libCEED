!-----------------------------------------------------------------------
      program test
      implicit none
      include 'ceed/fortran.h'

      integer ceed,err,i,j,kk
      real*8 a(16), q(16), qlambdaqt(16), lambda(4), sum

      character arg*32

      a = (/ 0.2, 0.0745355993, -0.0745355993, 0.0333333333,&
     &       0.0745355993, 1., 0.1666666667, -0.0745355993,&
     &      -0.0745355993, 0.1666666667, 1., 0.0745355993,&
     &      0.0333333333, -0.0745355993, 0.0745355993, 0.2 /)

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)

     do i=0,3
       do j=1,4
         q(4*i+j)=a(4*i+j)
       enddo
     enddo

      call ceedsymmetricschurdecomposition(ceed,q,lambda,4,err)

!     Check A = Q lambda Q^T
      do i=0,3
        do j=0,3
          sum = 0
          do kk=0,3
            sum=sum+q(kk+i*4+1)*lambda(kk+1)*q(kk+j*4+1)
          enddo
          qlambdaqt(j+i*4+1)=sum
        enddo
      enddo
      do i=0,3
        do j=1,4
          if (abs(a(i*4+j) - qlambdaqt(i*4+j))>1.0D-14) then
! LCOV_EXCL_START
            write(*,'(A,I1,A,I1,A,F12.8,A,F12.8)') 'Error: [', &
     &      i,',',j-1,'] ',a(i*4+j),'!=',qlambdaqt(i*4+j)
! LCOV_EXCL_STOP
          endif
        enddo
      enddo

      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
