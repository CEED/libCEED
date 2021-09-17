!-----------------------------------------------------------------------
      program test
      implicit none
      include 'ceed/fortran.h'

      integer ceed,err,i,j,kk
      real*8 m(16), k(16), x(16), lambda(4), sum, work(16), val

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
      call ceedsimultaneousdiagonalization(ceed,k,m,x,lambda,4,err)

!     Check X^T M X = I
      do i=0,3
        do j=0,3
          sum = 0
          do kk=0,3
            sum=sum+m(kk+i*4+1)*x(j+kk*4+1)
          enddo
          work(j+i*4+1)=sum
        enddo
      enddo
      do i=0,3
        do j=0,3
          sum = 0
          do kk=0,3
            sum=sum+x(i+kk*4+1)*work(j+kk*4+1)
          enddo
          m(j+i*4+1)=sum
        enddo
      enddo
      do i=0,3
        do j=1,4
          if (i+1 == j) then
            val=1.0
          else
            val=0.0
          endif
          if (abs(m(i*4+j) - val)>1.0D-13) then
! LCOV_EXCL_START
            write(*,'(A,I1,A,I1,A,F12.8,A,F12.8)') 'Error: [', &
     &      i,',',j,'] ',m(i*4+j),'!=',val
! LCOV_EXCL_STOP
          endif
        enddo
      enddo

!     Check X^T K X = Lambda
      do i=0,3
        do j=0,3
          sum = 0
          do kk=0,3
            sum=sum+k(kk+i*4+1)*x(j+kk*4+1)
          enddo
          work(j+i*4+1)=sum
        enddo
      enddo
      do i=0,3
        do j=0,3
          sum = 0
          do kk=0,3
            sum=sum+x(i+kk*4+1)*work(j+kk*4+1)
          enddo
          k(j+i*4+1)=sum
        enddo
      enddo
      do i=0,3
        do j=1,4
          if (i+1 == j) then
            val=lambda(j)
          else
            val=0.0
          endif
          if (abs(k(i*4+j) - val)>1.0D-13) then
! LCOV_EXCL_START
            write(*,'(A,I1,A,I1,A,F12.8,A,F12.8)') 'Error: [', &
     &      i,',',j,'] ',k(i*4+j),'!=',val
! LCOV_EXCL_STOP
          endif
        enddo
      enddo

      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
