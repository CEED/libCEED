!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      real*8 m(16), k(16), x(16), lambda(4)

      character arg*32

      m = (/ 0.19996678, 0.0745459, -0.07448852, 0.0332866,&
     &       0.0745459, 1., 0.16666509, -0.07448852,&
     &      -0.07448852, 0.16666509, 1., 0.0745459,&
     &       0.0332866, -0.07448852, 0.0745459, 0.19996678 /)
      k = (/ 3.03344425, -3.41501767, 0.49824435, -0.11667092,&
     &      -3.41501767, 5.83354662, -2.9167733, 0.49824435,&
     &       0.49824435, -2.9167733, 5.83354662, -3.41501767,&
     &      -0.11667092, 0.49824435, -3.41501767, 3.03344425 /)

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)
      call ceedsimultaneousdiagonalization(ceed,k,m,x,lambda,4,err);
      write (*,*) 'x:'
      do i=0,3
        do j=1,4
          if (abs(x(i*4+j))<1.0D-14) then
            x(i*4+j) = 0
          endif
        enddo
        write(*,'(A,F12.8,F12.8,F12.8,F12.8)') '',&
     &   x(i*4+1),x(i*4+2),x(i*4+3),x(i*4+4)
      enddo
      write (*,*) 'lambda:'
      do i=1,4
        if (abs(lambda(i))<1.0D-14) then
          lambda(i) = 0
        endif
        write(*,'(A,F12.8)') '',lambda(i)
      enddo

      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
