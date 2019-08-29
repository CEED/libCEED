!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      real*8 a(16), lambda(4)

      character arg*32

      a = (/ 0.19996678, 0.0745459, -0.07448852, 0.0332866,&
     &       0.0745459, 1., 0.16666509, -0.07448852,&
     &      -0.07448852, 0.16666509, 1., 0.0745459,&
     &       0.0332866, -0.07448852, 0.0745459, 0.19996678 /)

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)
      call ceedsymmetricschurdecomposition(ceed,a,lambda,4,err);
      write (*,*) 'Q:'
      do i=0,3
        do j=1,4
          if (abs(a(i*4+j))<1.0D-14) then
            a(i*4+j) = 0
          endif
        enddo
        write(*,'(A,F12.8,F12.8,F12.8,F12.8)') '',&
     &   a(i*4+1),a(i*4+2),a(i*4+3),a(i*4+4)
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
