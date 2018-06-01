c-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      real*8 qr(12)

      character arg*32

      qr = (/ 1, -1, 4, 1, 4, -2, 1, 4, 2, 1, -1, 0 /)

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)
      call ceedqrfactorization(qr,4,3,err);
      do i=1,12
        if (abs(qr(i))<1.0D-14) then
          qr(i) = 0
        endif
        write(*,'(A,F12.8)') '',qr(i)
      enddo

      call ceeddestroy(ceed,err)

      end
c-----------------------------------------------------------------------
