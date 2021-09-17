!-----------------------------------------------------------------------
      program test
      implicit none
      include 'ceed/fortran.h'

      integer ceed,err
      integer x,i,n
      real*8 a(10)
      real*8 norm,diff
      integer*8 aoffset
      character arg*32

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)

      n=10

      call ceedvectorcreate(ceed,n,x,err)

      do i=1,10
        if (mod(i,2) == 0) then
          a(i)=i-1
        else
          a(i)=-(i-1)
        endif
      enddo
      aoffset=0
      call ceedvectorsetarray(x,ceed_mem_host,ceed_use_pointer,a,aoffset,err)

      call ceedvectornorm(x,ceed_norm_1,norm,err)
      diff = norm - 45.
      if (abs(diff)>1.0D-14) then
! LCOV_EXCL_START
        write(*,*) 'Error L1 norm ',norm,' != 45.'
! LCOV_EXCL_STOP
      endif

      call ceedvectornorm(x,ceed_norm_2,norm,err)
      diff = norm - sqrt(285.)
      if (abs(diff)>1.0D-6) then
! LCOV_EXCL_START
        write(*,*) 'Error L2 norm ',norm,' != sqrt(285.)'
! LCOV_EXCL_STOP
      endif

      call ceedvectornorm(x,ceed_norm_max,norm,err)
      diff = norm - 9.
      if (abs(diff)>1.0D-14) then
! LCOV_EXCL_START
        write(*,*) 'Error Max norm ',norm,' != 9.'
! LCOV_EXCL_STOP
      endif

      call ceedvectordestroy(x,err)
      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
