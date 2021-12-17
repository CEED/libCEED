!-----------------------------------------------------------------------
      program test
      implicit none
      include 'ceed/fortran.h'

      integer ceed,err
      integer i,x,n
      real*8 a(10)
      real*8 b(10)
      real*8 c(10)
      real*8 diff
      integer*8 aoffset,boffset,coffset
      character arg*32

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)

      n=10

      call ceedvectorcreate(ceed,n,x,err)

      do i=1,10
        a(i)=0
      enddo
      a(3)=-3.14

      aoffset=0
      call ceedvectorsetarray(x,ceed_mem_host,ceed_use_pointer,a,aoffset,err)

! Taking array should return a
      call ceedvectortakearray(x,ceed_mem_host,c,coffset,err)
      diff=c(coffset+3)+3.14
      if (abs(diff)>1.0D-15) then
! LCOV_EXCL_START
        write(*,*) 'Error taking array c(3)=',c(3)
! LCOV_EXCL_STOP
      endif

! Getting array should not modify a
      call ceedvectorgetarraywrite(x,ceed_mem_host,b,boffset,err)
      b(boffset+5) = -3.14
      call ceedvectorrestorearray(x,b,boffset,err)
      diff=a(5)+3.14
      if (abs(diff)<1.0D-15) then
! LCOV_EXCL_START
        write(*,*) 'Error protecting array a(3)=',a(3)
! LCOV_EXCL_STOP
      endif

      call ceedvectordestroy(x,err)
      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
