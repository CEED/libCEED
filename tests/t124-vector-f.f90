!-----------------------------------------------------------------------
      program test
      implicit none
      include 'ceed/fortran.h'

      integer ceed,err
      integer i,x,n
      real*8 a(10)
      real*8 diff
      integer*8 aoffset
      character arg*32

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)

      n=10

      call ceedvectorcreate(ceed,n,x,err)

      aoffset=0
      call ceedvectorgetarraywrite(x,ceed_mem_host,a,aoffset,err)
      do i=1,10
        a(i+aoffset)=3*i
      enddo
      call ceedvectorrestorearray(x,a,aoffset,err)

      call ceedvectorgetarrayread(x,ceed_mem_host,a,aoffset,err)
      do i=1,10
        diff=a(i+aoffset)-3*i
        if (abs(diff)>1.0D-15) then
! LCOV_EXCL_START
          write(*,*) 'Error writing array a(',i,')=',a(i+aoffset)
! LCOV_EXCL_STOP
        endif
      enddo
      call ceedvectorrestorearrayread(x,a,aoffset,err)

      call ceedvectordestroy(x,err)
      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
