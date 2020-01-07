!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer x,y,n
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
      call ceedvectorcreate(ceed,n,y,err)

      do i=1,10
        a(i)=9+i
      enddo
      aoffset=0
      call ceedvectorsetarray(x,ceed_mem_host,ceed_use_pointer,a,aoffset,err)

      do i=1,10
        b(i)=0
      enddo
      boffset=0
      call ceedvectorsetarray(y,ceed_mem_host,ceed_use_pointer,b,boffset,err)

      call ceedvectorgetarrayread(x,ceed_mem_device,c,coffset,err)
      call ceedvectorsetarray(y,ceed_mem_device,ceed_copy_values,c,coffset,err)
      call ceedvectorrestorearrayread(x,c,coffset,err)

      call ceedvectorsyncarray(y,ceed_mem_host,err)
      do i=1,10
        diff = b(i+boffset)-(9+i)
        if (abs(diff)>1.0D-15) then
! LCOV_EXCL_START
          write(*,*) 'Error reading array b(',i,') = ',b(i+boffset),' != ',9+i
! LCOV_EXCL_STOP
        endif
      enddo

      call ceedvectordestroy(x,err)
      call ceedvectordestroy(y,err)
      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
