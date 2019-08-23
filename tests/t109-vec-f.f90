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

      n = 10

      call ceedvectorcreate(ceed,n,x,err)
      call ceedvectorcreate(ceed,n,y,err)

      do i=1,10
        a(i)=9+i
      enddo

      aoffset=0
      call ceedvectorsetarray(x,ceed_mem_host,ceed_use_pointer,a,aoffset,err)
      call ceedvectorgetarrayread(x,ceed_mem_device,b,boffset,err)
      call ceedvectorsetarray(y,ceed_mem_device,ceed_copy_values,b,boffset,err)
      call ceedvectorrestorearrayread(x,b,boffset,err)
      call ceedvectorgetarrayread(y,ceed_mem_host,c,coffset,err)
      do i=1,10
        diff = c(i+coffset)-(9+i)
        if (abs(diff)>1.0D-15) then
! LCOV_EXCL_START
          write(*,*) 'Error reading array c(',i,') = ',c(i+coffset),' != ',9+i
! LCOV_EXCL_STOP
        endif
      enddo
      call ceedvectorrestorearrayread(y,c,coffset,err)

      call ceedvectordestroy(x,err)
      call ceedvectordestroy(y,err)
      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
