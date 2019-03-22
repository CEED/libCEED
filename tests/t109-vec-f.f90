!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer x,y,n
      real*8 a(10)
      real*8 b(10)
      real*8 c(10)
      real*8 diff
      integer*8 boff,coff
      character arg*32

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)

      n = 10

      call ceedvectorcreate(ceed,n,x,err)
      call ceedvectorcreate(ceed,n,y,err)

      do i=1,10
        a(i)=9+i
      enddo

      call ceedvectorsetarray(x,ceed_mem_host,ceed_use_pointer,a,err)
      call ceedvectorgetarrayread(x,ceed_mem_device,b,boff,err)
      call ceedvectorsetarray(y,ceed_mem_device,ceed_copy_values,b,boff,err)
      call ceedvectorrestorearrayread(x,b,boff,err)
      call ceedvectorgetarrayread(y,ceed_mem_host,c,coff,err)
      do i=1,10
        diff = c(i+coff)-(9+i)
        if (abs(diff)>1.0D-15) then
          write(*,*) 'Error reading array c(',i,') = ',c(i+coff),' != ',9+i
        endif
      enddo
      call ceedvectorrestorearrayread(y,c,coff,err)

      call ceedvectordestroy(x,err)
      call ceedvectordestroy(y,err)
      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
