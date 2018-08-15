c-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer x,n
      real*8 a(10)
      real*8 b(10)
      real*8 diff
      real*8 val
      character arg*32

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)

      n=10
      val=3.0

      call ceedvectorcreate(ceed,n,x,err)

      do i=1,10
        a(i)=10+i
      enddo

      call ceedvectorsetarray(x,ceed_mem_host,ceed_use_pointer,a,err)
      call ceedvectorgetarrayread(x,ceed_mem_host,b,err)

      do i=1,10
        diff=b(i)-10-i
        if (abs(diff)>1.0D-15) then
          write(*,*) 'Error reading array b(',i,')=',b(i)
        endif
      enddo

      call ceedvectorrestorearrayread(x,b,err)

      call ceedvectorsetvalue(x,val,err)
      call ceedvectorgetarrayread(x,ceed_mem_host,b,err)

      do i=1,10
        diff=b(i)-val
        if (abs(diff)>1.0D-15) then
          write(*,*) 'Error reading array b(',i,')=',b(i)
        endif
      enddo

      call ceedvectorrestorearrayread(x,b,err)

      call ceedvectordestroy(x,err)
      call ceeddestroy(ceed,err)

      end
c-----------------------------------------------------------------------
