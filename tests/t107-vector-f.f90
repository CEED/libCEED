!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer x,n
      real*8 a(10)
      integer*8 aoffset
      character arg*32

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)

      n=10

      call ceedvectorcreate(ceed,n,x,err)

      do i=1,10
        a(i)=9+i
      enddo
      aoffset=0
      call ceedvectorsetarray(x,ceed_mem_host,ceed_use_pointer,a,aoffset,err)

      call ceedvectorview(x,err)

      call ceedvectordestroy(x,err)
      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
