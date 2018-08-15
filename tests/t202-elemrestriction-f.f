c-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer x,y
      integer r

      integer ne
      parameter(ne=8)
      integer blksize
      parameter(blksize=5)

      integer*4 ind(2*ne)
      real*8 a(ne+1)

      character arg*32

      call getarg(1,arg)
      call ceedinit(trim(arg)//char(0),ceed,err)

      call ceedvectorcreate(ceed,ne+1,x,err)

      do i=1,ne+1
        a(i)=10+i-1
      enddo

      call ceedvectorsetarray(x,ceed_mem_host,ceed_use_pointer,a,err)

      do i=1,ne
        ind(2*i-1)=i-1
        ind(2*i  )=i
      enddo

      call ceedelemrestrictioncreateblocked(ceed,ne,2,blksize,ne+1,1,
     $  ceed_mem_host,ceed_use_pointer,ind,r,err)

      call ceedvectorcreate(ceed,2*blksize*2,y,err);
      call ceedvectorsetarray(y,ceed_mem_host,ceed_copy_values,%val(0),
     $  err);

c    No Transpose
      call ceedelemrestrictionapply(r,ceed_notranspose,
     $  ceed_notranspose,x,y,ceed_request_immediate,err)
      call ceedvectorview(y,err)

c    Transpose
      call ceedvectorgetarray(x,ceed_mem_host,a,err)
      do i=1,ne+1
        a(i)=0.0
      enddo
      call ceedvectorrestorearray(x,a,err)
      
      call ceedelemrestrictionapply(r,ceed_transpose,
     $  ceed_notranspose,y,x,ceed_request_immediate,err)

      call ceedvectorview(x,err)

      call ceedvectordestroy(x,err)
      call ceedvectordestroy(y,err)
      call ceedelemrestrictiondestroy(r,err)
      call ceeddestroy(ceed,err)

      end
c-----------------------------------------------------------------------
