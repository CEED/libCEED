!-----------------------------------------------------------------------
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
      integer*8 aoffset

      character arg*32

      call getarg(1,arg)
      call ceedinit(trim(arg)//char(0),ceed,err)

      call ceedvectorcreate(ceed,ne+1,x,err)

      do i=1,ne+1
        a(i)=10+i-1
      enddo

      aoffset=0
      call ceedvectorsetarray(x,ceed_mem_host,ceed_use_pointer,a,aoffset,err)

      do i=1,ne
        ind(2*i-1)=i-1
        ind(2*i  )=i
      enddo

      call ceedelemrestrictioncreateblocked(ceed,ne,2,blksize,ne+1,1,&
     & ceed_mem_host,ceed_use_pointer,ind,r,err)

      call ceedvectorcreate(ceed,2*blksize*2,y,err);
      call ceedvectorsetvalue(y,0.d0,err);

!    No Transpose
      call ceedelemrestrictionapply(r,ceed_notranspose,ceed_notranspose,x,y,&
     & ceed_request_immediate,err)
      call ceedvectorview(y,err)

!    Transpose
      call ceedvectorgetarray(x,ceed_mem_host,a,aoffset,err)
      do i=1,ne+1
        a(aoffset+i)=0.0
      enddo
      call ceedvectorrestorearray(x,a,aoffset,err)
      
      call ceedelemrestrictionapply(r,ceed_transpose,ceed_notranspose,y,x,&
     & ceed_request_immediate,err)

      call ceedvectorview(x,err)

      call ceedvectordestroy(x,err)
      call ceedvectordestroy(y,err)
      call ceedelemrestrictiondestroy(r,err)
      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
