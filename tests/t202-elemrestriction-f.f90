!-----------------------------------------------------------------------
      program test
      implicit none
      include 'ceed/fortran.h'

      integer ceed,err
      integer x,y
      integer r
      integer i,j,k

      integer ne,elemsize,nb,blksize
      parameter(ne=8,elemsize=2,nb=2,blksize=5)
      integer ind(elemsize*ne)
      integer layout(3)
      integer blk,elem,indx

      real*8 a(ne+1)
      real*8 yy(nb*blksize*elemsize)
      real*8 xx(ne+1)
      real*8 diff
      integer*8 aoffset,xoffset,yoffset

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
      call ceedelemrestrictioncreateblocked(ceed,ne,elemsize,blksize,1,1,ne+1,&
     & ceed_mem_host,ceed_use_pointer,ind,r,err)

      call ceedvectorcreate(ceed,nb*blksize*elemsize,y,err);
      call ceedvectorsetvalue(y,0.d0,err)

!     NoTranspose
      call ceedelemrestrictionapply(r,ceed_notranspose,x,y,&
     & ceed_request_immediate,err)
      call ceedvectorgetarrayread(y,ceed_mem_host,yy,yoffset,err)
      call ceedelemrestrictiongetelayout(r,layout,err)
      do i=0,elemsize-1
        do j=0,0
          do k=0,ne-1
            blk = k/blksize
            elem = mod(k,blksize)
            indx = (i*blksize+elem)*layout(1)+j*layout(2)*blksize+blk*layout(3)*blksize
            diff=yy(yoffset+indx+1)
            diff=diff-a(ind(k*elemsize+i+1)+1)
            if (abs(diff) > 1.0D-15) then
! LCOV_EXCL_START
             write(*,*) 'Error in restricted array y(',i,')(',j,')(',k,')=',&
     &         yy(yoffset+indx+1)
! LCOV_EXCL_STOP
            endif
          enddo
        enddo
      enddo
      call ceedvectorrestorearrayread(y,yy,yoffset,err)

!     Transpose
      call ceedvectorsetvalue(x,0.d0,err)
      call ceedelemrestrictionapply(r,ceed_transpose,y,x,&
     & ceed_request_immediate,err)
      call ceedvectorgetarrayread(x,ceed_mem_host,xx,xoffset,err)
      do i=1,ne+1
        diff=xx(xoffset+i)
        if (i > 1 .and. i < ne+1) then
          diff=diff-2*(10+i-1)
        else
          diff=diff-(10+i-1)
        endif
        if (abs(diff) > 1.0D-15) then
! LCOV_EXCL_START
         write(*,*) 'Error in restricted array x(',i,')=',&
     &         xx(xoffset+i)
! LCOV_EXCL_STOP
        endif
      enddo
      call ceedvectorrestorearrayread(x,xx,xoffset,err)

      call ceedvectordestroy(x,err)
      call ceedvectordestroy(y,err)
      call ceedelemrestrictiondestroy(r,err)
      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
