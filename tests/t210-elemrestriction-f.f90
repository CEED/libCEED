!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer r
      integer i

      integer ne
      parameter(ne=3)

      integer*4 ind(2*ne)

      character arg*32

      call getarg(1,arg)
      call ceedinit(trim(arg)//char(0),ceed,err)

      do i=1,ne
        ind(2*i-1)=i-1
        ind(2*i  )=i
      enddo

      call ceedelemrestrictioncreate(ceed,ne,2,ne+1,1,ceed_mem_host,&
     & ceed_use_pointer,ind,r,err)

      call ceedelemrestrictionview(r,err)

      call ceedelemrestrictiondestroy(r,err)
      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
