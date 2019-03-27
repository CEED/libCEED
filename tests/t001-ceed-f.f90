!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,mtype,err
      character arg*32
      mtype = 1024


      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)

      call ceedgetpreferredmemtype(ceed,mtype,err)

      if (mtype == 1024) then
          write(*,*) 'Error getting preferred memory type.'
      endif

      call ceeddestroy(ceed,err)

      end
!-----------------------------------------------------------------------
