      program test

      integer ceed,err
      character arg*32

      call getarg(1,arg)

      call ceedinit(trim(arg)//char(0),ceed,err)
      call ceeddestroy(ceed,err)

      end
