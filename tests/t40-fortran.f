      program fortran1
      integer ceed, ndof, u, err
      character arg*32

      ceed = -1
      ndof = 3

      call getarg(1, arg)

      call ceedinit(trim(arg),ceed,err)

      call ceedvectorcreate(ceed,ndof,u,err)

      end
