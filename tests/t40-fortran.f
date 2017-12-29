      program fortran1
      integer ceed, ndof, u, err
      character arg*32

      call getarg(1, arg)
      call ceedinit(arg,ceed,err)

      call ceedvectorcreate(ceed,ndof,u)

      end
