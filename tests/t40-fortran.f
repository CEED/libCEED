      program ex1

      integer ceedh, err
      integer uh, rh, xcoordh, qdatah
      integer nelem, esize, ndof, eindices(123*125)
      integer b_basish
      integer er_restricth
      integer qf_massh, qf_poisson3dh, qf_buildcoeffsh
      integer op_massh, op_poisson3dh, op_buildcoeffsh

      character arg*32

      nelem=8
      esize=64
      ndof =343

      call getarg(1, arg)
      call ceedinit(trim(arg),ceedh,err)

      call ceedvectorcreate(ceedh,ndof  ,uh     ,err)
      call ceedvectorcreate(ceedh,ndof  ,rh     ,err)
      call ceedvectorcreate(ceedh,ndof*3,xcoordh,err)

      call ceedelemrestrictioncreate(ceedh, nelem, esize, ndof, 0,
     $  1, eindices, er_restricth)

      call ceedbasiscreatetensorh1lagrange(ceedh, 3, 1, 4, 4, 0,
     $  b_basish)

      end
