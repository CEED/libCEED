      subroutine f_mass(ctx, qdata, q, u, v, err)
      integer q, err, i
      real ctx, qdata(q), u(1,q), v(1,q)

      do i=1,q
        v(1,i) = qdata(i)*u(1,i)
      enddo

      end

      program ex1

      external mass, f_mass

      integer ceedh, err
      integer uh, rh, xcoordh, qdatah
      integer nelem, esize, ndof, eindices(123*125)
      integer basish
      integer erstrh
      integer massh
      integer op_massh, op_poisson3dh, op_buildcoeffsh

      nelem=8
      esize=64
      ndof =343

c     TODO: get rid of //char(0), essentially we need to convert
c     fortran strings to c-strings
      call ceedinit('/cpu/self'//char(0),ceedh,err)

      call ceedvectorcreate(ceedh,ndof  ,uh     ,err)
      call ceedvectorcreate(ceedh,ndof  ,rh     ,err)
      call ceedvectorcreate(ceedh,ndof*3,xcoordh,err)

      call ceedelemrestrictioncreate(ceedh,nelem,esize,ndof,0,
     $  1,eindices,erstrh,err)

      call ceedbasiscreatetensorh1lagrange(ceedh,3,1,4,4,0,
     $  basish,err)

c     TODO: get rid of //char(0), essentially we need to convert
c     fortran strings to c-strings
      call ceedqfunctioncreateinterior(ceedh,1,1,8,1,1,f_mass,
     $  't40-fortran.f:f_mass'//char(0),massh,err)

      call ceedvectordestroy(uh     ,err)
      call ceedvectordestroy(rh     ,err)
      call ceedvectordestroy(xcoordh,err)
      call ceedelemrestrictiondestroy(erstrh,err)
      call ceedbasisdestroy(basish,err)
      call ceedqfunctiondestroy(massh,err)
      call ceeddestroy(ceedh,err)

      end
