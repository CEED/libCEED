!-----------------------------------------------------------------------
      program test

      include 'ceedf.h'

      integer ceed,err
      integer qf_setup,qf_mass
      character arg*32

      call getarg(1,arg)
      call ceedinit(trim(arg)//char(0),ceed,err)

      call ceedqfunctioncreateinteriorbyname(ceed,'Mass1DBuild',qf_setup,err)
      call ceedqfunctioncreateinteriorbyname(ceed,'MassApply',qf_mass,err)

      call ceedqfunctionview(qf_setup,err)
      call ceedqfunctionview(qf_mass,err)

      call ceedqfunctiondestroy(qf_setup,err)
      call ceedqfunctiondestroy(qf_mass,err)
      call ceeddestroy(ceed,err)
      end
!-----------------------------------------------------------------------
