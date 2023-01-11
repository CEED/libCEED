GMSH = gmsh

ALL = $(foreach q,1,$(foreach n,20 40,cylinder-q$(q)-n$(n).msh))

all : $(ALL)

cylinder-q1-n20.msh:

cylinder-q1-n20.msh: GMSH_FLAGS = -order 1 -setnumber N 20
cylinder-q1-n40.msh: GMSH_FLAGS = -order 1 -setnumber N 40
cylinder-q2-n20.msh: GMSH_FLAGS = -order 2 -setnumber N 20
cylinder-q2-n40.msh: GMSH_FLAGS = -order 2 -setnumber N 40
cylinder-q1-n12-recirc.msh: GMSH_FLAGS = -order 1 -setnumber N 12 -setnumber H 1 -setnumber xL 1 -setnumber xR .1

cylinder%.msh: cylinder.geo
	$(GMSH) -3 $(GMSH_FLAGS) $< -o $@

clean:
	$(RM) $(ALL)

print-% :
	$(info [ variable name]: $*)
	$(info [        origin]: $(origin $*))
	$(info [        flavor]: $(flavor $*))
	$(info [         value]: $(value $*))
	$(info [expanded value]: $($*))
	$(info )
	@true
