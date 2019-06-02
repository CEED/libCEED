# Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
# All Rights reserved. See files LICENSE and NOTICE for details.
#
# This file is part of CEED, a collection of benchmarks, miniapps, software
# libraries and APIs for efficient high-order finite element and spectral
# element discretizations for exascale applications. For more information and
# source code availability see http://github.com/ceed.
#
# The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
# a collaborative effort of two U.S. Department of Energy organizations (Office
# of Science and the National Nuclear Security Administration) responsible for
# the planning and preparation of a capable exascale ecosystem, including
# software, applications, hardware, advanced system engineering and early
# testbed platforms, in support of the nation's exascale computing imperative.

from sys import stdout as out
import fileinput
import pprint

#####   Read all input files specified on the command line, or stdin and parse
#####   the content, storing it as a list of dictionaries - one dictionary for
#####   each test run.

it=fileinput.input()
state=0
line=''
i=0
mesh_p=0
config='unknown'
backend='unknown'
test='unknown'
num_procs=0
num_procs_node=0
lnfmt='%05i'
data={}
runs=[]
while True:
   ##
   if state%2==0:
      ##
      try:
         line=next(it)
         i=i+1
      except StopIteration:
         break
      state=state+1
      ##
   elif state==1:
      ##
      state=0
      if 'Reading configuration' in line:
         ##
         ## This is the beginning of a new file.
         ##
         config=line.split()[2]
         num_procs=0
         num_procs_node=0
      ## Number of MPI tasks
      elif 'Running the tests using a total of' in line:
         num_procs=int(line.split('a total of ',1)[1].split(None,1)[0])
      ## MPI tasks per node
      elif 'tasks per node' in line:
         num_procs_node=int(line.split(' tasks per',1)[0].rsplit(None,1)[1])
      elif line == 'Running test:\n':
         ##
         ## This is the beginning of a new run.
         ##
         if 'cg-iteration-dps' in data:
            runs.append(data)
         data={}
         data['file']=fileinput.filename()
         data['config']=config
         data['backend']=backend
         data['test']=test
         data['num-procs']=num_procs
         data['num-procs-node']=num_procs_node
         data['order']=mesh_p
         data['quadrature-pts']=mesh_p
         data['code']="libCEED"
         test_=test.rsplit('/',1)[-1]
         data['case']='scalar'
      ## Benchmark Problem
      elif "CEED Benchmark Problem" in line:
        data['test'] = line.split('-- ')[1]
        data['case']='scalar' if (('Problem 1' in line) or ('Problem 3' in line)
                                  or ('Problem 5' in line)) else 'vector'
      ## Backend
      elif 'libCEED Backend' in line:
         data['backend']=line.split(':')[1].strip()
      ## P
      elif 'Basis Nodes' in line:
         data['order']=int(line.split(':')[1])
      ## Q
      elif 'Quadrature Points' in line:
         qpts=int(line.split(':')[1])
         data['quadrature-pts']=qpts**3
      ## Total DOFs
      elif 'Global DOFs' in line:
         data['num-unknowns']=int(line.split(':')[1])
      ## Number of elements
      elif 'Local Elements' in line:
         data['num-elem']=int(line.split(':')[1].split()[0])*data['num-procs']
      ## CG DOFs/Sec
      elif 'DOFs/Sec in CG' in line:
         data['cg-iteration-dps']=1e6*float(line.split(':')[1].split()[0])
      ## End of output

if 'cg-iteration-dps' in data:
   runs.append(data)

print('Number of test runs read: %i'%len(runs))
