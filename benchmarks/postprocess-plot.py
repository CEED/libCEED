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


#####   Adjustable plot parameters:
log_y=0               # use log scale on the y-axis?
x_range=(1e1,4e6)     # plot range for the x-axis; comment out for auto
y_range=(0,8e7)       # plot range for the y-axis; comment out for auto
draw_iter_lines=0     # draw the "iter/s" lines?
ymin_iter_lines=3e5   # minimal y value for the "iter/s" lines
ymax_iter_lines=8e8   # maximal y value for the "iter/s" lines
legend_ncol=(2 if log_y else 1)   # number of columns in the legend
write_figures=1       # save the figures to files?
show_figures=1        # display the figures on the screen?


#####   Load the data
exec(compile(open('postprocess-base.py').read(), 'postprocess-base.py', 'exec'))


#####   Sample plot output

from matplotlib import use
if not show_figures:
   use('pdf')
from pylab import *

rcParams['font.sans-serif'].insert(0,'Noto Sans')
rcParams['font.sans-serif'].insert(1,'Open Sans')
rcParams['figure.figsize']=[10, 8] # default: 8 x 6

cm=get_cmap('Set1') # 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3'
if '_segmentdata' in cm.__dict__:
   cm_size=len(cm.__dict__['_segmentdata']['red'])
elif 'colors' in cm.__dict__:
   cm_size=len(cm.__dict__['colors'])
colors=[cm(1.*i/(cm_size-1)) for i in range(cm_size)]

# colors=['blue','green','crimson','turquoise','m','gold','cornflowerblue',
#         'darkorange']

sel_runs=runs

tests=list(set([run['test'].rsplit('/',1)[-1].rsplit('.sh',1)[0]
                for run in sel_runs]))
# print 'Present tests:', tests
test=tests[0]
print('Using test:', test)
test_short=test
sel_runs=[run for run in sel_runs if
          run['test'].rsplit('/',1)[-1].rsplit('.sh',1)[0]==test]

if 'case' in sel_runs[0]:
   cases=list(set([run['case'] for run in sel_runs]))
   case=cases[0]
   vdim=1 if case=='scalar' else 3
   print('Using case:', case)
   sel_runs=[run for run in sel_runs if run['case']==case]

codes = list(set([run['code'] for run in sel_runs]))
code  = codes[0]
sel_runs=[run for run in sel_runs if run['code']==code]

pl_set=[(run['backend'],run['num-procs'],run['num-procs-node'])
        for run in sel_runs]
pl_set=sorted(set(pl_set))
print()
pprint.pprint(pl_set)

for plt in pl_set:
   backend=plt[0]
   num_procs=plt[1]
   num_procs_node=plt[2]
   num_nodes=num_procs/num_procs_node
   pl_runs=[run for run in sel_runs
            if run['backend']==backend and
               run['num-procs']==num_procs and
               run['num-procs-node']==num_procs_node]
   if len(pl_runs)==0:
      continue

   print()
   print('backend: %s, compute nodes: %i, number of MPI tasks = %i'%(
      backend,num_nodes,num_procs))

   figure()
   i=0
   sol_p_set=sorted(set([run['order'] for run in pl_runs]))
   for sol_p in sol_p_set:
      qpts=sorted(list(set([run['quadrature-pts'] for run in pl_runs
                            if run['order']==sol_p])))
      qpts.reverse()
      print('Order: %i, quadrature points:'%sol_p, qpts)
      qpts_1d=[int(q**(1./3)+0.5) for q in qpts]

      d=[[run['order'],run['num-elem'],1.*run['num-unknowns']/num_nodes/vdim,
          run['cg-iteration-dps']/num_nodes]
         for run in pl_runs
         if run['order']==sol_p and
            run['quadrature-pts']==qpts[0]]
      # print
      # print 'order = %i'%sol_p
      # pprint.pprint(sorted(d))
      d=[[e[2],e[3]] for e in d if e[0]==sol_p]
      # (DOFs/[sec/iter]/node)/(DOFs/node) = iter/sec
      d=[[nun,
          min([e[1] for e in d if e[0]==nun]),
          max([e[1] for e in d if e[0]==nun])]
         for nun in set([e[0] for e in d])]
      d=asarray(sorted(d))
      plot(d[:,0],d[:,2],'o-',color=colors[i%cm_size],
           label='p=%i'%(sol_p))
      if list(d[:,1]) != list(d[:,2]):
         plot(d[:,0],d[:,1],'o-',color=colors[i])
         fill_between(d[:,0],d[:,1],d[:,2],facecolor=colors[i],alpha=0.2)
      ##
      if len(qpts)==1:
         i=i+1
         continue
      d=[[run['order'],run['num-elem'],1.*run['num-unknowns']/num_nodes/vdim,
          run['cg-iteration-dps']/num_nodes]
         for run in pl_runs
         if run['order']==sol_p and (run['quadrature-pts']==qpts[1])]
      d=[[e[2],e[3]] for e in d if e[0]==sol_p]
      if len(d)==0:
         i=i+1
         continue
      d=[[nun,
          min([e[1] for e in d if e[0]==nun]),
          max([e[1] for e in d if e[0]==nun])]
         for nun in set([e[0] for e in d])]
      d=asarray(sorted(d))
      plot(d[:,0],d[:,2],'s--',color=colors[i],
           label='p=%i, q=p+%i'%(sol_p,qpts_1d[1]-sol_p))
      if list(d[:,1]) != list(d[:,2]):
         plot(d[:,0],d[:,1],'s--',color=colors[i])
      ##
      i=i+1
   ##
   if draw_iter_lines:
      y0,y1=ymin_iter_lines,ymax_iter_lines
      y=asarray([y0,y1]) if log_y else exp(linspace(log(y0), log(y1)))
      slope1=600.
      slope2=6000.
      plot(y/slope1,y,'k--',label='%g iter/s'%(slope1/vdim))
      plot(y/slope2,y,'k-',label='%g iter/s'%(slope2/vdim))

   title('Config: %s (%i node%s, %i tasks/node), %s, %s'%(
         code,num_nodes,'' if num_nodes==1 else 's',
         num_procs_node,backend,test_short))
   xscale('log') # subsx=[2,4,6,8]
   if log_y:
      yscale('log')
   if 'x_range' in vars() and len(x_range)==2:
      xlim(x_range)
   if 'y_range' in vars() and len(y_range)==2:
      ylim(y_range)
   # rng=arange(1e7,1.02e8,1e7)
   # yticks(rng,['%i'%int(v/1e6) for v in rng])
   # ylim(min(rng),max(rng))
   # xlim(0.5,max([run['order'] for run in pl_runs])+0.5)
   grid('on', color='gray', ls='dotted')
   grid('on', axis='both', which='minor', color='gray', ls='dotted')
   gca().set_axisbelow(True)
   xlabel('Points per compute node')
   ylabel('[DOFs x CG iterations] / [compute nodes x seconds]')
   legend(ncol=legend_ncol, loc='best')

   if write_figures: # write .pdf file?
      shortbackend=backend.replace('/','')
      pdf_file='plot_%s_%s_%s_N%03i_pn%i.pdf'%(
               code,test_short,shortbackend,num_nodes,num_procs_node)
      print('saving figure --> %s'%pdf_file)
      savefig(pdf_file, format='pdf', bbox_inches='tight')

if show_figures: # show the figures?
   print('\nshowing figures ...')
   show()
