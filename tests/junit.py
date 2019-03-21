#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'junit-xml')))
from junit_xml import TestCase, TestSuite

def parse_testargs(file):
    if os.path.splitext(file)[1] in ['.c', '.cpp']:
        return sum([line.split()[1:] for line in open(file).readlines()
                    if line.startswith('//TESTARGS')], [])
    elif os.path.splitext(file)[1] == '.usr':
        return sum([line.split()[2:] for line in open(file).readlines()
                    if line.startswith('C TESTARGS')], [])
    raise RuntimeError('Unrecognized extension for file: {}'.format(file))

def get_source(test):
    if test.startswith('petsc-'):
        return os.path.join('examples', 'petsc', test[6:] + '.c')
    elif test.startswith('mfem-'):
        return os.path.join('examples', 'mfem', test[5:] + '.cpp')
    elif test.startswith('nek-'):
        return os.path.join('examples', 'nek5000', test[4:] + '.usr')
    elif test.startswith('ex'):
        return os.path.join('examples', 'ceed', test + '.c')

def get_testargs(test):
    source = get_source(test)
    if source is None:
        return ['{ceed_resource}']
    return parse_testargs(source)

def run(test, backends):
    import subprocess
    import time
    import difflib
    args = get_testargs(test)
    testcases = []
    for ceed_resource in backends:
        rargs = [os.path.join('build', test)] + args.copy()
        rargs[rargs.index('{ceed_resource}')] = ceed_resource
        start = time.time()
        proc = subprocess.run(rargs,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              encoding='utf-8')

        case = TestCase('{} {}'.format(test, ceed_resource),
                        elapsed_sec=time.time()-start,
                        timestamp=time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(start)),
                        stdout=proc.stdout,
                        stderr=proc.stderr)
        ref_stdout = os.path.join('output', test + '.out')
        if proc.stderr:
            if 'OCCA backend failed to use' in proc.stderr:
                case.add_skipped_info('occa mode not supported {} {}'.format(test, ceed_resource))
            elif 'Backend does not implement' in proc.stderr:
                case.add_skipped_info('not implemented {} {}'.format(test, ceed_resource))
            elif 'access' in proc.stderr and test[:4] in 't103 t104 t105 t106 t107'.split():
                case.add_skipped_info('expected failure')
            elif 'vectors incompatible' in proc.stderr and test[:4] in ['t308']:
                case.add_skipped_info('expected failure')
            else:
                case.add_failure_info('stderr', proc.stderr)
        if not case.is_skipped():
            if proc.returncode != 0:
                case.add_error_info('returncode = {}'.format(proc.returncode))
            elif os.path.isfile(ref_stdout):
                with open(ref_stdout) as ref:
                    diff = list(difflib.unified_diff(ref.readlines(),
                                                     proc.stdout.splitlines(keepends=True),
                                                     fromfile=ref_stdout,
                                                     tofile='New'))
                if diff:
                    case.add_failure_info('stdout', output=''.join(diff))
            elif proc.stdout:
                case.add_failure_info('stdout', output=proc.stdout)
        testcases.append(case)
    return TestSuite(test, testcases)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Test runner with JUnit output')
    parser.add_argument('--output', help='Output file to write test', default=None)
    parser.add_argument('--gather', help='Gather all *.junit files into XML', action='store_true')
    parser.add_argument('test', help='Test executable', nargs='?')
    args = parser.parse_args()

    if args.gather:
        gather()
    else:
        backends = os.environ['BACKENDS'].split()

        result = run(args.test, backends)
        output = (os.path.join('build', args.test + '.junit')
                  if args.output is None
                  else args.output)
        with open(output, 'w') as fd:
            TestSuite.to_file(fd, [result])
    
