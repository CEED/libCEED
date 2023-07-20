#!/usr/bin/env python3

from dataclasses import dataclass, field
import difflib
from itertools import combinations
import os
from pathlib import Path
import re
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).parent / "junit-xml"))
from junit_xml import TestCase, TestSuite


@dataclass
class TestSpec:
    name: str
    only: list = field(default_factory=list)
    args: list = field(default_factory=list)
    

def parse_test_line(line: str) -> TestSpec:
    args = line.strip().split()
    if args[0] == 'TESTARGS':
        return TestSpec(name='', args=args[1:])
    test_args = args[0][args[0].index('TESTARGS(')+9:args[0].rindex(')')]
    # transform 'name="myname",only="serial,int32"' into {'name': 'myname', 'only': 'serial,int32'}
    test_args = dict([''.join(t).split('=') for t in re.findall(r"""([^,=]+)(=)"([^"]*)\"""", test_args)])
    constraints = test_args['only'].split(',') if 'only' in test_args else []
    if len(args) > 1:
        return TestSpec(name=test_args['name'], only=constraints, args=args[1:])
    else:
        return TestSpec(name=test_args['name'], only=constraints)


def get_testargs(file : Path) -> list[TestSpec]:
    if file.suffix in ['.c', '.cpp']: comment_str = '//'
    elif file.suffix in ['.py']:      comment_str = '#'
    elif file.suffix in ['.usr']:     comment_str = 'C_'
    elif file.suffix in ['.f90']:     comment_str = '! '
    else:                             raise RuntimeError(f'Unrecognized extension for file: {file}')

    return [parse_test_line(line.strip(comment_str)) 
            for line in file.read_text().splitlines() 
            if line.startswith(f'{comment_str}TESTARGS')] or [TestSpec('', args=['{ceed_resource}'])]


def get_source(test: str) -> Path:
    prefix, rest = test.split('-', 1)
    if prefix == 'petsc':
        return (Path('examples') / 'petsc' / rest).with_suffix('.c')
    elif prefix == 'mfem':
        return (Path('examples') / 'mfem' / rest).with_suffix('.cpp')
    elif prefix == 'nek':
        return (Path('examples') / 'nek' / 'bps' / rest).with_suffix('.usr')
    elif prefix == 'fluids':
        return (Path('examples') / 'fluids' / rest).with_suffix('.c')
    elif prefix == 'solids':
        return (Path('examples') / 'solids' / rest).with_suffix('.c')
    elif test.startswith('ex'):
        return (Path('examples') / 'ceed' / test).with_suffix('.c')
    elif test.endswith('-f'):
        return (Path('tests') / test).with_suffix('.f90')
    else:
        return (Path('tests') / test).with_suffix('.c')


def check_required_failure(test_case: TestCase, stderr: str, required: str) -> None:
    if required in stderr:
        test_case.status = 'fails with required: {}'.format(required)
    else:
        test_case.add_failure_info('required: {}'.format(required))


def contains_any(resource: str, substrings: list[str]) -> bool:
    return any((sub in resource for sub in substrings))


def skip_rule(test: str, resource: str) -> bool:
    return any((
        test.startswith('t4') and contains_any(resource, ['occa']),
        test.startswith('t5') and contains_any(resource, ['occa']),
        test.startswith('ex') and contains_any(resource, ['occa']),
        test.startswith('mfem') and contains_any(resource, ['occa']),
        test.startswith('nek') and contains_any(resource, ['occa']),
        test.startswith('petsc-') and contains_any(resource, ['occa']),
        test.startswith('fluids-') and contains_any(resource, ['occa']),
        test.startswith('solids-') and contains_any(resource, ['occa']),
        test.startswith('t318') and contains_any(resource, ['/gpu/cuda/ref']),
        test.startswith('t506') and contains_any(resource, ['/gpu/cuda/shared']),
    ))


def run(test: str, backends: list[str], mode: str) -> TestSuite:
    source = get_source(test)
    test_specs = get_testargs(source)

    if mode.lower() == "tap":
        print('1..' + str(len(test_specs) * len(backends)))

    test_cases = []
    my_env = os.environ.copy()
    my_env["CEED_ERROR_HANDLER"] = 'exit'
    index = 1
    for spec in test_specs:
        for ceed_resource in backends:
            rargs = [str(Path('build') / test), *spec.args]
            rargs[rargs.index('{ceed_resource}')] = ceed_resource

            # run test
            if skip_rule(test, ceed_resource):
                test_case = TestCase(f'{test} {ceed_resource}',
                                     elapsed_sec=0,
                                     timestamp=time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime()),
                                     stdout='',
                                     stderr='')
                test_case.add_skipped_info('Pre-run skip rule')
            else:
                start = time.time()
                proc = subprocess.run(rargs,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      env=my_env)
                proc.stdout = proc.stdout.decode('utf-8')
                proc.stderr = proc.stderr.decode('utf-8')

                test_case = TestCase(f'{test} {spec.name} {ceed_resource}',
                                     classname=source.parent,
                                     elapsed_sec=time.time() - start,
                                     timestamp=time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(start)),
                                     stdout=proc.stdout,
                                     stderr=proc.stderr)
                ref_stdout = (Path('tests') / 'output' / test).with_suffix('.out')

            # check for allowed errors
            if not test_case.is_skipped() and proc.stderr:
                if 'OCCA backend failed to use' in proc.stderr:
                    test_case.add_skipped_info('occa mode not supported {} {}'.format(test, ceed_resource))
                elif 'Backend does not implement' in proc.stderr:
                    test_case.add_skipped_info('not implemented {} {}'.format(test, ceed_resource))
                elif 'Can only provide HOST memory for this backend' in proc.stderr:
                    test_case.add_skipped_info('device memory not supported {} {}'.format(test, ceed_resource))
                elif 'Test not implemented in single precision' in proc.stderr:
                    test_case.add_skipped_info('not implemented {} {}'.format(test, ceed_resource))
                elif 'No SYCL devices of the requested type are available' in proc.stderr:
                    test_case.add_skipped_info('sycl device type not available {} {}'.format(test, ceed_resource))

            # check required failures
            if not test_case.is_skipped():
                if test[:4] in ['t006', 't007']:
                    check_required_failure(test_case, proc.stderr, 'No suitable backend:')
                if test[:4] in ['t008']:
                    check_required_failure(test_case, proc.stderr, 'Available backend resources:')
                if test[:4] in ['t110', 't111', 't112', 't113', 't114']:
                    check_required_failure(test_case, proc.stderr, 'Cannot grant CeedVector array access')
                if test[:4] in ['t115']:
                    check_required_failure(test_case, proc.stderr, 'Cannot grant CeedVector read-only array access, the access lock is already in use')
                if test[:4] in ['t116']:
                    check_required_failure(test_case, proc.stderr, 'Cannot destroy CeedVector, the writable access lock is in use')
                if test[:4] in ['t117']:
                    check_required_failure(test_case, proc.stderr, 'Cannot restore CeedVector array access, access was not granted')
                if test[:4] in ['t118']:
                    check_required_failure(test_case, proc.stderr, 'Cannot sync CeedVector, the access lock is already in use')
                if test[:4] in ['t215']:
                    check_required_failure(test_case, proc.stderr, 'Cannot destroy CeedElemRestriction, a process has read access to the offset data')
                if test[:4] in ['t303']:
                    check_required_failure(test_case, proc.stderr, 'Length of input/output vectors incompatible with basis dimensions')
                if test[:4] in ['t408']:
                    check_required_failure(test_case, proc.stderr, 'CeedQFunctionContextGetData(): Cannot grant CeedQFunctionContext data access, a process has read access')
                if test[:4] in ['t409'] and contains_any(ceed_resource, ['memcheck']):
                    check_required_failure(test_case, proc.stderr, 'Context data changed while accessed in read-only mode')

            # classify other results
            if not test_case.is_skipped() and not test_case.status:
                if proc.stderr:
                    test_case.add_failure_info('stderr', proc.stderr)
                elif proc.returncode != 0:
                    test_case.add_error_info(f'returncode = {proc.returncode}')
                elif ref_stdout.is_file():
                    diff = list(difflib.unified_diff(ref_stdout.read_text().splitlines(keepends=True),
                                                     proc.stdout.splitlines(keepends=True),
                                                     fromfile=str(ref_stdout),
                                                     tofile='New'))
                    if diff:
                        test_case.add_failure_info('stdout', output=''.join(diff))
                elif proc.stdout and test[:4] not in 't003':
                    test_case.add_failure_info('stdout', output=proc.stdout)

            # store result
            test_case.args = ' '.join(rargs)
            test_cases.append(test_case)

            if mode.lower() == "tap":
                # print incremental output if TAP mode
                print('# Test: {}'.format(test_case.name.split(' ')[1]))
                print('# $ {}'.format(test_case.args))
                if test_case.is_error():
                    print('not ok {} - ERROR: {}'.format(index, (test_case.errors[0]['message'] or "NO MESSAGE").strip()))
                    print('Output: \n{}'.format((test_case.errors[0]['output'] or "NO OUTPUT").strip()))
                    if test_case.is_failure():
                        print('            FAIL: {}'.format(index, (test_case.failures[0]['message'] or "NO MESSAGE").strip()))
                        print('Output: \n{}'.format((test_case.failures[0]['output'] or "NO OUTPUT").strip()))
                elif test_case.is_failure():
                    print('not ok {} - FAIL: {}'.format(index, (test_case.failures[0]['message'] or "NO MESSAGE").strip()))
                    print('Output: \n{}'.format((test_case.failures[0]['output'] or "NO OUTPUT").strip()))
                elif test_case.is_skipped():
                    print('ok {} - SKIP: {}'.format(index, (test_case.skipped[0]['message'] or "NO MESSAGE").strip()))
                else:
                    print('ok {} - PASS'.format(index))
                sys.stdout.flush()
            else:
                # print error or failure information if JUNIT mode
                if test_case.is_error() or test_case.is_failure():
                    print('Test: {} {}'.format(test_case.name.split(' ')[0], test_case.name.split(' ')[1]))
                    print('  $ {}'.format(test_case.args))
                    if test_case.is_error():
                        print('ERROR: {}'.format((test_case.errors[0]['message'] or "NO MESSAGE").strip()))
                        print('Output: \n{}'.format((test_case.errors[0]['output'] or "NO OUTPUT").strip()))
                    if test_case.is_failure():
                        print('FAIL: {}'.format((test_case.failures[0]['message'] or "NO MESSAGE").strip()))
                        print('Output: \n{}'.format((test_case.failures[0]['output'] or "NO OUTPUT").strip()))
                sys.stdout.flush()
            index += 1

    return TestSuite(test, test_cases)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Test runner with JUnit and TAP output')
    parser.add_argument('--mode', help='Output mode, JUnit or TAP', default="JUnit")
    parser.add_argument('--output', help='Output file to write test', default=None)
    parser.add_argument('--gather', help='Gather all *.junit files into XML', action='store_true')
    parser.add_argument('test', help='Test executable', nargs='?')
    args = parser.parse_args()

    if args.gather:
        gather()
    else:
        backends = os.environ['BACKENDS'].split()

        # run tests
        result = run(args.test, backends, args.mode)

        # build output
        if args.mode.lower() == "junit":
            junit_batch = ''
            try:
                junit_batch = '-' + os.environ['JUNIT_BATCH']
            except:
                pass
            output = Path('build') / (args.test + junit_batch + '.junit') if args.output is None else Path(args.output)

            with output.open('w') as fd:
                TestSuite.to_file(fd, [result])
        elif args.mode.lower() != "tap":
            raise Exception("output mode not recognized")

        # check return code
        for t in result.test_cases:
            failures = len([c for c in result.test_cases if c.is_failure()])
            errors = len([c for c in result.test_cases if c.is_error()])
            if failures + errors > 0 and args.mode.lower() != "tap":
                sys.exit(1)
