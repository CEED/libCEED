#!/usr/bin/env python3

import argparse
from junit_common import *


def contains_any(resource: str, substrings: list[str]) -> bool:
    return any((sub in resource for sub in substrings))


# create parser for command line arguments
def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('Test runner with JUnit and TAP output')
    parser.add_argument('-c', '--ceed-backends', type=str, nargs='*', default=['/cpu/self'], help='libCEED backend to use with convergence tests')
    parser.add_argument('-m', '--mode', help='Output mode, JUnit or TAP', default='JUnit')
    parser.add_argument('-n', '--nproc', type=int, default=1, help='number of MPI processes')
    parser.add_argument('-o', '--output', help='Output file to write test', default=None)
    parser.add_argument('test', help='Test executable', nargs='?')

    return parser


# Necessary functions for running tests
class CeedSuiteSpec(SuiteSpec):
    def get_source_path(self, test: str) -> Path:
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

    # get path to executable
    def get_run_path(self, test: str) -> Path:
        return Path('build') / test

    def get_output_path(self, test: str, output_file: str) -> Path:
        return Path('tests') / 'output' / output_file

    def check_pre_skip(self, test: str, spec: TestSpec, resource: str, nproc: int) -> Optional[str]:
        return 'Pre-Check Skipped' if any((
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
        )) else None

    def check_post_skip(self, test: str, spec: TestSpec, resource: str, stderr: str) -> Optional[str]:
        if 'OCCA backend failed to use' in stderr:
            return f'occa mode not supported {test} {resource}'
        elif 'Backend does not implement' in stderr:
            return f'not implemented {test} {resource}'
        elif 'Can only provide HOST memory for this backend' in stderr:
            return f'device memory not supported {test} {resource}'
        elif 'Test not implemented in single precision' in stderr:
            return f'not implemented {test} {resource}'
        elif 'No SYCL devices of the requested type are available' in stderr:
            return f'sycl device type not available {test} {resource}'
        return None

    def check_required_failure(self, test: str, spec: TestSpec, resource: str, stderr: str) -> tuple[str, bool]:
        test_id: str = test[:4]
        fail_str: str = ''
        if test_id in ['t006', 't007']:
            fail_str = 'No suitable backend:'
        elif test_id in ['t008']:
            fail_str = 'Available backend resources:'
        elif test_id in ['t110', 't111', 't112', 't113', 't114']:
            fail_str = 'Cannot grant CeedVector array access'
        elif test_id in ['t115']:
            fail_str = 'Cannot grant CeedVector read-only array access, the access lock is already in use'
        elif test_id in ['t116']:
            fail_str = 'Cannot destroy CeedVector, the writable access lock is in use'
        elif test_id in ['t117']:
            fail_str = 'Cannot restore CeedVector array access, access was not granted'
        elif test_id in ['t118']:
            fail_str = 'Cannot sync CeedVector, the access lock is already in use'
        elif test_id in ['t215']:
            fail_str = 'Cannot destroy CeedElemRestriction, a process has read access to the offset data'
        elif test_id in ['t303']:
            fail_str = 'Length of input/output vectors incompatible with basis dimensions'
        elif test_id in ['t408']:
            fail_str = 'CeedQFunctionContextGetData(): Cannot grant CeedQFunctionContext data access, a process has read access'
        elif test_id in ['t409'] and contains_any(resource, ['memcheck']):
            fail_str = 'Context data changed while accessed in read-only mode'

        return fail_str, fail_str in stderr

    def check_allowed_stdout(self, test: str) -> bool:
        return test[:4] in ['t003']


if __name__ == '__main__':
    args = create_argparser().parse_args()

    # run tests
    mode: RunMode = RunMode(args.mode.lower())
    result: TestSuite = run_tests(args.test, args.ceed_backends, mode, args.nproc, CeedSuiteSpec())

    # build output
    if mode is RunMode.JUNIT:
        junit_batch: str = f'-{os.environ["JUNIT_BATCH"]}' if 'JUNIT_BATCH' in os.environ else ''
        output: Path = Path('build') / (args.test + junit_batch + '.junit') if args.output is None else Path(args.output)
        with output.open('w') as fd:
            TestSuite.to_file(fd, [result])

    # check return code
    for t in result.test_cases:
        any_failures: bool = any(c.is_failure() or c.is_error() for c in result.test_cases)
        if any_failures and mode is not RunMode.TAP:
            sys.exit(1)
