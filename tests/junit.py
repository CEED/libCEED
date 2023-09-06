#!/usr/bin/env python3
from junit_common import *


def create_argparser() -> argparse.ArgumentParser:
    """Creates argument parser to read command line arguments

    Returns:
        argparse.ArgumentParser: Created `ArgumentParser`
    """
    parser = argparse.ArgumentParser('Test runner with JUnit and TAP output')
    parser.add_argument('-c', '--ceed-backends', type=str, nargs='*', default=['/cpu/self'], help='libCEED backend to use with convergence tests')
    parser.add_argument('-m', '--mode', type=RunMode, action=CaseInsensitiveEnumAction, help='Output mode, junit or tap', default=RunMode.JUNIT)
    parser.add_argument('-n', '--nproc', type=int, default=1, help='number of MPI processes')
    parser.add_argument('-o', '--output', type=Optional[Path], default=None, help='Output file to write test')
    parser.add_argument('-b', '--junit-batch', type=str, default='', help='Name of JUnit batch for output file')
    parser.add_argument('test', help='Test executable', nargs='?')

    return parser


# Necessary functions for running tests
class CeedSuiteSpec(SuiteSpec):
    def get_source_path(self, test: str) -> Path:
        """Compute path to test source file

        Args:
            test (str): Name of test

        Returns:
            Path: Path to source file
        """
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
        """Compute path to built test executable file

        Args:
            test (str): Name of test

        Returns:
            Path: Path to test executable
        """
        return Path('build') / test

    def get_output_path(self, test: str, output_file: str) -> Path:
        """Compute path to expected output file

        Args:
            test (str): Name of test
            output_file (str): File name of output file

        Returns:
            Path: Path to expected output file
        """
        return Path('tests') / 'output' / output_file

    def check_pre_skip(self, test: str, spec: TestSpec, resource: str, nproc: int) -> Optional[str]:
        """Check if a test case should be skipped prior to running, returning the reason for skipping

        Args:
            test (str): Name of test
            spec (TestSpec): Test case specification
            resource (str): libCEED backend
            nproc (int): Number of MPI processes to use when running test case

        Returns:
            Optional[str]: Skip reason, or `None` if test case should not be skipped
        """
        if contains_any(resource, ['occa']) and startswith_any(test, ['t4', 't5', 'ex', 'mfem', 'nek', 'petsc', 'fluids', 'solids']):
            return 'OCCA mode not supported'
        if test.startswith('t318') and contains_any(resource, ['/gpu/cuda/ref']):
            return 'CUDA ref backend not supported'
        if test.startswith('t506') and contains_any(resource, ['/gpu/cuda/shared']):
            return 'CUDA shared backend not supported'

    def check_post_skip(self, test: str, spec: TestSpec, resource: str, stderr: str) -> Optional[str]:
        """Check if a test case should be allowed to fail, based on its stderr output

        Args:
            test (str): Name of test
            spec (TestSpec): Test case specification
            resource (str): libCEED backend
            stderr (str): Standard error output from test case execution

        Returns:
            Optional[str]: Skip reason, or `None` if unexpeced error
        """
        if 'OCCA backend failed to use' in stderr:
            return f'OCCA mode not supported'
        elif 'Backend does not implement' in stderr:
            return f'Backend does not implement'
        elif 'Can only provide HOST memory for this backend' in stderr:
            return f'Device memory not supported'
        elif 'Can only set HOST memory for this backend' in stderr:
            return f'Device memory not supported'
        elif 'Test not implemented in single precision' in stderr:
            return f'Test not implemented in single precision'
        elif 'No SYCL devices of the requested type are available' in stderr:
            return f'SYCL device type not available'
        return None

    def check_required_failure(self, test: str, spec: TestSpec, resource: str, stderr: str) -> tuple[str, bool]:
        """Check whether a test case is expected to fail and if it failed expectedly

        Args:
            test (str): Name of test
            spec (TestSpec): Test case specification
            resource (str): libCEED backend
            stderr (str): Standard error output from test case execution

        Returns:
            tuple[str, bool]: Tuple of the expected failure string and whether it was present in `stderr`
        """
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
        """Check whether a test is allowed to print console output

        Args:
            test (str): Name of test

        Returns:
            bool: True if the test is allowed to print console output
        """
        return test[:4] in ['t003']


if __name__ == '__main__':
    args = create_argparser().parse_args()

    # run tests
    if 'smartsim' in args.test:
        sys.path.insert(0, str(Path(__file__).parents[1] / "examples" / "fluids"))
        from smartsim_regression_framework import setup, teardown, test_junit  # nopep8

        setup(Path(__file__).parent / 'test_dir')
        results = []
        print(f'1..{len(args.ceed_backends)}')
        for i, backend in enumerate(args.ceed_backends):
            results.append(test_junit(args.ceed_backends))
            print_test_case(results[i], TestSpec("SmartSim Tests"), args.mode, i)
        teardown(Path(__file__).parent / 'test_dir')
        result: TestSuite = TestSuite('SmartSim Tests', results)
    else:
        result: TestSuite = run_tests(args.test, args.ceed_backends, args.mode, args.nproc, CeedSuiteSpec())

    # write output and check for failures
    if args.mode is RunMode.JUNIT:
        write_junit_xml(result, args.output, args.junit_batch)
        if has_failures(result):
            sys.exit(1)
