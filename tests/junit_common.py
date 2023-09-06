from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass, field
import difflib
from enum import Enum
from math import isclose
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent / "junit-xml"))
from junit_xml import TestCase, TestSuite, to_xml_report_string  # nopep8


class CaseInsensitiveEnumAction(argparse.Action):
    """Action to convert input values to lower case prior to converting to an Enum type"""

    def __init__(self, option_strings, dest, type, default, **kwargs):
        if not (issubclass(type, Enum) and issubclass(type, str)):
            raise ValueError(f"{type} must be a StrEnum or str and Enum")
        # store provided enum type
        self.enum_type = type
        if isinstance(default, str):
            default = self.enum_type(default.lower())
        else:
            default = [self.enum_type(v.lower()) for v in default]
        # prevent automatic type conversion
        super().__init__(option_strings, dest, default=default, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, str):
            values = self.enum_type(values.lower())
        else:
            values = [self.enum_type(v.lower()) for v in values]
        setattr(namespace, self.dest, values)


@dataclass
class TestSpec:
    """Dataclass storing information about a single test case"""
    name: str
    only: list = field(default_factory=list)
    args: list = field(default_factory=list)


class RunMode(str, Enum):
    """Enumeration of run modes, either `RunMode.TAP` or `RunMode.JUNIT`"""
    __str__ = str.__str__
    __format__ = str.__format__
    TAP: str = 'tap'
    JUNIT: str = 'junit'


class SuiteSpec(ABC):
    """Abstract Base Class defining the required interface for running a test suite"""
    @abstractmethod
    def get_source_path(self, test: str) -> Path:
        """Compute path to test source file

        Args:
            test (str): Name of test

        Returns:
            Path: Path to source file
        """
        raise NotImplementedError

    @abstractmethod
    def get_run_path(self, test: str) -> Path:
        """Compute path to built test executable file

        Args:
            test (str): Name of test

        Returns:
            Path: Path to test executable
        """
        raise NotImplementedError

    @abstractmethod
    def get_output_path(self, test: str, output_file: str) -> Path:
        """Compute path to expected output file

        Args:
            test (str): Name of test
            output_file (str): File name of output file

        Returns:
            Path: Path to expected output file
        """
        raise NotImplementedError

    def post_test_hook(self, test: str, spec: TestSpec) -> None:
        """Function callback ran after each test case

        Args:
            test (str): Name of test
            spec (TestSpec): Test case specification
        """
        pass

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
        return None

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
        return '', True

    def check_allowed_stdout(self, test: str) -> bool:
        """Check whether a test is allowed to print console output

        Args:
            test (str): Name of test

        Returns:
            bool: True if the test is allowed to print console output
        """
        return False


def has_cgnsdiff() -> bool:
    """Check whether `cgnsdiff` is an executable program in the current environment

    Returns:
        bool: True if `cgnsdiff` is found
    """
    my_env: dict = os.environ.copy()
    proc = subprocess.run('cgnsdiff',
                          shell=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          env=my_env)
    return 'not found' not in proc.stderr.decode('utf-8')


def contains_any(base: str, substrings: list[str]) -> bool:
    """Helper function, checks if any of the substrings are included in the base string

    Args:
        base (str): Base string to search in
        substrings (list[str]): List of potential substrings

    Returns:
        bool: True if any substrings are included in base string
    """
    return any((sub in base for sub in substrings))


def startswith_any(base: str, prefixes: list[str]) -> bool:
    """Helper function, checks if the base string is prefixed by any of `prefixes`

    Args:
        base (str): Base string to search
        prefixes (list[str]): List of potential prefixes

    Returns:
        bool: True if base string is prefixed by any of the prefixes
    """
    return any((base.startswith(prefix) for prefix in prefixes))


def parse_test_line(line: str) -> TestSpec:
    """Parse a single line of TESTARGS and CLI arguments into a `TestSpec` object

    Args:
        line (str): String containing TESTARGS specification and CLI arguments

    Returns:
        TestSpec: Parsed specification of test case
    """
    args: list[str] = re.findall("(?:\".*?\"|\\S)+", line.strip())
    if args[0] == 'TESTARGS':
        return TestSpec(name='', args=args[1:])
    raw_test_args: str = args[0][args[0].index('TESTARGS(') + 9:args[0].rindex(')')]
    # transform 'name="myname",only="serial,int32"' into {'name': 'myname', 'only': 'serial,int32'}
    test_args: dict = dict([''.join(t).split('=') for t in re.findall(r"""([^,=]+)(=)"([^"]*)\"""", raw_test_args)])
    constraints: list[str] = test_args['only'].split(',') if 'only' in test_args else []
    if len(args) > 1:
        return TestSpec(name=test_args['name'], only=constraints, args=args[1:])
    else:
        return TestSpec(name=test_args['name'], only=constraints)


def get_test_args(source_file: Path) -> list[TestSpec]:
    """Parse all test cases from a given source file

    Args:
        source_file (Path): Path to source file

    Raises:
        RuntimeError: Errors if source file extension is unsupported

    Returns:
        list[TestSpec]: List of parsed `TestSpec` objects, or a list containing a single, default `TestSpec` if none were found
    """
    comment_str: str = ''
    if source_file.suffix in ['.c', '.cpp']:
        comment_str = '//'
    elif source_file.suffix in ['.py']:
        comment_str = '#'
    elif source_file.suffix in ['.usr']:
        comment_str = 'C_'
    elif source_file.suffix in ['.f90']:
        comment_str = '! '
    else:
        raise RuntimeError(f'Unrecognized extension for file: {source_file}')

    return [parse_test_line(line.strip(comment_str))
            for line in source_file.read_text().splitlines()
            if line.startswith(f'{comment_str}TESTARGS')] or [TestSpec('', args=['{ceed_resource}'])]


def diff_csv(test_csv: Path, true_csv: Path, zero_tol: float = 3e-10, rel_tol: float = 1e-2) -> str:
    """Compare CSV results against an expected CSV file with tolerances

    Args:
        test_csv (Path): Path to output CSV results
        true_csv (Path): Path to expected CSV results
        zero_tol (float, optional): Tolerance below which values are considered to be zero. Defaults to 3e-10.
        rel_tol (float, optional): Relative tolerance for comparing non-zero values. Defaults to 1e-2.

    Returns:
        str: Diff output between result and expected CSVs
    """
    test_lines: list[str] = test_csv.read_text().splitlines()
    true_lines: list[str] = true_csv.read_text().splitlines()

    if test_lines[0] != true_lines[0]:
        return ''.join(difflib.unified_diff([f'{test_lines[0]}\n'], [f'{true_lines[0]}\n'],
                       tofile='found CSV columns', fromfile='expected CSV columns'))

    diff_lines: list[str] = list()
    column_names: list[str] = true_lines[0].strip().split(',')
    for test_line, true_line in zip(test_lines[1:], true_lines[1:]):
        test_vals: list[float] = [float(val.strip()) for val in test_line.strip().split(',')]
        true_vals: list[float] = [float(val.strip()) for val in true_line.strip().split(',')]
        for test_val, true_val, column_name in zip(test_vals, true_vals, column_names):
            true_zero: bool = abs(true_val) < zero_tol
            test_zero: bool = abs(test_val) < zero_tol
            fail: bool = False
            if true_zero:
                fail = not test_zero
            else:
                fail = not isclose(test_val, true_val, rel_tol=rel_tol)
            if fail:
                diff_lines.append(f'step: {true_line[0]}, column: {column_name}, expected: {true_val}, got: {test_val}')
    return '\n'.join(diff_lines)


def diff_cgns(test_cgns: Path, true_cgns: Path, tolerance: float = 1e-12) -> str:
    """Compare CGNS results against an expected CGSN file with tolerance

    Args:
        test_cgns (Path): Path to output CGNS file
        true_cgns (Path): Path to expected CGNS file
        tolerance (float, optional): Tolerance for comparing floating-point values

    Returns:
        str: Diff output between result and expected CGNS files
    """
    my_env: dict = os.environ.copy()

    run_args: list[str] = ['cgnsdiff', '-d', '-t', f'{tolerance}', str(test_cgns), str(true_cgns)]
    proc = subprocess.run(' '.join(run_args),
                          shell=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          env=my_env)

    return proc.stderr.decode('utf-8') + proc.stdout.decode('utf-8')


def run_tests(test: str, ceed_backends: list[str], mode: RunMode, nproc: int, suite_spec: SuiteSpec) -> TestSuite:
    """Run all test cases for `test` with each of the provided `ceed_backends`

    Args:
        test (str): Name of test
        ceed_backends (list[str]): List of libCEED backends
        mode (RunMode): Output mode, either `RunMode.TAP` or `RunMode.JUNIT`
        nproc (int): Number of MPI processes to use when running each test case
        suite_spec (SuiteSpec): Object defining required methods for running tests

    Returns:
        TestSuite: JUnit `TestSuite` containing results of all test cases
    """
    source_path: Path = suite_spec.get_source_path(test)
    test_specs: list[TestSpec] = get_test_args(source_path)

    if mode is RunMode.TAP:
        print(f'1..{len(test_specs) * len(ceed_backends)}')

    test_cases: list[TestCase] = []
    my_env: dict = os.environ.copy()
    my_env['CEED_ERROR_HANDLER'] = 'exit'

    index: int = 1
    for spec in test_specs:
        for ceed_resource in ceed_backends:
            run_args: list = [suite_spec.get_run_path(test), *spec.args]

            if '{ceed_resource}' in run_args:
                run_args[run_args.index('{ceed_resource}')] = ceed_resource
            if '{nproc}' in run_args:
                run_args[run_args.index('{nproc}')] = f'{nproc}'
            elif nproc > 1 and source_path.suffix != '.py':
                run_args = ['mpiexec', '-n', f'{nproc}', *run_args]

            # run test
            skip_reason: str = suite_spec.check_pre_skip(test, spec, ceed_resource, nproc)
            if skip_reason:
                test_case: TestCase = TestCase(f'{test}, "{spec.name}", n{nproc}, {ceed_resource}',
                                               elapsed_sec=0,
                                               timestamp=time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime()),
                                               stdout='',
                                               stderr='')
                test_case.add_skipped_info(skip_reason)
            else:
                start: float = time.time()
                proc = subprocess.run(' '.join(str(arg) for arg in run_args),
                                      shell=True,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      env=my_env)

                test_case = TestCase(f'{test}, "{spec.name}", n{nproc}, {ceed_resource}',
                                     classname=source_path.parent,
                                     elapsed_sec=time.time() - start,
                                     timestamp=time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(start)),
                                     stdout=proc.stdout.decode('utf-8'),
                                     stderr=proc.stderr.decode('utf-8'),
                                     allow_multiple_subelements=True)
                ref_csvs: list[Path] = []
                if output_files := [arg for arg in spec.args if 'ascii:' in arg]:
                    ref_csvs = [suite_spec.get_output_path(test, file.split('ascii:')[-1]) for file in output_files]
                ref_cgns: list[Path] = []
                if output_files := [arg for arg in spec.args if 'cgns:' in arg]:
                    ref_cgns = [suite_spec.get_output_path(test, file.split('cgns:')[-1]) for file in output_files]
                ref_stdout: Path = suite_spec.get_output_path(test, test + '.out')
                suite_spec.post_test_hook(test, spec)

            # check allowed failures
            if not test_case.is_skipped() and test_case.stderr:
                skip_reason: str = suite_spec.check_post_skip(test, spec, ceed_resource, test_case.stderr)
                if skip_reason:
                    test_case.add_skipped_info(skip_reason)

            # check required failures
            if not test_case.is_skipped():
                required_message, did_fail = suite_spec.check_required_failure(test, spec, ceed_resource, test_case.stderr)
                if required_message and did_fail:
                    test_case.status = f'fails with required: {required_message}'
                elif required_message:
                    test_case.add_failure_info(f'required failure missing: {required_message}')

            # classify other results
            if not test_case.is_skipped() and not test_case.status:
                if test_case.stderr:
                    test_case.add_failure_info('stderr', test_case.stderr)
                if proc.returncode != 0:
                    test_case.add_error_info(f'returncode = {proc.returncode}')
                if ref_stdout.is_file():
                    diff = list(difflib.unified_diff(ref_stdout.read_text().splitlines(keepends=True),
                                                     test_case.stdout.splitlines(keepends=True),
                                                     fromfile=str(ref_stdout),
                                                     tofile='New'))
                    if diff:
                        test_case.add_failure_info('stdout', output=''.join(diff))
                elif test_case.stdout and not suite_spec.check_allowed_stdout(test):
                    test_case.add_failure_info('stdout', output=test_case.stdout)
                # expected CSV output
                for ref_csv in ref_csvs:
                    if not ref_csv.is_file():
                        test_case.add_failure_info('csv', output=f'{ref_csv} not found')
                    else:
                        diff: str = diff_csv(Path.cwd() / ref_csv.name, ref_csv)
                        if diff:
                            test_case.add_failure_info('csv', output=diff)
                        else:
                            (Path.cwd() / ref_csv.name).unlink()
                # expected CGNS output
                for ref_cgn in ref_cgns:
                    if not ref_cgn.is_file():
                        test_case.add_failure_info('cgns', output=f'{ref_cgn} not found')
                    else:
                        diff = diff_cgns(Path.cwd() / ref_cgn.name, ref_cgn)
                        if diff:
                            test_case.add_failure_info('cgns', output=diff)
                        else:
                            (Path.cwd() / ref_cgn.name).unlink()

            # store result
            test_case.args = ' '.join(str(arg) for arg in run_args)
            test_cases.append(test_case)

            print_test_case(test_case, spec, mode, index)
            index += 1
    return TestSuite(test, test_cases)


def print_test_case(test_case: TestCase, spec: TestSpec, mode: RunMode, index: int) -> None:
    if mode is RunMode.TAP:
        # print incremental output if TAP mode
        print(f'# Test: {spec.name}')
        if spec.only:
            print('# Only: {}'.format(','.join(spec.only)))
        if hasattr(test_case, 'args'):
            print(f'# $ {test_case.args}')
        if test_case.is_skipped():
            print('ok {} - SKIP: {}'.format(index, (test_case.skipped[0]['message'] or 'NO MESSAGE').strip()))
        elif test_case.is_failure() or test_case.is_error():
            print(f'not ok {index}')
            if test_case.is_error():
                print(f'  ERROR: {test_case.errors[0]["message"]}')
            if test_case.is_failure():
                for i, failure in enumerate(test_case.failures):
                    print(f'  FAILURE {i}: {failure["message"]}')
                    print(f'    Output: \n{failure["output"]}')
        else:
            print(f'ok {index} - PASS')
        sys.stdout.flush()
    else:
        # print error or failure information if JUNIT mode
        if test_case.is_error() or test_case.is_failure():
            print(f'Test: {test} {spec.name}')
            print(f'  $ {test_case.args}')
            if test_case.is_error():
                print('ERROR: {}'.format((test_case.errors[0]['message'] or 'NO MESSAGE').strip()))
                print('Output: \n{}'.format((test_case.errors[0]['output'] or 'NO MESSAGE').strip()))
            if test_case.is_failure():
                for failure in test_case.failures:
                    print('FAIL: {}'.format((failure['message'] or 'NO MESSAGE').strip()))
                    print('Output: \n{}'.format((failure['output'] or 'NO MESSAGE').strip()))
        sys.stdout.flush()


def write_junit_xml(test_suite: TestSuite, output_file: Optional[Path], batch: str = '') -> None:
    """Write a JUnit XML file containing the results of a `TestSuite`

    Args:
        test_suite (TestSuite): JUnit `TestSuite` to write
        output_file (Optional[Path]): Path to output file, or `None` to generate automatically as `build/{test_suite.name}{batch}.junit`
        batch (str): Name of JUnit batch, defaults to empty string
    """
    output_file: Path = output_file or Path('build') / (f'{test_suite.name}{batch}.junit')
    output_file.write_text(to_xml_report_string([test_suite]))


def has_failures(test_suite: TestSuite) -> bool:
    """Check whether any test cases in a `TestSuite` failed

    Args:
        test_suite (TestSuite): JUnit `TestSuite` to check

    Returns:
        bool: True if any test cases failed
    """
    return any(c.is_failure() or c.is_error() for c in test_suite.test_cases)
