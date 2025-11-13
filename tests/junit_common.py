from abc import ABC, abstractmethod
from collections.abc import Iterable
import argparse
import csv
from dataclasses import dataclass, field, fields
import difflib
from enum import Enum
from math import isclose
import os
from pathlib import Path
import re
import subprocess
import multiprocessing as mp
import sys
import time
from typing import Optional, Tuple, List, Dict, Callable, Iterable, get_origin
import shutil

sys.path.insert(0, str(Path(__file__).parent / "junit-xml"))
from junit_xml import TestCase, TestSuite, to_xml_report_string  # nopep8


class ParseError(RuntimeError):
    """A custom exception for failed parsing."""

    def __init__(self, message):
        super().__init__(message)


class CaseInsensitiveEnumAction(argparse.Action):
    """Action to convert input values to lower case prior to converting to an Enum type"""

    def __init__(self, option_strings, dest, type, default, **kwargs):
        if not issubclass(type, Enum):
            raise ValueError(f"{type} must be an Enum")
        # store provided enum type
        self.enum_type = type
        if isinstance(default, self.enum_type):
            pass
        elif isinstance(default, str):
            default = self.enum_type(default.lower())
        elif isinstance(default, Iterable):
            default = [self.enum_type(v.lower()) for v in default]
        else:
            raise argparse.ArgumentTypeError("Invalid value type, must be str or iterable")
        # prevent automatic type conversion
        super().__init__(option_strings, dest, default=default, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, self.enum_type):
            pass
        elif isinstance(values, str):
            values = self.enum_type(values.lower())
        elif isinstance(values, Iterable):
            values = [self.enum_type(v.lower()) for v in values]
        else:
            raise argparse.ArgumentTypeError("Invalid value type, must be str or iterable")
        setattr(namespace, self.dest, values)


@dataclass
class TestSpec:
    """Dataclass storing information about a single test case"""
    name: str = field(default_factory=str)
    csv_rtol: float = -1
    csv_ztol: float = -1
    cgns_tol: float = -1
    only: List = field(default_factory=list)
    args: List = field(default_factory=list)
    key_values: Dict = field(default_factory=dict)


class RunMode(Enum):
    """Enumeration of run modes, either `RunMode.TAP` or `RunMode.JUNIT`"""
    TAP = 'tap'
    JUNIT = 'junit'

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


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

    @property
    def test_failure_artifacts_path(self) -> Path:
        """Path to test failure artifacts"""
        return Path('build') / 'test_failure_artifacts'

    @property
    def cgns_tol(self):
        """Absolute tolerance for CGNS diff"""
        return getattr(self, '_cgns_tol', 1.0e-12)

    @cgns_tol.setter
    def cgns_tol(self, val):
        self._cgns_tol = val

    @property
    def csv_ztol(self):
        """Keyword arguments to be passed to diff_csv()"""
        return getattr(self, '_csv_ztol', 3e-10)

    @csv_ztol.setter
    def csv_ztol(self, val):
        self._csv_ztol = val

    @property
    def csv_rtol(self):
        """Keyword arguments to be passed to diff_csv()"""
        return getattr(self, '_csv_rtol', 1e-6)

    @csv_rtol.setter
    def csv_rtol(self, val):
        self._csv_rtol = val

    @property
    def csv_comment_diff_fn(self):  # -> Any | Callable[..., None]:
        return getattr(self, '_csv_comment_diff_fn', None)

    @csv_comment_diff_fn.setter
    def csv_comment_diff_fn(self, test_fn):
        self._csv_comment_diff_fn = test_fn

    @property
    def csv_comment_str(self):
        return getattr(self, '_csv_comment_str', '#')

    @csv_comment_str.setter
    def csv_comment_str(self, comment_str):
        self._csv_comment_str = comment_str

    def post_test_hook(self, test: str, spec: TestSpec, backend: str) -> None:
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
            Optional[str]: Skip reason, or `None` if unexpected error
        """
        return None

    def check_required_failure(self, test: str, spec: TestSpec, resource: str, stderr: str) -> Tuple[str, bool]:
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


def contains_any(base: str, substrings: List[str]) -> bool:
    """Helper function, checks if any of the substrings are included in the base string

    Args:
        base (str): Base string to search in
        substrings (List[str]): List of potential substrings

    Returns:
        bool: True if any substrings are included in base string
    """
    return any((sub in base for sub in substrings))


def startswith_any(base: str, prefixes: List[str]) -> bool:
    """Helper function, checks if the base string is prefixed by any of `prefixes`

    Args:
        base (str): Base string to search
        prefixes (List[str]): List of potential prefixes

    Returns:
        bool: True if base string is prefixed by any of the prefixes
    """
    return any((base.startswith(prefix) for prefix in prefixes))


def find_matching(line: str, open: str = '(', close: str = ')') -> Tuple[int, int]:
    """Find the start and end positions of the first outer paired delimeters

    Args:
        line (str): Line to search
        open (str, optional): Opening delimiter, must be different than `close`. Defaults to '('.
        close (str, optional): Closing delimeter, must be different than `open`. Defaults to ')'.

    Raises:
        RuntimeError: If open or close is not a single character
        RuntimeError: If open and close are the same characters

    Returns:
        Tuple[int]: If matching delimeters are found, return indices in `list`. Otherwise, return end < start.
    """
    if len(open) != 1 or len(close) != 1:
        raise RuntimeError("`open` and `close` must be single characters")
    if open == close:
        raise RuntimeError("`open` and `close` must be different characters")
    start: int = line.find(open)
    if start < 0:
        return -1, -1
    count: int = 1
    for i in range(start + 1, len(line)):
        if line[i] == open:
            count += 1
        if line[i] == close:
            count -= 1
            if count == 0:
                return start, i
    return start, -1


def parse_test_line(line: str, fallback_name: str = '') -> TestSpec:
    """Parse a single line of TESTARGS and CLI arguments into a `TestSpec` object

    Args:
        line (str): String containing TESTARGS specification and CLI arguments

    Returns:
        TestSpec: Parsed specification of test case
    """
    test_fields = fields(TestSpec)
    field_names = [f.name for f in test_fields]
    known: Dict = dict()
    other: Dict = dict()
    if line[0] == "(":
        # have key/value pairs to parse
        start, end = find_matching(line)
        if end < start:
            raise ParseError(f"Mismatched parentheses in TESTCASE: {line}")

        keyvalues_str = line[start:end + 1]
        keyvalues_pattern = re.compile(r'''
            (?:\(\s*|\s*,\s*)   # start with open parentheses or comma, no capture
            ([A-Za-z]+[\w\-]+)  # match key starting with alpha, containing alphanumeric, _, or -; captured as Group 1
            \s*=\s*             # key is followed by = (whitespace ignored)
            (?:                 # uncaptured group for OR
              "((?:[^"]|\\")+)" #   match quoted value (any internal " must be escaped as \"); captured as Group 2
            | ([^=]+)           #   OR match unquoted value (no equals signs allowed); captured as Group 3
            )                   # end uncaptured group for OR
            \s*(?=,|\))         # lookahead for either next comma or closing parentheses
        ''', re.VERBOSE)

        for match in re.finditer(keyvalues_pattern, keyvalues_str):
            if not match:  # empty
                continue
            key = match.group(1)
            value = match.group(2) if match.group(2) else match.group(3)
            try:
                index = field_names.index(key)
                if key == "only":  # weird bc only is a list
                    value = [constraint.strip() for constraint in value.split(',')]
                try:
                    # TODO: stop supporting python <=3.8
                    known[key] = test_fields[index].type(value)  # type: ignore
                except TypeError:
                    # TODO: this is still liable to fail for complex types
                    known[key] = get_origin(test_fields[index].type)(value)  # type: ignore
            except ValueError:
                other[key] = value

        line = line[end + 1:]

    if not 'name' in known.keys():
        known['name'] = fallback_name

    args_pattern = re.compile(r'''
        \s+(            # remove leading space
            (?:"[^"]+") # match quoted CLI option
          | (?:[\S]+)   # match anything else that is space separated
        )
    ''', re.VERBOSE)
    args: List[str] = re.findall(args_pattern, line)
    for k, v in other.items():
        print(f"warning, unknown TESTCASE option for test '{known['name']}': {k}={v}")
    return TestSpec(**known, key_values=other, args=args)


def get_test_args(source_file: Path) -> List[TestSpec]:
    """Parse all test cases from a given source file

    Args:
        source_file (Path): Path to source file

    Raises:
        RuntimeError: Errors if source file extension is unsupported

    Returns:
        List[TestSpec]: List of parsed `TestSpec` objects, or a list containing a single, default `TestSpec` if none were found
    """
    comment_str: str = ''
    if source_file.suffix in ['.c', '.cc', '.cpp']:
        comment_str = '//'
    elif source_file.suffix in ['.py']:
        comment_str = '#'
    elif source_file.suffix in ['.usr']:
        comment_str = 'C_'
    elif source_file.suffix in ['.f90']:
        comment_str = '! '
    else:
        raise RuntimeError(f'Unrecognized extension for file: {source_file}')

    return [parse_test_line(line.strip(comment_str).removeprefix("TESTARGS"), source_file.stem)
            for line in source_file.read_text().splitlines()
            if line.startswith(f'{comment_str}TESTARGS')] or [TestSpec(source_file.stem, args=['{ceed_resource}'])]


def diff_csv(test_csv: Path, true_csv: Path, zero_tol: float, rel_tol: float,
             comment_str: str = '#', comment_func: Optional[Callable[[str, str], Optional[str]]] = None) -> str:
    """Compare CSV results against an expected CSV file with tolerances

    Args:
        test_csv (Path): Path to output CSV results
        true_csv (Path): Path to expected CSV results
        zero_tol (float): Tolerance below which values are considered to be zero.
        rel_tol (float): Relative tolerance for comparing non-zero values.
        comment_str (str, optional): String to denoting commented line
        comment_func (Callable, optional): Function to determine if test and true line are different

    Returns:
        str: Diff output between result and expected CSVs
    """
    test_lines: List[str] = test_csv.read_text().splitlines()
    true_lines: List[str] = true_csv.read_text().splitlines()
    # Files should not be empty
    if len(test_lines) == 0:
        return f'No lines found in test output {test_csv}'
    if len(true_lines) == 0:
        return f'No lines found in test source {true_csv}'
    if len(test_lines) != len(true_lines):
        return f'Number of lines in {test_csv} and {true_csv} do not match'

    # Process commented lines
    uncommented_lines: List[int] = []
    for n, (test_line, true_line) in enumerate(zip(test_lines, true_lines)):
        if test_line[0] == comment_str and true_line[0] == comment_str:
            if comment_func:
                output = comment_func(test_line, true_line)
                if output:
                    return output
        elif test_line[0] == comment_str and true_line[0] != comment_str:
            return f'Commented line found in {test_csv} at line {n} but not in {true_csv}'
        elif test_line[0] != comment_str and true_line[0] == comment_str:
            return f'Commented line found in {true_csv} at line {n} but not in {test_csv}'
        else:
            uncommented_lines.append(n)

    # Remove commented lines
    test_lines = [test_lines[line] for line in uncommented_lines]
    true_lines = [true_lines[line] for line in uncommented_lines]

    test_reader: csv.DictReader = csv.DictReader(test_lines)
    true_reader: csv.DictReader = csv.DictReader(true_lines)
    if not test_reader.fieldnames:
        return f'No CSV columns found in test output {test_csv}'
    if not true_reader.fieldnames:
        return f'No CSV columns found in test source {true_csv}'
    if test_reader.fieldnames != true_reader.fieldnames:
        return ''.join(difflib.unified_diff([f'{test_lines[0]}\n'], [f'{true_lines[0]}\n'],
                       tofile='found CSV columns', fromfile='expected CSV columns'))

    diff_lines: List[str] = list()
    for test_line, true_line in zip(test_reader, true_reader):
        for key in test_reader.fieldnames:
            # Check if the value is numeric
            try:
                true_val: float = float(true_line[key])
                test_val: float = float(test_line[key])
                true_zero: bool = abs(true_val) < zero_tol
                test_zero: bool = abs(test_val) < zero_tol
                fail: bool = False
                if true_zero:
                    fail = not test_zero
                else:
                    fail = not isclose(test_val, true_val, rel_tol=rel_tol)
                if fail:
                    diff_lines.append(f'column: {key}, expected: {true_val}, got: {test_val}')
            except ValueError:
                if test_line[key] != true_line[key]:
                    diff_lines.append(f'column: {key}, expected: {true_line[key]}, got: {test_line[key]}')

    return '\n'.join(diff_lines)


def diff_cgns(test_cgns: Path, true_cgns: Path, cgns_tol: float) -> str:
    """Compare CGNS results against an expected CGSN file with tolerance

    Args:
        test_cgns (Path): Path to output CGNS file
        true_cgns (Path): Path to expected CGNS file
        cgns_tol (float): Tolerance for comparing floating-point values

    Returns:
        str: Diff output between result and expected CGNS files
    """
    my_env: dict = os.environ.copy()

    run_args: List[str] = ['cgnsdiff', '-d', '-t', f'{cgns_tol}', str(test_cgns), str(true_cgns)]
    proc = subprocess.run(' '.join(run_args),
                          shell=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          env=my_env)

    return proc.stderr.decode('utf-8') + proc.stdout.decode('utf-8')


def diff_ascii(test_file: Path, true_file: Path, backend: str) -> str:
    """Compare ASCII results against an expected ASCII file

    Args:
        test_file (Path): Path to output ASCII file
        true_file (Path): Path to expected ASCII file

    Returns:
        str: Diff output between result and expected ASCII files
    """
    tmp_backend: str = backend.replace('/', '-')
    true_str: str = true_file.read_text().replace('{ceed_resource}', tmp_backend)
    diff = list(difflib.unified_diff(test_file.read_text().splitlines(keepends=True),
                                     true_str.splitlines(keepends=True),
                                     fromfile=str(test_file),
                                     tofile=str(true_file)))
    return ''.join(diff)


def test_case_output_string(test_case: TestCase, spec: TestSpec, mode: RunMode,
                            backend: str, test: str, index: int, verbose: bool) -> str:
    output_str = ''
    if mode is RunMode.TAP:
        # print incremental output if TAP mode
        if test_case.is_skipped():
            output_str += f'    ok {index} - {spec.name}, {backend} # SKIP {test_case.skipped[0]["message"]}\n'
        elif test_case.is_failure() or test_case.is_error():
            output_str += f'    not ok {index} - {spec.name}, {backend} ({test_case.elapsed_sec} s)\n'
        else:
            output_str += f'    ok {index} - {spec.name}, {backend} ({test_case.elapsed_sec} s)\n'
        if test_case.is_failure() or test_case.is_error() or verbose:
            output_str += f'      ---\n'
            if spec.only:
                output_str += f'      only: {",".join(spec.only)}\n'
            output_str += f'      args: {test_case.args}\n'
            if spec.csv_ztol > 0:
                output_str += f'      csv_ztol: {spec.csv_ztol}\n'
            if spec.csv_rtol > 0:
                output_str += f'      csv_rtol: {spec.csv_rtol}\n'
            if spec.cgns_tol > 0:
                output_str += f'      cgns_tol: {spec.cgns_tol}\n'
            for k, v in spec.key_values.items():
                output_str += f'      {k}: {v}\n'
            if test_case.is_error():
                output_str += f'      error: {test_case.errors[0]["message"]}\n'
            if test_case.is_failure():
                output_str += f'      failures:\n'
                for i, failure in enumerate(test_case.failures):
                    output_str += f'        -\n'
                    output_str += f'          message: {failure["message"]}\n'
                    if failure["output"]:
                        out = failure["output"].strip().replace('\n', '\n            ')
                        output_str += f'          output: |\n            {out}\n'
            output_str += f'      ...\n'
    else:
        # print error or failure information if JUNIT mode
        if test_case.is_error() or test_case.is_failure():
            output_str += f'Test: {test} {spec.name}\n'
            output_str += f'  $ {test_case.args}\n'
            if test_case.is_error():
                output_str += 'ERROR: {}\n'.format((test_case.errors[0]['message'] or 'NO MESSAGE').strip())
                output_str += 'Output: \n{}\n'.format((test_case.errors[0]['output'] or 'NO MESSAGE').strip())
            if test_case.is_failure():
                for failure in test_case.failures:
                    output_str += 'FAIL: {}\n'.format((failure['message'] or 'NO MESSAGE').strip())
                    output_str += 'Output: \n{}\n'.format((failure['output'] or 'NO MESSAGE').strip())
    return output_str


def save_failure_artifact(suite_spec: SuiteSpec, file: Path) -> Path:
    """Attach a file to a test case

    Args:
        test_case (TestCase): Test case to attach the file to
        file (Path): Path to the file to attach
    """
    save_path: Path = suite_spec.test_failure_artifacts_path / file.name
    shutil.copyfile(file, save_path)
    return save_path


def run_test(index: int, test: str, spec: TestSpec, backend: str,
             mode: RunMode, nproc: int, suite_spec: SuiteSpec, verbose: bool = False) -> TestCase:
    """Run a single test case and backend combination

    Args:
        index (int): Index of backend for current spec
        test (str): Path to test
        spec (TestSpec): Specification of test case
        backend (str): CEED backend
        mode (RunMode): Output mode
        nproc (int): Number of MPI processes to use when running test case
        suite_spec (SuiteSpec): Specification of test suite
        verbose (bool, optional): Print detailed output for all runs, not just failures. Defaults to False.

    Returns:
        TestCase: Test case result
    """
    source_path: Path = suite_spec.get_source_path(test)
    run_args: List = [f'{suite_spec.get_run_path(test)}', *map(str, spec.args)]

    if '{ceed_resource}' in run_args:
        run_args[run_args.index('{ceed_resource}')] = backend
    for i, arg in enumerate(run_args):
        if '{ceed_resource}' in arg:
            run_args[i] = arg.replace('{ceed_resource}', backend.replace('/', '-'))
    if '{nproc}' in run_args:
        run_args[run_args.index('{nproc}')] = f'{nproc}'
    elif nproc > 1 and source_path.suffix != '.py':
        run_args = ['mpiexec', '-n', f'{nproc}', *run_args]

    # run test
    skip_reason: Optional[str] = suite_spec.check_pre_skip(test, spec, backend, nproc)
    if skip_reason:
        test_case: TestCase = TestCase(f'{test}, "{spec.name}", n{nproc}, {backend}',
                                       elapsed_sec=0,
                                       timestamp=time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime()),
                                       stdout='',
                                       stderr='',
                                       category=spec.name,)
        test_case.add_skipped_info(skip_reason)
    else:
        start: float = time.time()
        proc = subprocess.run(' '.join(str(arg) for arg in run_args),
                              shell=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              env=my_env)

        test_case = TestCase(f'{test}, "{spec.name}", n{nproc}, {backend}',
                             classname=source_path.parent,
                             elapsed_sec=time.time() - start,
                             timestamp=time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(start)),
                             stdout=proc.stdout.decode('utf-8'),
                             stderr=proc.stderr.decode('utf-8'),
                             allow_multiple_subelements=True,
                             category=spec.name,)
        ref_csvs: List[Path] = []
        ref_ascii: List[Path] = []
        output_files: List[str] = [arg.split(':')[1] for arg in run_args if arg.startswith('ascii:')]
        if output_files:
            ref_csvs = [suite_spec.get_output_path(test, file)
                        for file in output_files if file.endswith('.csv')]
            ref_ascii = [suite_spec.get_output_path(test, file)
                         for file in output_files if not file.endswith('.csv')]
        ref_cgns: List[Path] = []
        output_files = [arg.split(':')[1] for arg in run_args if arg.startswith('cgns:')]
        if output_files:
            ref_cgns = [suite_spec.get_output_path(test, file) for file in output_files]
        ref_stdout: Path = suite_spec.get_output_path(test, test + '.out')
        suite_spec.post_test_hook(test, spec, backend)

    # check allowed failures
    if not test_case.is_skipped() and test_case.stderr:
        skip_reason: Optional[str] = suite_spec.check_post_skip(test, spec, backend, test_case.stderr)
        if skip_reason:
            test_case.add_skipped_info(skip_reason)

    # check required failures
    if not test_case.is_skipped():
        required_message, did_fail = suite_spec.check_required_failure(
            test, spec, backend, test_case.stderr)
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
            csv_name = ref_csv.name
            out_file = Path.cwd() / csv_name
            if not ref_csv.is_file():
                # remove _{ceed_backend} from path name
                ref_csv = (ref_csv.parent / ref_csv.name.rsplit('_', 1)[0]).with_suffix('.csv')
            if not ref_csv.is_file():
                test_case.add_failure_info('csv', output=f'{ref_csv} not found')
            elif not out_file.is_file():
                test_case.add_failure_info('csv', output=f'{out_file} not found')
            else:
                csv_ztol: float = spec.csv_ztol if spec.csv_ztol > 0 else suite_spec.csv_ztol
                csv_rtol: float = spec.csv_rtol if spec.csv_rtol > 0 else suite_spec.csv_rtol
                diff = diff_csv(
                    out_file,
                    ref_csv,
                    csv_ztol,
                    csv_rtol,
                    suite_spec.csv_comment_str,
                    suite_spec.csv_comment_diff_fn)
                if diff:
                    save_path: Path = suite_spec.test_failure_artifacts_path / csv_name
                    shutil.move(out_file, save_path)
                    test_case.add_failure_info(f'csv: {save_path}', output=diff)
                else:
                    out_file.unlink()
        # expected CGNS output
        for ref_cgn in ref_cgns:
            cgn_name = ref_cgn.name
            out_file = Path.cwd() / cgn_name
            if not ref_cgn.is_file():
                # remove _{ceed_backend} from path name
                ref_cgn = (ref_cgn.parent / ref_cgn.name.rsplit('_', 1)[0]).with_suffix('.cgns')
            if not ref_cgn.is_file():
                test_case.add_failure_info('cgns', output=f'{ref_cgn} not found')
            elif not out_file.is_file():
                test_case.add_failure_info('cgns', output=f'{out_file} not found')
            else:
                cgns_tol = spec.cgns_tol if spec.cgns_tol > 0 else suite_spec.cgns_tol
                diff = diff_cgns(out_file, ref_cgn, cgns_tol=cgns_tol)
                if diff:
                    save_path: Path = suite_spec.test_failure_artifacts_path / cgn_name
                    shutil.move(out_file, save_path)
                    test_case.add_failure_info(f'cgns: {save_path}', output=diff)
                else:
                    out_file.unlink()
        # expected ASCII output
        for ref_file in ref_ascii:
            ref_name = ref_file.name
            out_file = Path.cwd() / ref_name
            if not ref_file.is_file():
                # remove _{ceed_backend} from path name
                ref_file = (ref_file.parent / ref_file.name.rsplit('_', 1)[0]).with_suffix(ref_file.suffix)
            if not ref_file.is_file():
                test_case.add_failure_info('ascii', output=f'{ref_file} not found')
            elif not out_file.is_file():
                test_case.add_failure_info('ascii', output=f'{out_file} not found')
            else:
                diff = diff_ascii(out_file, ref_file, backend)
                if diff:
                    save_path: Path = suite_spec.test_failure_artifacts_path / ref_name
                    shutil.move(out_file, save_path)
                    test_case.add_failure_info(f'ascii: {save_path}', output=diff)
                else:
                    out_file.unlink()

    # store result
    test_case.args = ' '.join(str(arg) for arg in run_args)
    output_str = test_case_output_string(test_case, spec, mode, backend, test, index, verbose)

    return test_case, output_str


def init_process():
    """Initialize multiprocessing process"""
    # set up error handler
    global my_env
    my_env = os.environ.copy()
    my_env['CEED_ERROR_HANDLER'] = 'exit'


def run_tests(test: str, ceed_backends: List[str], mode: RunMode, nproc: int,
              suite_spec: SuiteSpec, pool_size: int = 1, search: str = ".*", verbose: bool = False) -> TestSuite:
    """Run all test cases for `test` with each of the provided `ceed_backends`

    Args:
        test (str): Name of test
        ceed_backends (List[str]): List of libCEED backends
        mode (RunMode): Output mode, either `RunMode.TAP` or `RunMode.JUNIT`
        nproc (int): Number of MPI processes to use when running each test case
        suite_spec (SuiteSpec): Object defining required methods for running tests
        pool_size (int, optional): Number of processes to use when running tests in parallel. Defaults to 1.
        search (str, optional): Regular expression used to match tests. Defaults to ".*".
        verbose (bool, optional): Print detailed output for all runs, not just failures. Defaults to False.

    Returns:
        TestSuite: JUnit `TestSuite` containing results of all test cases
    """
    test_specs: List[TestSpec] = [
        t for t in get_test_args(suite_spec.get_source_path(test)) if re.search(search, t.name, re.IGNORECASE)
    ]
    suite_spec.test_failure_artifacts_path.mkdir(parents=True, exist_ok=True)
    if mode is RunMode.TAP:
        print('TAP version 13')
        print(f'1..{len(test_specs)}')

    with mp.Pool(processes=pool_size, initializer=init_process) as pool:
        async_outputs: List[List[mp.pool.AsyncResult]] = [
            [pool.apply_async(run_test, (i, test, spec, backend, mode, nproc, suite_spec, verbose))
             for (i, backend) in enumerate(ceed_backends, start=1)]
            for spec in test_specs
        ]

        test_cases = []
        for (i, subtest) in enumerate(async_outputs, start=1):
            is_new_subtest = True
            subtest_ok = True
            for async_output in subtest:
                test_case, print_output = async_output.get()
                test_cases.append(test_case)
                if is_new_subtest and mode == RunMode.TAP:
                    is_new_subtest = False
                    print(f'# Subtest: {test_case.category}')
                    print(f'    1..{len(ceed_backends)}')
                print(print_output, end='')
                if test_case.is_failure() or test_case.is_error():
                    subtest_ok = False
            if mode == RunMode.TAP:
                print(f'{"" if subtest_ok else "not "}ok {i} - {test_case.category}')

    return TestSuite(test, test_cases)


def write_junit_xml(test_suite: TestSuite, batch: str = '') -> None:
    """Write a JUnit XML file containing the results of a `TestSuite`

    Args:
        test_suite (TestSuite): JUnit `TestSuite` to write
        batch (str): Name of JUnit batch, defaults to empty string
    """
    output_file = Path('build') / (f'{test_suite.name}{batch}.junit')
    output_file.write_text(to_xml_report_string([test_suite]))


def has_failures(test_suite: TestSuite) -> bool:
    """Check whether any test cases in a `TestSuite` failed

    Args:
        test_suite (TestSuite): JUnit `TestSuite` to check

    Returns:
        bool: True if any test cases failed
    """
    return any(c.is_failure() or c.is_error() for c in test_suite.test_cases)
