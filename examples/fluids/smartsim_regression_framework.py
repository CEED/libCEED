#!/usr/bin/env python3
from junit_xml import TestCase
from smartsim import Experiment
from smartsim.settings import RunSettings
from smartredis import Client
import numpy as np
from pathlib import Path
import argparse
import traceback
import sys
import time
from typing import Tuple
import os
import shutil
import logging

# autopep8 off
sys.path.insert(0, (Path(__file__).parents[3] / "tests/junit-xml").as_posix())
# autopep8 on

logging.disable(logging.WARNING)

fluids_example_dir = Path(__file__).parent.absolute()


<<<<<<< HEAD
=======
def getOpenSocket():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    addr = s.getsockname()
    s.close()
    return addr[1]


>>>>>>> main
class NoError(Exception):
    pass


class SmartSimTest(object):

    def __init__(self, directory_path: Path):
        self.exp: Experiment
        self.database = None
        self.directory_path: Path = directory_path
        self.original_path: Path

    def setup(self):
        """To create the test directory and start SmartRedis database"""
        self.original_path = Path(os.getcwd())

        if self.directory_path.exists() and self.directory_path.is_dir():
            shutil.rmtree(self.directory_path)
        self.directory_path.mkdir()
        os.chdir(self.directory_path)

<<<<<<< HEAD
        PORT = 6780
=======
        PORT = getOpenSocket()
>>>>>>> main
        self.exp = Experiment("test", launcher="local")
        self.database = self.exp.create_database(port=PORT, batch=False, interface="lo")
        self.exp.generate(self.database)
        self.exp.start(self.database)

        # SmartRedis will complain if these aren't set
        os.environ['SR_LOG_FILE'] = 'R'
        os.environ['SR_LOG_LEVEL'] = 'INFO'

    def test(self, ceed_resource) -> Tuple[bool, Exception, str]:
        client = None
        arguments = []
        try:
            exe_path = "../../build/fluids-navierstokes"
            arguments = [
                '-ceed', ceed_resource,
                '-options_file', (fluids_example_dir / 'blasius.yaml').as_posix(),
                '-ts_max_steps', '2',
                '-diff_filter_grid_based_width',
                '-diff_filter_width_scaling', '1,0.7,1',
                '-ts_monitor', '-snes_monitor',
                '-diff_filter_ksp_max_it', '50', '-diff_filter_ksp_monitor',
                '-degree', '1',
                '-sgs_train_enable',
                '-sgs_train_put_tensor_interval', '2',
            ]

            run_settings = RunSettings(exe_path, exe_args=arguments)

            client_exp = self.exp.create_model(f"client_{ceed_resource.replace('/', '_')}", run_settings)

            # Start the client model
            self.exp.start(client_exp, summary=False, block=True)

            client = Client(cluster=False)

            assert client.poll_tensor("sizeInfo", 250, 5)
            assert np.all(client.get_tensor("sizeInfo") == np.array([5002, 12, 6, 1, 1, 0]))

            assert client.poll_tensor("check-run", 250, 5)
            assert client.get_tensor("check-run")[0] == 1

            assert client.poll_tensor("tensor-ow", 250, 5)
            assert client.get_tensor("tensor-ow")[0] == 1

            assert client.poll_tensor("step", 250, 5)
            assert client.get_tensor("step")[0] == 2

            assert client.poll_tensor("y.0", 250, 5)
            test_data_path = fluids_example_dir / "tests-output/y0_output.npy"
            assert test_data_path.is_file()

            y0_correct_value = np.load(test_data_path)
            y0_database_value = client.get_tensor("y.0")
            rtol = 1e-8
            atol = 1e-8
            if not np.allclose(y0_database_value, y0_correct_value, atol=atol, rtol=rtol):
                # Check whether the S-frame-oriented vorticity vector's second component is just flipped.
                # This can happen due to the eigenvector ordering changing based on whichever one is closest to the vorticity vector.
                # If two eigenvectors are very close to the vorticity vector, this can cause the ordering to flip.
                # This flipping of the vorticity vector is not incorrect, just a known sensitivity of the model.

                total_tolerances = atol + rtol * np.abs(y0_correct_value)  # mimic np.allclose tolerance calculation
                idx_notclose = np.where(np.abs(y0_database_value - y0_correct_value) > total_tolerances)
                if not np.all(idx_notclose[1] == 4):
                    # values other than vorticity are not close
                    test_fail = True
                else:
                    database_vorticity = y0_database_value[idx_notclose]
                    correct_vorticity = y0_correct_value[idx_notclose]
                    test_fail = False if np.allclose(-database_vorticity, correct_vorticity,
                                                     atol=atol, rtol=rtol) else True

                if test_fail:
                    database_output_path = Path(
                        f"./y0_database_values_{ceed_resource.replace('/', '_')}.npy").absolute()
                    np.save(database_output_path, y0_database_value)
                    raise AssertionError(f"Array values in database max difference: {np.max(np.abs(y0_correct_value - y0_database_value))}\n"
                                         f"Array saved to {database_output_path.as_posix()}")

            client.flush_db([os.environ["SSDB"]])
            output = (True, NoError(), exe_path + ' ' + ' '.join(arguments))
        except Exception as e:
            output = (False, e, exe_path + ' ' + ' '.join(arguments))

        finally:
            if client:
                client.flush_db([os.environ["SSDB"]])

        return output

    def test_junit(self, ceed_resource):
        start: float = time.time()

        passTest, exception, args = self.test(ceed_resource)

        output = "" if isinstance(exception, NoError) else ''.join(
            traceback.TracebackException.from_exception(exception).format())

        test_case = TestCase(f'SmartSim Test {ceed_resource}',
                             elapsed_sec=time.time() - start,
                             timestamp=time.strftime(
                                 '%Y-%m-%d %H:%M:%S %Z', time.localtime(start)),
                             stdout=output,
                             stderr=output,
                             allow_multiple_subelements=True,
                             category=f'SmartSim Tests')
        test_case.args = args
        if not passTest and 'occa' in ceed_resource:
            test_case.add_skipped_info("OCCA mode not supported")
        elif not passTest:
            test_case.add_failure_info("exception", output)

        return test_case

    def teardown(self):
        self.exp.stop(self.database)
        os.chdir(self.original_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Testing script for SmartSim integration')
    parser.add_argument(
        '-c',
        '--ceed-backends',
        type=str,
        nargs='*',
        default=['/cpu/self'],
        help='libCEED backend to use with convergence tests')
    args = parser.parse_args()

    test_dir = fluids_example_dir / "test_dir"
    print("Setting up database...", end='')
    test_framework = SmartSimTest(test_dir)
    test_framework.setup()
    print(" Done!")
    for ceed_resource in args.ceed_backends:
        print("working on " + ceed_resource + ' ...', end='')
        passTest, exception, _ = test_framework.test(ceed_resource)

        if passTest:
            print("Passed!")
        else:
            print("Failed!", file=sys.stderr)
            print('\t' + ''.join(traceback.TracebackException.from_exception(exception).format()), file=sys.stderr)

    print("Cleaning up database...", end='')
    test_framework.teardown()
    print(" Done!")
