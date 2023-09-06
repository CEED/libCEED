#!/usr/bin/env python3
# autopep8: off
#TESTARGS(name="SmartSim Regression Test") -c {ceed_resource}
# autopep8: on
# smartsim and smartredis imports
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


logging.disable(logging.WARNING)

fluids_example_dir = Path(__file__).parent.absolute()


class NoError(Exception):
    pass


def setup(directory_path: Path):
    """To create the test directory, then delete it if any exception is raised"""
    if directory_path.exists() and directory_path.is_dir():
        shutil.rmtree(directory_path)
    directory_path.mkdir()
    os.chdir(directory_path)

    PORT = 6780
    global exp
    global db
    exp = Experiment("test", launcher="local")
    db = exp.create_database(port=PORT, batch=False, interface="lo")
    exp.generate(db)
    exp.start(db)

    # SmartRedis will complain if these aren't set
    os.environ['SR_LOG_FILE'] = 'R'
    os.environ['SR_LOG_LEVEL'] = 'INFO'


def test(ceed_resource) -> Tuple[bool, Exception]:
    client = None
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

        client_exp = exp.create_model(f"client_{ceed_resource.replace('/', '_')}", run_settings)

        # Start the client model
        exp.start(client_exp, summary=False, block=True)

        client = Client(cluster=False)

        assert (client.poll_tensor("sizeInfo", 0, 1))
        assert (np.all(client.get_tensor("sizeInfo") == np.array([5002, 12, 6, 1, 1, 0])))

        assert (client.poll_tensor("check-run", 0, 1))
        assert (client.get_tensor("check-run")[0] == 1)

        assert (client.poll_tensor("tensor-ow", 0, 1))
        assert (client.get_tensor("tensor-ow")[0] == 1)

        assert (client.poll_tensor("step", 0, 1))
        assert (client.get_tensor("step")[0] == 2)

        assert (client.poll_tensor("y.0", 0, 1))
        test_data_path = fluids_example_dir / "tests-output/y0_output.npy"
        assert (test_data_path.is_file())

        y0_correct_value = np.load(test_data_path)
        y0_database_value = client.get_tensor("y.0")
        if not np.allclose(y0_database_value, y0_correct_value, rtol=1e-10):
            database_output_path = Path(f"./y0_database_values_{ceed_resource.replace('/', '_')}.npy")
            np.save(database_output_path, y0_database_value)
            print(f"Array values in database max difference: {np.max(np.abs(y0_correct_value - y0_database_value))}\n"
                  f"Array saved to {database_output_path.as_posix()}")

        client.flush_db([os.environ["SSDB"]])
        output = (True, NoError())
    except Exception as e:
        output = (False, e)

    finally:
        if client:
            client.flush_db([os.environ["SSDB"]])

    return output


def teardown(return_path: Path):
    os.chdir(return_path)
    exp.stop(db)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Testing script for SmartSim integration')
    parser.add_argument('-c', '--ceed-backends', type=str, nargs='*', default=['/cpu/self'], help='libCEED backend to use with convergence tests')
    parser.add_argument(
        '-e',
        '--executable',
        type=Path,
        default=fluids_example_dir.parents[3] /
        'build/fluids-navierstokes',
        help='Path to naverstokes executable')
    args = parser.parse_args()

    test_dir = fluids_example_dir / "test_dir"
    print("Setting up database...", end='')
    setup(test_dir)
    print(" Done!")
    for ceed_resource in args.ceed_backends:
        # print("working on " + ceed_resource + ' ...', end='')
        passTest, exception = test(ceed_resource)

        if passTest:
            print("Passed!")
        else:
            print("Failed!", file=sys.stderr)
            print('\t' + ''.join(traceback.TracebackException.from_exception(exception).format()), file=sys.stderr)

    print("Cleaning up database...", end='')
    teardown(Path(__file__).parent)
    print(" Done!")
