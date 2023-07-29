#!/usr/bin/env python3
# smartsim and smartredis imports
from smartsim import Experiment
from smartsim.settings import RunSettings
from smartredis import Client
import numpy as np
import os
from pathlib import Path
import shutil
import logging
from contextlib import contextmanager

logging.disable(logging.WARNING)

@contextmanager
def test_directory(directory_path: Path):
    """To create the test directory, then delete it if any exception is raised"""
    directory_path.mkdir()
    os.chdir(directory_path)
    try:
        yield
    finally:
        os.chdir(Path(__file__).parent)
        shutil.rmtree(test_dir)

## SmartRedis will complain if these aren't set
os.environ['SR_LOG_FILE'] = 'stdout'
os.environ['SR_LOG_LEVEL'] = 'INFO'

fluids_example_dir = (Path(__file__).parent / "../").absolute()
test_dir = fluids_example_dir / "test_dir"

with test_directory(test_dir):
    # Set up database and start it
    PORT = 6780
    exp = Experiment("test", launcher="local")
    db = exp.create_database(port=PORT,
                            batch=False,
                            # run_command="mpirun"
                            interface="lo",
                            )
    exp.generate(db)
    exp.start(db)

    # Set the run settings, including the client executable and how to run it
    exe_path = fluids_example_dir / "navierstokes"
    arguments = [
        '-ceed', '/cpu/self/memcheck/serial',
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

    run_settings = RunSettings(
                exe_path,
                exe_args=arguments,
                )

    client_exp = exp.create_model("client", run_settings)

    # Start the client model
    exp.start(client_exp, summary=False, block=True)

    # Add some python code here that connects to the database and check tensors in it
    client = Client(cluster=False)

    assert(client.poll_tensor("sizeInfo", 0, 1))
    assert(np.all(client.get_tensor("sizeInfo") == np.array([5002, 12, 6, 1, 1, 0])))

    assert(client.poll_tensor("check-run", 0, 1))
    assert(client.get_tensor("check-run")[0] == 1)

    assert(client.poll_tensor("tensor-ow", 0, 1))
    assert(client.get_tensor("tensor-ow")[0] == 1)

    assert(client.poll_tensor("step", 0, 1))
    assert(client.get_tensor("step")[0] == 2)

    assert(client.poll_tensor("y.0", 0, 1))
    test_data_path = fluids_example_dir / "tests-output/y0_output.npy"
    assert(test_data_path.is_file())

    y0_correct_value = np.load(test_data_path)
    assert(np.allclose(client.get_tensor("y.0"), y0_correct_value))

    # Stop database
    exp.stop(db)
