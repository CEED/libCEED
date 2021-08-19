import inspect
import os


def output(capsys, suffix='.out'):
    stdout, stderr = capsys.readouterr()

    caller = inspect.stack()[1]
    caller_dirname = os.path.dirname(caller.filename)
    output_file = os.path.join(
        caller_dirname,
        'output',
        caller.function +
        suffix)
    with open(output_file) as output_file:
        ref_stdout = output_file.read()

    return stdout, stderr, ref_stdout
