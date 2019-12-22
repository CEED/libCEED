import re
import pandas as pd
from pathlib import Path

# Directory location of log files
runlogDir = Path("./runlogs/")

# Regex to parse STDOUT of the nsplex run
logreg = re.compile(
    r"(?:^Global FEM nodes: (\d{2,})).*(?:^Time taken for solution: (\d+\.?\d+)).*(?:^Max Error: (\d+\.?\d+))",
    re.S | re.M,
)


def parseFile(file, filenameRegexStr):
    """Returns dictionary of parsed logfile contents.

    Parameters
    ----------
    file : Path-like object
        Path to the file to be parsed.

    Returns
    -------
    dict
        Values of "dofs",  "time", "error", "degree", and "meshres"'
    """

    values = {}
    filenameMatch = re.match(filenameRegexStr, file.as_posix())
    values["degree"], values["meshres"] = filenameMatch[1], filenameMatch[2]

    with file.open() as filer:
        filestring = filer.read()
    match = logreg.match(filestring)
    values["dofs"] = match[1]
    values["time"] = match[2]
    values["error"] = match[3]

    return values


if __name__ == "__main__":
    results = pd.DataFrame(columns=["degree", "meshres", "dofs", "time", "error"])
    filenameRegexStr = r".*(\d)_(\d+).log"
    for file in runlogDir.glob("*.log"):
        values = parseFile(file)
        results = results.append(values, ignore_index=True)

        # Convert string values to numeric type
    results = results.apply(lambda col: pd.to_numeric(col, errors="coerce"))
