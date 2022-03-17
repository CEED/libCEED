# Copyright (c) 2017, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
# All Rights reserved. See files LICENSE and NOTICE for details.
#
# This file is part of CEED, a collection of benchmarks, miniapps, software
# libraries and APIs for efficient high-order finite element and spectral
# element discretizations for exascale applications. For more information and
# source code availability see http://github.com/ceed
#
# The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
# a collaborative effort of two U.S. Department of Energy organizations (Office
# of Science and the National Nuclear Security Administration) responsible for
# the planning and preparation of a capable exascale ecosystem, including
# software, applications, hardware, advanced system engineering and early
# testbed platforms, in support of the nation's exascale computing imperative.

import re
import pandas as pd
from pathlib import Path

# Regex to parse STDOUT of the navierstokes run
logreg = re.compile(
    r".*(?:^Degree of FEM Space: (\d+)).*(?:^Global FEM nodes: (\d{2,})).*(?:^dm_plex_box_faces: (\S+)).*(?:^Time taken for solution: (\d*\.?\d+)).*(?:^Relative Error: (\d*\.?\d+))",
    re.DOTALL | re.MULTILINE,
)


def parseFile(file):
    """Returns dictionary of parsed logfile contents.

    Parameters
    ----------
    file : Path-like object
        Path to the file to be parsed.

    Returns
    -------
    dict
        Values of "dofs",  "time", "error", "degree", and "box_faces"'
    """

    values = {}
    with file.open() as filer:
        filestring = filer.read()
    match = logreg.match(filestring)
    values["degree"] = match[1]
    values["dofs"] = match[2]
    box_faceStr = match[3]
    values["time"] = match[4]
    values["error"] = match[5]

    # Splitting box_face argument str into individual entries
    box_faceList = box_faceStr.split(",")
    for i, box_face in enumerate(box_faceList):
        values["box_face" + str(i)] = box_face

    return values


if __name__ == "__main__":
    # Directory location of log files
    runlogDir = Path("./")

    results = pd.DataFrame()
    for file in runlogDir.glob("*.log"):
        values = parseFile(file)
        results = results.append(values, ignore_index=True)

    # Convert string values to numeric type
    results = results.apply(lambda col: pd.to_numeric(col, errors="coerce"))
