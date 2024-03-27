#!/usr/bin/env python3

# Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
# All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-2-Clause
#
# This file is part of CEED:  http://github.com/ceed

import pandas as pd
from postprocess_base import read_logs

# Load the data
runs = read_logs()

# Data output
print('Writing data to \'benchmark_data.csv\'...')
runs.to_csv('benchmark_data.csv', sep='\t', index=False)
print('Writing complete')
