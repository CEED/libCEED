// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed
//
/// @file
/// Enums for the values in differential filtering fields

enum DifferentialFilterStateComponent {
  DIFF_FILTER_PRESSURE,
  DIFF_FILTER_VELOCITY_X,
  DIFF_FILTER_VELOCITY_Y,
  DIFF_FILTER_VELOCITY_Z,
  DIFF_FILTER_TEMPERATURE,
  DIFF_FILTER_STATE_NUM,
};

enum DifferentialFilterVelocitySquared {
  DIFF_FILTER_VELOCITY_SQUARED_XX,
  DIFF_FILTER_VELOCITY_SQUARED_YY,
  DIFF_FILTER_VELOCITY_SQUARED_ZZ,
  DIFF_FILTER_VELOCITY_SQUARED_YZ,
  DIFF_FILTER_VELOCITY_SQUARED_XZ,
  DIFF_FILTER_VELOCITY_SQUARED_XY,
  DIFF_FILTER_VELOCITY_SQUARED_NUM,
};
