// Copyright (c) 2017-2024, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#![doc = include_str!("../README.md")]

/**
Bindings generated from libCEED's C headers using bindgen.

See `build.rs` to customize the process and refer to the [libCEED API
docs](https://libceed.org/en/latest/api/) for usage.
*/
pub mod bind_ceed {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(dead_code)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
