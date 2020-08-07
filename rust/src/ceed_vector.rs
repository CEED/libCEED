#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::mem;
use std::fmt;
use std::ffi::CString;
use std::io::prelude::*;
use std::io::{BufReader, SeekFrom};

use cfile;
// use std::io::{self, Write};

mod rust_ceed {
  include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

/// CeedVector context wrapper
struct CeedVector {
  ceed_reference : ceed::Ceed,
  ceed_vec_ptr : rust_ceed::CeedVector,
}

/// Destructor
impl Drop for CeedVector {
  fn drop(&mut self) {
    unsafe {
      rust_ceed::CeedVectorDestroy(&mut self.ceed_vec_ptr);
    }
  }
}