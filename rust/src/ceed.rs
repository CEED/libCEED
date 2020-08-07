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
mod ceed_vector;

/// A Ceed context wrapper
pub struct Ceed {
  // Pointer to C object
  backend : String,
  ceed_ptr : rust_ceed::Ceed,
}

/// Returns a Ceed context initalized with the specified resource
///
/// # arguments
///
/// * 'resource' - Resource to use, e.g., "/cpu/self"
/// 
/// 
/// let ceed = init_ceed("/cpu/self/ref/serial");
/// 
pub fn init_ceed(resource: &str) -> Ceed {
  // Convert to C string
  let c_resource = CString::new(resource).expect("CString::new failed");
  
  // Call to libCEED
  unsafe {
    let mut ceed_ptr: rust_ceed::Ceed = libc::malloc(mem::size_of::<rust_ceed::Ceed>()) as rust_ceed::Ceed;
    rust_ceed::CeedInit(c_resource.as_ptr() as *const i8, &mut ceed_ptr);
    Ceed { backend: resource.to_string(), ceed_ptr: ceed_ptr }
  }
}

/// Display
impl fmt::Display for Ceed {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "{}", self.backend)


    //let tmpout = cfile::tmpfile().unwrap();
    //rust_ceed::CeedView(self.ceed_ptr, libc::stdout);
    //tmpout.seek(SeekFrom::Start(0)).unwrap();
    //let mut r = BufReader::new(tmpout);
    //let mut s = String::new();
    // r.read(&mut s).unwrap();
    //write!(f, "{}", s)
  }
}

/// Destructor
impl Drop for Ceed {
  fn drop(&mut self) {
    unsafe {
      rust_ceed::CeedDestroy(&mut self.ceed_ptr);
    }
  }
}


// Object constructors
impl Ceed {
  pub fn vector_create(&self, n: i32) -> ceed_vector::CeedVector {
    let mut vec_ptr = libc::malloc(mem::size_of::<rust_ceed::CeedVector>()) as rust_ceed::CeedVector;
    rust_ceed::CeedVectorCreate(self.ceed_ptr, n, &mut vec_ptr);
    ceed_vector::CeedVector { ceed_ptr: self.ceed_ptr, ceed_vec_ptr: vec_ptr }
  }
}


/*
use std::io::prelude::*;
use std::io::{BufReader, SeekFrom};
use cfile::CFile;

let mut f = CFile::open_tmpfile().unwrap(); // open a tempfile

assert_eq!(f.write(b"test").unwrap(), 4); // write something to the stream

f.flush().unwrap(); // force to flush the stream

assert_eq!(f.seek(SeekFrom::Start(0)).unwrap(), 0); // seek to the beginning of stream

let mut r = BufReader::new(f);
let mut s = String::new();
assert_eq!(r.read_line(&mut s).unwrap(), 4); // read back the text
assert_eq!(s, "test");
*/



// use std::fs::File;
// use std::io::BufReader;
// use std::io::prelude::*;

/*
fn main() -> std::io::Result<()> {
  let file = File::open("foo.txt")?;
  let mut buf_reader = BufReader::new(file);
  let mut contents = String::new();
  buf_reader.read_to_string(&mut contents)?;
  assert_eq!(contents, "Hello, world!");
  Ok(())
}
*/
//CeedView(Ceed ceed, FILE *stream)

