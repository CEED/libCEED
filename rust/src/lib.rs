#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
  use super::*;
  use std::mem;

  #[test]
  fn ceed_init() {
    unsafe {
      
      let mut ceed: Ceed = libc::malloc(mem::size_of::<Ceed>()) as Ceed;
      let resource = "/cpu/self/ref/serial";
      CeedInit(resource.as_ptr() as *const i8, &mut  ceed);
      CeedDestroy(&mut ceed);
    }
  }
}