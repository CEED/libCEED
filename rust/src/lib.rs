#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

#![allow(dead_code)]

mod rust_ceed {
  include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
mod ceed;
mod ceed_vector;

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn ceed_t000() {
    let ceed = ceed::Ceed::init("/cpu/self/ref/serial");
    println!("{}", ceed);
    /*
    unsafe {
      let mut ceed: rust_ceed::Ceed = libc::malloc(mem::size_of::<rust_ceed::Ceed>()) as rust_ceed::Ceed;
      let resource = "/cpu/self/ref/serial";
      rust_ceed::CeedInit(resource.as_ptr() as *const i8, &mut  ceed);
      rust_ceed::CeedDestroy(&mut ceed);
    }
    */
  }

  fn ceed_t001() {
    let ceed = ceed::Ceed::init("/cpu/self/ref/serial");
    let vec = ceed.vector(10);
  }
}