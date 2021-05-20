pub mod bind_ceed {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(dead_code)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
