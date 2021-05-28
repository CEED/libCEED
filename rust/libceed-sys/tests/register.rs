/// Basic smoke test to confirm that the library is callable.
#[test]
fn test_import() {
    use libceed_sys::bind_ceed;
    unsafe {
        bind_ceed::CeedRegisterAll();
    }
}
