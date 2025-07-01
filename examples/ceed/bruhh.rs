#![no_std]
use core::panic::PanicInfo;

extern "C" {
    fn abort() -> !;
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    unsafe { abort() }
}

#[no_mangle]
pub extern "C" fn add_num(x: u32) -> u32 {
    return x + 1;
}
