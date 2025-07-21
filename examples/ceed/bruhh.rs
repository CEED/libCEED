#![no_std]
#![feature(asm_experimental_arch, abi_ptx)]
use core::arch::asm;
use core::panic::PanicInfo;

pub fn abort() -> ! {
    unsafe { asm!("trap;") }
    unreachable!();
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    abort()
}

#[no_mangle]
pub extern "C" fn add_num(x: i32) -> i32 {
    return x + 1;
}
