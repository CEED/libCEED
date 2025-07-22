#![no_std]
#![feature(asm_experimental_arch, abi_ptx, core_intrinsics)]
use core::ffi::c_void;
use core::intrinsics::abort;
use core::panic::PanicInfo;

use ndarray::ArrayView;

// This is a dummy allocator that always returns null. Heap allocations do not work on GPUs
use core::alloc::{GlobalAlloc, Layout};
pub struct Allocator;
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, _layout: Layout) -> *mut u8 {
        0 as *mut u8
    }
    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        abort(); // since we never allocate
    }
}
#[global_allocator]
static GLOBAL_ALLOCATOR: Allocator = Allocator;

#[doc = " A structure used to pass additional data to f_build_mass"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct BuildContext {
    pub dim: i32,
    pub space_dim: i32,
}

/*#[cfg(target = "nvptx64-nvidia-cuda")]
pub fn abort() -> ! {
    unsafe { asm!("trap;") }
    unreachable!();
}*/

//#[cfg(target = "nvptx64-nvidia-cuda")]
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    abort()
}

#[no_mangle]
pub unsafe extern "C" fn build_mass_rs(
    ctx: *mut c_void,
    q: i32,
    in_: *const *const f64,
    out: *mut *mut f64,
) -> i8 {
    let ctx: *mut BuildContext = unsafe { core::mem::transmute(ctx) };
    let ctx: &mut BuildContext = &mut *ctx;

    let in_slice = core::slice::from_raw_parts(in_, 2); // assuming 2 inputs: J and w

    let j_ptr = in_slice[0];
    let w_ptr = in_slice[1];

    let j = ArrayView::from_shape_ptr((ctx.dim as usize, ctx.dim as usize, q as usize), j_ptr);

    let w = core::slice::from_raw_parts(w_ptr, q as usize);

    let out_slice = core::slice::from_raw_parts_mut(out, 1);
    let q_data = core::slice::from_raw_parts_mut(out_slice[0], q as usize);

    match ctx.dim * 10 + ctx.space_dim {
        11 => {
            for i in 0..q as usize {
                q_data[i] = j[[0, 0, i]] * w[i];
            }
        }
        22 => {
            let q = q as usize;
            for i in 0..q {
                q_data[i] = (j[[0, 0, i]] * j[[1, 1, i]] - j[[0, 1, i]] * j[[1, 0, i]]) * w[i];
            }
        }
        33 => {
            let q = q as usize;
            for i in 0..q {
                q_data[i] = (j[[0, 0, i]]
                    * (j[[1, 1, i]] * j[[2, 2, i]] - j[[1, 2, i]] * j[[2, 1, i]])
                    - j[[0, 1, i]] * (j[[1, 0, i]] * j[[2, 2, i]] - j[[1, 2, i]] * j[[2, 0, i]])
                    + j[[0, 2, i]] * (j[[1, 0, i]] * j[[2, 1, i]] - j[[1, 1, i]] * j[[2, 0, i]]))
                    * w[i];
            }
        }
        _ => {
            abort();
        }
    }

    0
}
