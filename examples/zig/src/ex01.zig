const std = @import("std");
const libceed = @cImport({
    @cInclude("ceed/ceed.h");
});

fn ceed_call(err: c_int) void {
    if (err != libceed.CEED_ERROR_SUCCESS) {
        std.debug.panic("libCEED Error: {s}", .{libceed.CeedErrorTypes[@intCast(err)]});
    }
}

pub fn main() void {
    var ceed: libceed.Ceed = undefined;

    ceed_call(libceed.CeedInit("/cpu/self", &ceed));
    ceed_call(libceed.CeedView(ceed, libceed.stdout));
    ceed_call(libceed.CeedDestroy(&ceed));
}
