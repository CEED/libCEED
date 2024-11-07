const std = @import("std");

pub fn build(b: *std.Build) void {
    const exe_name = b.option(
        []const u8,
        "exe_name",
        "Name of the executable",
    ) orelse "ex01";
    var buffer = [_]u8{undefined} ** 16;
    const exe_path = std.fmt.bufPrint(&buffer, "src/{s}.zig", .{exe_name}) catch |err| std.debug.panic("Error: {any}", .{err});
    const exe = b.addExecutable(.{
        .name = exe_name,
        .root_source_file = b.path(exe_path),
        .target = b.standardTargetOptions(.{}),
        .optimize = b.standardOptimizeOption(.{}),
    });
    exe.addIncludePath(b.path("../../include"));
    exe.addLibraryPath(b.path("../../lib"));
    exe.linkSystemLibrary("ceed");                                                                    
    exe.linkLibC();

    b.installArtifact(exe);
}
