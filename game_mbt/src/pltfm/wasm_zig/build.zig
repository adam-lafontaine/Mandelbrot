const std = @import("std");

const name = "mandelbrot";

const main_cpp = "../sdl2/mbt_sdl2_main.cpp";

const cpp_flags = &[_][]const u8 {
    "-std=c++20",

    "-O3",
    "-DNDEBUG",

    "-s USE_SDL=2",
    "-s USE_SDL_MIXER=2",
    "-s USE_OGG=1",
    "-sALLOW_MEMORY_GROWTH"
};


pub fn build(b: *std.Build) void 
{
    // Define the target: WebAssembly (wasm32-freestanding)
    const target = b.standardTargetOptions(.{
        .default_target = .{
            .cpu_arch = .wasm32,
            .os_tag = .freestanding,
        },
    });

    // Define the optimization mode (e.g., ReleaseSmall for smaller WASM output)
    const optimize = b.standardOptimizeOption(.{});

    // Create an executable for the C++ program
    const exe = b.addExecutable(.{
        .name = name,
        .root_source_file = null, // No Zig root file; we're compiling C++
        .target = target,
        .optimize = optimize,
    });

    exe.addCSourceFiles(.{
        .files = &[_][]const u8{
            main_cpp,
            //"utils.cpp",
        },
        .flags = cpp_flags,
    });

    // Ensure the module is exported (needed for WASM)
    exe.export_symbol_names = &[_][]const u8{"add"};

    // Install the WASM file to the output directory
    b.installArtifact(exe);

    // Run step
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| 
    {
        run_cmd.addArgs(args);
    }
    b.step("run", "Run the app").dependOn(&run_cmd.step);
}