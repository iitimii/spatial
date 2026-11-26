# Spatial Engine

An open, native runtime for spatial-temporal world models—built from scratch in C++20 and Metal.

## Vision

The ability to realistically reconstruct, understand, and simulate anything physical—or imagined—in the digital world will reshape how we build, learn, play, and interact with machines.

Today, spatial AI is fragmented between heavy datacenter research stacks and traditional game engines that were never designed for neural models. **Spatial Engine** is an effort to build the missing runtime layer: a single, lightweight engine powering both **spatial understanding** and **spatial-temporal generation**, designed to scale across every class of hardware—from AI glasses and phones to laptops and datacenter clusters.

## Design Principles

- **Unified spatial-temporal primitives:** First-class support for N-D positional geometry, streaming video autoencoders, block-causal attention, and hierarchical spatial-temporal KV memory.
- **Native and dependency-light:** Clean C++20 core with zero Python runtime overhead, zero-copy unified memory (`mmap` → GPU), and static activation arenas so modern world models run comfortably on everyday laptops and edge devices.
- **Backend-portable:** Built with custom Apple Silicon Metal (`MSL`) kernels first, structured cleanly for Vulkan, WebGPU, and CUDA backends.

## Repository Layout

```text
include/spatial/   # Core tensor views, .spat format, Metal backend, spatial KV cache, model APIs
src/               # Engine implementation, custom Metal shaders, CLI, and native macOS interactive app
tools/             # Offline weight packing and numerical verification utilities
tests/             # Unit and numerical parity tests
```

## Building

Requires macOS on Apple Silicon with `cmake` and Apple Clang (`C++20`).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```

## Running

1. Pack open-weights checkpoints into the zero-copy `.spat` format using `tools/pack_waypoint.py`.
2. Run the headless rollout CLI or launch the native interactive Metal app:

```bash
# Headless rollout
./build/spatial_cli --rollout <model.spat> <vae.spat> <seed.ppm> <out_dir> [chunks] [steps]

# Interactive window (WASD to move, drag mouse to look, 2/3/4 to adjust steps, R to reset)
./build/spatial_app <model.spat> <vae.spat> [seed.ppm] [steps]
```

## License

See [LICENSE](LICENSE).
