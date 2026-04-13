# CUDA-CenterPoint, CUDA-BEVFusion, and 3DSparseConvolution

This note explains how `CUDA-CenterPoint`, `CUDA-BEVFusion`, and `libraries/3DSparseConvolution`
fit together in `Lidar_AI_Solution_vivid`, based on the current source code.

It also explains why switching a sparse backbone ONNX to a PTQ/QAT export is not enough by itself,
and why the caller must often pass `spconv::Precision::Int8`.

## Scope

- `CUDA-CenterPoint`
- `CUDA-BEVFusion`
- `libraries/3DSparseConvolution`

This document focuses on the lidar sparse backbone path only. TensorRT camera, fuser, and head
engines are mentioned only where they connect to the sparse backbone.

## One-sentence summary

`libraries/3DSparseConvolution` provides the sparse convolution runtime and ONNX parser, while
`CUDA-CenterPoint` and `CUDA-BEVFusion` are application layers that do voxelization, call into
that sparse runtime for the lidar backbone, and then continue with their own downstream heads.

## High-level relationship

### `libraries/3DSparseConvolution`

This is the reusable sparse backbone runtime:

- It defines the public `spconv` API.
- It parses sparse-conv-compatible ONNX graphs.
- It builds and runs a sparse execution engine.
- It owns the notion of sparse inference precision such as `Float16` and `Int8`.

Key source files:

- `libraries/3DSparseConvolution/src/onnx-parser.hpp`
- `libraries/3DSparseConvolution/src/onnx-parser.cpp`
- `libraries/3DSparseConvolution/src/infer.cpp`
- `libraries/3DSparseConvolution/libspconv/include/spconv/engine.hpp`

### `CUDA-CenterPoint`

This is an application that wraps:

1. CUDA voxelization
2. sparse lidar backbone via `spconv`
3. TensorRT neck and head
4. CUDA decode and NMS

It does not implement sparse convolution itself. It calls `spconv::load_engine_from_onnx(...)`
and feeds sparse tensors into the resulting engine.

### `CUDA-BEVFusion`

This is another application layer that also uses the same sparse backbone runtime for lidar.

It wraps:

1. camera preprocessing and TensorRT camera path
2. lidar voxelization
3. sparse lidar backbone via `spconv`
4. fusion and detection heads

Like `CUDA-CenterPoint`, it does not implement sparse convolution itself.

## Where each project touches `spconv`

### `CUDA-CenterPoint`

The sparse backbone is created in `CenterPoint::CenterPoint(...)`:

- `CUDA-CenterPoint/src/centerpoint.cpp`

Current vivid code:

```cpp
std::string scn_path = "../model/centerpoint.scn.onnx.ptq";
scn_engine_ = spconv::load_engine_from_onnx(scn_path, spconv::Precision::Int8);
```

This tells us two things:

- `CUDA-CenterPoint` is responsible for choosing which sparse ONNX file to load.
- `CUDA-CenterPoint` is also responsible for choosing the sparse inference precision.

Later in `CenterPoint::doinfer(...)`, the app:

- runs voxelization
- obtains sparse features and indices
- passes them into `scn_engine_`
- takes the backbone output and feeds it into TensorRT

So the data flow is:

`points -> voxelization -> sparse backbone via spconv -> TensorRT neck/head -> decode/NMS`

### `CUDA-BEVFusion`

The sparse lidar backbone is created in:

- `CUDA-BEVFusion/src/bevfusion/lidar-scn.cpp`

The key line is:

```cpp
native_scn_ = spconv::load_engine_from_onnx(param_.model, static_cast<spconv::Precision>(param_.precision));
```

So `CUDA-BEVFusion` is wired the same way:

- the application chooses the ONNX path
- the application chooses the sparse inference precision
- `libraries/3DSparseConvolution` executes the model

The precision value itself is configured earlier from the app entry point:

- `CUDA-BEVFusion/src/main.cpp`

If the string precision is `"int8"`, then:

```cpp
scn.precision = bevfusion::lidar::Precision::Int8;
```

otherwise it uses `Float16`.

## What `libraries/3DSparseConvolution` actually provides

The most important API boundary is:

- `libraries/3DSparseConvolution/src/onnx-parser.hpp`

```cpp
std::shared_ptr<Engine> load_engine_from_onnx(
    const std::string& onnx_file,
    Precision inference_precision = Precision::Float16,
    bool sortmask = false,
    bool enable_blackwell = false,
    bool with_auxiliary_stream = false,
    unsigned int fixed_launch_points = 10000,
    void* stream = nullptr
);
```

This header is the clearest source of truth for precision behavior:

- if the caller does not pass a precision, the default is `Precision::Float16`
- therefore the application must opt into `Int8`

The parser implementation shows how ONNX layer attributes are consumed:

- `libraries/3DSparseConvolution/src/onnx-parser.cpp`

For sparse conv nodes:

```cpp
get_attribute(node, "precision").s() == "int8" ? Precision::Int8 : Precision::Float16,
get_attribute(node, "output_precision").s() == "int8" ? Precision::Int8 : Precision::Float16,
```

For add and quant-add nodes:

```cpp
get_attribute(node, "precision").s() == "int8" ? Precision::Int8 : Precision::Float16,
get_attribute(node, "output_precision").s() == "int8" ? Precision::Int8 : Precision::Float16,
```

At the end, the parser builds the runtime engine with the caller-selected precision:

```cpp
return builder->build(precision, sortmask, enable_blackwell, with_auxiliary_stream, stream);
```

That is the second important source of truth: the final engine build still receives a top-level
`precision` selected by the caller.

## Why `spconv::Precision::Int8` must be added explicitly

This is the main point that usually causes confusion.

### Short answer

Because the sparse runtime defaults to `Float16` unless the caller explicitly asks for `Int8`.

### Source-code evidence

1. The API default is `Precision::Float16`.

From `libraries/3DSparseConvolution/src/onnx-parser.hpp`:

```cpp
Precision inference_precision = Precision::Float16
```

2. The standalone `infer` tool also treats `Int8` as an explicit opt-in.

From `libraries/3DSparseConvolution/src/infer.cpp`:

```cpp
task.main_precision = task.int8 ? spconv::Precision::Int8 : spconv::Precision::Float16;
```

3. The help text says the same thing: `--int8` is a separate switch, and it may be ignored if the
model does not contain the required dynamic-range information.

4. The parser reads per-layer `precision` and `output_precision` from the ONNX, but the final
engine is still built with a top-level `precision` argument selected by the caller.

### What this means in practice

Even if your ONNX was exported from PTQ/QAT and contains layer attributes such as:

- `precision = "int8"`
- `output_precision = "int8"`
- dynamic-range attributes like `input_dynamic_range`

that still does not automatically guarantee that the application will execute the sparse backbone
in `Int8` mode.

If the caller does this:

```cpp
spconv::load_engine_from_onnx(path);
```

then the runtime uses the default:

```cpp
spconv::Precision::Float16
```

So changing only the ONNX file path is not enough when the application code still uses the default
precision.

### Important nuance

`Int8` compute does not mean the app input tensors become `Int8`.

Both application paths still feed:

- sparse features as `Float16`
- sparse indices as `Int32`

For example, `CUDA-BEVFusion/src/bevfusion/lidar-scn.cpp` passes:

```cpp
spconv::DataType::Float16
spconv::DataType::Int32
```

So the runtime precision choice is about how the sparse engine computes internally, not about
changing the application-side sparse tensor interface.

## Build and link relationship

### `CUDA-CenterPoint`

`CUDA-CenterPoint/CMakeLists.txt` tries to use the repo-local `libraries/3DSparseConvolution`
first:

- it auto-detects `../libraries/3DSparseConvolution/libspconv`
- it searches for `libspconv.so`
- it can also include parser sources from `../libraries/3DSparseConvolution/src`

This means `CUDA-CenterPoint` in vivid is intended to consume the shared sparse runtime from the
repo, not a completely separate implementation.

### `CUDA-BEVFusion`

`CUDA-BEVFusion/CMakeLists.txt` links directly against:

- `../libraries/3DSparseConvolution/libspconv/include`
- `../libraries/3DSparseConvolution/libspconv/lib/${arch}_cuda$ENV{SPCONV_CUDA_VERSION}`

So `CUDA-BEVFusion` is even more explicitly tied to `libraries/3DSparseConvolution`.

## End-to-end dataflow comparison

### `CUDA-CenterPoint`

1. `main.cpp` loads the TensorRT plan for neck/head.
2. `src/centerpoint.cpp` creates voxelization and postprocess helpers.
3. `src/centerpoint.cpp` loads sparse backbone ONNX through `spconv`.
4. `doinfer(...)` voxelizes raw points into sparse features/indices.
5. `spconv` runs the 3D backbone.
6. TensorRT consumes the backbone dense output for `RPN + CenterHead`.
7. CUDA postprocess decodes and applies NMS.

### `CUDA-BEVFusion`

1. `src/main.cpp` builds app parameters.
2. It chooses lidar backbone path and lidar sparse precision.
3. `src/bevfusion/lidar-scn.cpp` constructs the sparse engine through `spconv`.
4. Lidar voxelization produces sparse features/indices.
5. `spconv` runs the lidar backbone.
6. Other TensorRT modules handle camera, view transform, fusion, and heads.

## Practical debugging checklist

If a PTQ sparse ONNX does not speed up the backbone path, verify the following in order:

1. The application is loading the intended ONNX file path.
2. The application is passing `spconv::Precision::Int8`.
3. The ONNX was exported with quantization attributes and dynamic ranges.
4. The binary is linked against the expected `libspconv.so`.
5. The runtime log prints enough information to confirm the selected path and precision.

## Bottom line

The codebase split is:

- `libraries/3DSparseConvolution`: sparse runtime and ONNX parser
- `CUDA-CenterPoint`: CenterPoint app using that sparse runtime
- `CUDA-BEVFusion`: BEVFusion app using that sparse runtime

And the reason we know `spconv::Precision::Int8` must be added is not guesswork. It comes directly
from the library API and its caller examples:

- `load_engine_from_onnx(...)` defaults to `Precision::Float16`
- `infer.cpp` switches to `Int8` only when explicitly requested
- both apps are the ones responsible for passing the sparse precision into `spconv`

Therefore, swapping in a PTQ/QAT sparse ONNX without also selecting `spconv::Precision::Int8` can
still leave the sparse backbone running in `Float16`.
