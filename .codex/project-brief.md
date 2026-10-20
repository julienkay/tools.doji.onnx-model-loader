# Project Brief

## Intent

ONNX Model Loader is a Unity package for loading ONNX models at runtime into
Unity Inference Engine. It also exposes lightweight ONNX inspection APIs for
tools that need model metadata or input signatures without importing a full
model or materializing tensor payloads.

## Repository Layout

- `tools.doji.onnx-model-loader/`: package source, tests, samples, and package
  manifest.
- `projects/ONNX Model Loader/`: Unity companion project used to work on and
  test the package.
- `docs/`: user-facing DocFX documentation.
- `.codex/`: maintainer-only context and architecture notes.

## Package Boundaries

`tools.doji.onnx-model-loader` depends on `tools.doji.formats.onnx` for generated
ONNX schema types and shared protobuf runtime helpers in the `Doji.ONNX`
namespace.

Do not modify generated ONNX schema code in this package. Generated schema
types, the generator, and shared protobuf reader behavior belong in
`G:\Unity\tools.doji.formats.onnx`.

Loader-specific utilities belong in this package. Examples include:

- model conversion into Unity Inference Engine objects
- metadata-only readers shaped for this package API
- graph input signature readers for package consumers

## Streaming Parser Constraints

Lightweight inspection APIs must keep memory proportional to the metadata being
returned, not to ONNX file size.

For metadata or signature inspection:

- do not call `File.ReadAllBytes`
- do not call `ModelProto.Deserialize`
- do not call `GraphProto.Deserialize`
- do not allocate byte arrays for graph, node, initializer, or raw tensor
  payloads
- use `FileStream` and bounded protobuf field scanning
- seek over skipped length-delimited payloads
- preserve useful exception behavior for malformed protobuf or missing files

## Validation

Unity project instructions prohibit `dotnet build` or compile checks unless the
user explicitly requests them.

When changing protobuf scanning behavior, add focused package tests with
synthetic ONNX protobuf streams. Large payload skip tests should prove the
reader seeks over skipped ranges instead of reading them.

