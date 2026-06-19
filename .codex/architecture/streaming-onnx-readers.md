# Streaming ONNX Readers

## Purpose

Some package consumers need lightweight information from ONNX files before a
full import is appropriate. Full generated deserialization is intentionally not
used for these paths because `ModelProto.Graph` materializes graph nodes,
initializers, and raw tensor payloads.

The package provides streaming readers that scan only the protobuf fields they
need and seek over everything else.

## APIs

`Doji.AI.OnnxModelLoader.LoadMetadata(string filePath)` returns model-level
metadata through the existing `ONNXModelMetadata` type.

`Doji.AI.OnnxModelSignatureReader.ReadSignature(string filePath)` and
`ReadSignature(Stream stream)` return graph input signatures:

- input name
- tensor element type as ONNX integer value
- ordered dimensions
- static `dim_value` as `long`
- symbolic `dim_param`
- unknown dimensions where neither value nor parameter is present

The signature reader excludes graph inputs whose names appear in
`GraphProto.initializer`, matching ONNX convention for initializer-backed
constant inputs.

## Field Scanning

Metadata inspection scans `ModelProto` top-level fields and skips large graph
or function payloads.

Signature inspection scans:

- `ModelProto` field `7` (`graph`)
- `GraphProto` field `5` (`initializer`), but only `TensorProto.name`
- `GraphProto` field `11` (`input`), enough of `ValueInfoProto` to read tensor
  element type and shape dimensions

All other fields are skipped. Length-delimited nested messages are parsed with
bounded loops:

```csharp
var end = ReadLengthDelimitedEnd(stream);
while (stream.Position < end)
{
    // read selected fields, skip the rest
}
stream.Position = end;
```

This avoids `ReadBytes` for graph, node, initializer, and raw tensor fields.

## Dependency Boundary

The reader uses the generated `Doji.ONNX` types only as return-value-compatible
containers where needed and uses `ProtoReader` for primitive wire reads.

Shared protobuf helpers and generated schema output are owned by
`com.doji.formats.onnx`. If a reader needs a generally useful protobuf helper,
add it there first. Keep loader-specific parsing policy in this package.

## Test Expectations

Tests should build minimal protobuf streams directly. Do not rely on Unity
Editor ONNX import APIs for streaming-reader coverage.

Cover these cases when changing signature parsing:

- input names, element type, and dimensions are returned
- initializer-backed inputs are excluded
- input order is preserved
- static, symbolic, unknown, and large dimensions are represented correctly
- large initializer `raw_data` is skipped without reading
- large node payloads are skipped without reading
- valid models with no graph inputs return an empty signature

The strongest skip tests use a custom seekable stream that throws if the reader
attempts to read bytes from ranges that should be skipped.

