# Architecture

## Assemblies

| Assembly | Contents |
|---|---|
| `Doji.AI.ONNXModelLoader` | `Runtime/Loader`: converter, loader, metadata reader |
| `Unity.InferenceEngine` (via asmref) | `Runtime/InferenceEngine/AssemblyInfo.cs`: `InternalsVisibleTo` only |
| `Doji.AI.ONNXModelLoader.Editor.Tests` | EditMode tests in `Tests/Editor` |

The converter builds models from Sentis internals such as `Layers` and `Model`
construction. `Runtime/InferenceEngine` holds an assembly definition
reference to Sentis' `Unity.InferenceEngine` with one file, which grants
`InternalsVisibleTo` to `Doji.AI.ONNXModelLoader` and the test assembly. All
other code compiles into `Doji.AI.ONNXModelLoader`, in namespace
`Doji.AI.InferenceEngine`. That assembly references `Unity.InferenceEngine`
and `Doji.Onnx`, so the converter parses with the `Doji.ONNX` types and
`ProtoReader` from `tools.doji.formats.onnx`. This package has no parser code
of its own.

`Runtime/Loader/AssemblyInfo.cs` adds `InternalsVisibleTo` for the test
assembly, so tests can use the internal converter types.

The companion project references the package with `file:` and lists it in
`testables`.

## Conversion

`OnnxModelLoader.Load(path)` runs `ONNXModelConverter.Convert(path)` and
returns a Sentis `Model`. The converter follows the Sentis ONNX importer:
`ONNXNodeWrapper` adapts nodes, `ONNXConstantsLoader` reads initializers,
`ModelConverterBase` builds layers.

## Streaming readers

Some callers need model data before a full conversion.
`ModelProto.Deserialize` would materialize nodes, initializers and raw tensor
payloads, so these readers scan only the fields they need and seek past the
rest.

- `OnnxModelLoader.LoadMetadata(path)` returns `ONNXModelMetadata` from the
  top-level `ModelProto` fields. It seeks past graph, opset, training,
  function, and configuration payloads with `ProtoReader.SkipField`.
- Graph inputs come from `OnnxModelSignatureReader` in
  `tools.doji.formats.onnx`.

The reader uses the bounded `ProtoReader` API: every length is checked
against the end of its message, and skipped payloads are never read. Memory
depends on the returned data, not on the file size.

## Package boundary

Generated schema types (`Doji.ONNX`), `ProtoReader`, and the generator
belong to `tools.doji.formats.onnx`. Make shared protobuf helpers and
generator changes there first. Loader-specific parsing policy and Sentis
conversion stay here.

## Tests

Tests build protobuf bytes in memory with `ProtoWriter` and do not use Editor
ONNX import APIs.

| File | Covers |
|---|---|
| `ONNXModelLoaderTests.cs` | Conversion of the models in `Tests/Editor/Models`, a Conv with `pads` run on the CPU, metadata reads that skip the graph |
| `ONNXNodeWrapperTests.cs` | INTS, FLOATS, and STRINGS attributes parsed from wire bytes |

## Known limits

- The asmref adds assembly attributes to the Sentis assembly. It adds no
  types to it or to a Unity namespace.
- `ONNXNodeWrapper.GetOptionalFloatArray` looks for attribute type `FLOAT`
  instead of `FLOATS`, as the Sentis importer does. Only LSTM
  `activation_alpha` and `activation_beta` use it.
- The converter depends on Sentis internals through `InternalsVisibleTo`. A Sentis
  upgrade can break it. No supported Sentis version range is declared beyond
  the `2.6.1` dependency.
- The sample asmdef is named `Doji.Temp_delete_useReference instead.Samples`.
- The companion project still uses Unity 6000.3.10f1.
