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

### Tensor data

`Convert` loads the file with `OnnxModelData.Load` from `tools.doji.formats.onnx`.
It reads the file into one managed buffer and parses metadata only; tensor data
stays in that buffer. `ONNXConstantsLoader.LoadConstant` finds the data of an
initializer or Constant node value with `OnnxModelData.TryGetData`:

- float32 and int32 `raw_data`: the constant is an `ArraySegment<byte>` of the
  file buffer (`TryGetRawBytes`), with no copy. Sentis pins segments with their
  offset; offsets are not aligned to the element size.
- float16, double, int64, bool, uint8 and int8 `raw_data`: converted from the
  borrowed bytes into one new array by span versions of the `ConstantTensor`
  conversions, which run the Sentis Burst cast jobs.
- typed fields (`float_data`, `int64_data`, ...): `DecodeTo` writes the
  `raw_data` layout into a new array, which is then used or converted as above.
- external data: one `OnnxExternalDataReader` per conversion reads the range
  given by `location`, `offset` and `length` into the final array. It opens
  each file once and checks that the location is relative and stays inside the
  model directory. The loader uses its `byte[]` overload, because Mono's
  `Stream.Read(Span)` copies through a temporary buffer.

A borrowed constant keeps the whole file buffer alive with the model,
including metadata, converted tensors and constants that Sentis later
replaces. So the converter borrows only when float32 and int32 `raw_data`
fill at least half of the buffer (`ShouldBorrowTensorData`); otherwise these
constants are copied too and the buffer is freed after conversion. The
converter drops its `OnnxModelData` reference when `Convert` returns.

The converter records problems with `Warn` instead of stopping, as the
Sentis importer does. After conversion, `Convert` throws `OnnxImportException`
if any error was recorded, with the model file name and every error message
in plain text. Warnings are logged with `Debug.LogWarning`. A missing external
weights file throws `FileNotFoundException`.

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
| `ConstantEncodingTests.cs` | Initializers in `raw_data`, typed fields, float16, double and external data with an offset give the same values; borrowed and converted constants |
| `ONNXNodeWrapperTests.cs` | INTS, FLOATS, and STRINGS attributes parsed from wire bytes |
| `LstmActivationArgumentsTests.cs` | LSTM `activation_alpha` and `activation_beta` assignment and defaults |

## Differences from the Sentis importer

- LSTM `activation_alpha` and `activation_beta` are read. Sentis looks for
  attribute type `FLOAT` instead of `FLOATS`, so it always uses the defaults.
- The values are consumed in order, only by activations that take that
  argument, as ONNX Runtime does. Sentis assigns them by activation index.

## Known limits

- The asmref adds assembly attributes to the Sentis assembly. It adds no
  types to it or to a Unity namespace.
- The converter depends on Sentis internals through `InternalsVisibleTo`. A Sentis
  upgrade can break it. No supported Sentis version range is declared beyond
  the `2.6.1` dependency.
- The sample asmdef is named `Doji.Temp_delete_useReference instead.Samples`.
