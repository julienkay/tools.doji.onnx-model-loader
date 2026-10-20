# Architecture

## Assemblies

| Assembly | Contents |
|---|---|
| `Unity.InferenceEngine` (via asmref) | `Runtime/InferenceEngine`: converter, loader, streaming readers |
| `Doji.AI.ONNX.ModelLoader` | `Runtime/New Folder`: asmdef only, no sources |
| `Doji.AI.ONNXModelLoader.Editor.Tests` | EditMode tests in `Tests/Editor` |

`Runtime/InferenceEngine` holds an assembly definition reference to Sentis'
`Unity.InferenceEngine`. Its code compiles into the Sentis assembly, so the
converter can use Sentis internals such as `ModelConverterBase` layers and
`Model` construction.

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
  top-level `ModelProto` fields and skips graph and function payloads.
- `OnnxModelSignatureReader.ReadSignature(path | stream)` returns graph
  inputs: name, element type, and ordered dimensions (`dim_value`,
  `dim_param`, or unknown). It reads `ModelProto.graph` (field 7), only
  `TensorProto.name` of `GraphProto.initializer` (field 5), and
  `GraphProto.input` (field 11). Inputs that match an initializer name are
  removed.

Length-delimited messages are read in bounded loops that end with
`stream.Position = end`. Readers never call `File.ReadAllBytes`,
`ModelProto.Deserialize`, or `GraphProto.Deserialize`, and never allocate
graph, node, initializer, or tensor payloads. Memory depends on the returned
data, not on the file size.

## Package boundary

Generated schema types (`Doji.ONNX`), `ProtoReader`, and the generator
belong to `tools.doji.formats.onnx`. Make shared protobuf helpers and
generator changes there first. Loader-specific parsing policy and Sentis
conversion stay here.

## Tests

Tests build minimal protobuf streams in memory and do not use Editor ONNX
import APIs. Skip tests use a seekable stream that throws when the reader
touches a byte range that it must skip. `ONNXModelLoaderTests` also loads
the small models in `Tests/Editor/Models`.

## Known limits

- `Runtime/InferenceEngine/Onnx` still has its own copies of `Onnx.cs`,
  `ProtoReader.cs`, and `OnnxModelSignatureReader.cs`. They duplicate
  `tools.doji.formats.onnx` and define `Doji.ONNX` types a second time,
  inside `Unity.InferenceEngine`.
- The converter depends on Sentis internals through the asmref. A Sentis
  upgrade can break it. No supported Sentis version range is declared beyond
  the `2.6.1` dependency.
- `Doji.AI.ONNX.ModelLoader` in `Runtime/New Folder` compiles no code.
- The sample asmdef is named `Doji.Temp_delete_useReference instead.Samples`.
- The companion project still uses Unity 6000.3.10f1.
