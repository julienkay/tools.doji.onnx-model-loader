# Loading models

## Read metadata without loading the model

`OnnxModelLoader.LoadMetadata` returns the model-level fields of the file:
producer, IR version, model version, domain, doc string and `metadata_props`.
It seeks past the graph and the weights, so it is fast even for
multi-gigabyte models.

```csharp
ONNXModelMetadata metadata = OnnxModelLoader.LoadMetadata(path);
Debug.Log($"{metadata.ProducerName} {metadata.ProducerVersion}, IR {metadata.IRVersion}");
foreach (var pair in metadata.MetadataProps)
    Debug.Log($"{pair.Key} = {pair.Value}");
```

## Read the inputs without loading the model

To read input names, element types and shapes, use `OnnxModelSignatureReader`
from the ONNX package (`tools.doji.formats.onnx`). It is installed with the
loader:

```csharp
using Doji.ONNX;

OnnxModelSignature signature = OnnxModelSignatureReader.ReadSignature(path);
foreach (OnnxModelInputSignature input in signature.Inputs)
    Debug.Log($"{input.Name}: {input.ElementType}, {input.Dimensions.Length} dimensions");
```

## External weights

Models larger than 2 GB keep their weights in separate files next to the
`.onnx` file. `Load` opens these files from the folder of the `.onnx` file,
with the names that the model stores. Keep all files of a model in one folder.

If a weights file is missing, `Load` throws `FileNotFoundException`. The
message names the missing file and the folder it was expected in.

## Errors and warnings

`Load` throws `OnnxImportException` (namespace `Doji.AI.InferenceEngine`) when
the model cannot be converted, for example when it uses an operator or a
tensor data type that Inference Engine does not support. The message names
the model file and lists every error:

```csharp
try
{
    model = OnnxModelLoader.Load(path);
}
catch (OnnxImportException e)
{
    Debug.LogError(e.Message);
}
```

Attribute values that Inference Engine does not support, such as an unknown
`auto_pad` mode, fall back to the same default as the Editor importer. `Load`
logs each of these with `Debug.LogWarning`, prefixed with the model file
name.

## Supported Inference Engine version

The converter uses internal APIs of Inference Engine, so each release supports
one Inference Engine version: 2.6.1 for this release. Do not upgrade Inference
Engine on its own; upgrade ONNX Model Loader to a release for the new
version.
