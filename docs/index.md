# ONNX Model Loader

ONNX Model Loader loads ONNX files at runtime into Unity Inference Engine
(Sentis) models. Use it when the model is not known at build time, for
example a model that users supply or that the application downloads after
install.

The loaded model matches what the Inference Engine Editor importer makes from
the same file: same layers, same constants, same outputs.

## Load a model

```csharp
using Doji.AI;
using Unity.InferenceEngine;

Model model = OnnxModelLoader.Load("/path/to/model.onnx");
using var worker = new Worker(model, BackendType.GPUCompute);
```

[Getting started](manual/getting-started.md) explains installation and the
assembly reference. [Loading models](manual/loading-models.md) explains
metadata, input signatures, external weights and errors.
[Quantization and .sentis files](manual/quantization-and-sentis-files.md)
explains how to quantize weights while loading, how to convert an ONNX file
to a `.sentis` file, and which API loads which file.

The [Scripting API](api/index.md) describes every public type and member.
