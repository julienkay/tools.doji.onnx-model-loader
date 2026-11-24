# Quantization and .sentis files

A model reaches your application as one of two file types. Each type loads
with a different API:

| File | Load with | Conversion at load | Weights |
|---|---|---|---|
| `.onnx` | `OnnxModelLoader.Load` (this package) | Yes, on every load | float32, or quantized while loading |
| `.sentis` | `ModelLoader.Load` (Inference Engine) | No | As stored in the file |

`OnnxModelLoader.Load` does not read `.sentis` files, and `ModelLoader.Load`
does not read `.onnx` files.

## Load an ONNX file

`OnnxModelLoader.Load(path)` converts the file and returns a model with
float32 weights, the same as the ONNX file.

`OnnxModelLoader.Load(path, quantization)` converts the file and quantizes the
weights while it loads. Pass `QuantizationType.Float16` or
`QuantizationType.Uint8`:

```csharp
Model model = OnnxModelLoader.Load(path, QuantizationType.Float16);
```

- The quantized weights are the weights of `MatMul`, `Gemm`, `Conv`,
  `ConvTranspose` and `Gather`, the same weights that Inference Engine's
  `ModelQuantizer.QuantizeWeights` quantizes. Rank-1 tensors, such as
  biases, stay float32.
- The loader quantizes one weight at a time. Only that weight is in memory as
  float32, so a model loads if its quantized weights fit in memory, even if
  its float32 weights do not.
- The returned model holds all the quantized weights. A float16 model needs
  about half the memory of the float32 model, a uint8 model about a quarter.
- The model dequantizes each weight to float32 when it runs, as a model
  quantized with `ModelQuantizer` does.

`Load` converts the file every time. To load the same model often, convert
it to a `.sentis` file once.

## Convert an ONNX file to a .sentis file

`OnnxModelLoader.Save` converts an ONNX file and writes a `.sentis` file. With
a quantization type, the file holds the quantized weights:

```csharp
OnnxModelLoader.Save(onnxPath, sentisPath, QuantizationType.Float16);
```

`Save` never holds all weights in memory, quantized or not. It reads each
weight from the ONNX file, quantizes it, and writes it to the `.sentis` file,
one weight at a time. Use `Save` instead of `Load` followed by
`ModelWriter.Save`, which needs all weights in memory.

`Save` can change the model before it writes it, for example to fix input
shapes. When `beforeSave` runs, the weight constants have no data yet: change
only the layers, inputs and outputs.

```csharp
OnnxModelLoader.Save(onnxPath, sentisPath, QuantizationType.Uint8, model =>
{
    // Change model.inputs or model.layers here.
});
```

## Load a .sentis file

Load a `.sentis` file with Inference Engine. It needs no conversion, and its
weights load as the file stores them: a file that `Save` wrote with
quantization loads with quantized weights.

```csharp
Model model = ModelLoader.Load(sentisPath);
```
