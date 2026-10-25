# ONNX Model Loader

Loads ONNX files at runtime into Unity Inference Engine (Sentis) models. The
loaded model matches what the Editor importer makes from the same file.

```csharp
Model model = Doji.AI.OnnxModelLoader.Load("/path/to/model.onnx");
```

Documentation: https://docs.doji-tech.com/tools.doji.onnx-model-loader/
