# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `OnnxModelLoader.Save(path, sentisPath, quantization, beforeSave)` converts an ONNX model to a .sentis file without holding all float32 weights in memory. `beforeSave` can change the model before it is written, for example to fix input shapes.
- `OnnxModelLoader.Load(path, quantization)` quantizes weights to float16 or uint8 while loading, one weight at a time, so models larger than the memory load. It quantizes what `ModelQuantizer.QuantizeWeights` quantizes, except rank-1 tensors such as biases.

### Fixed

- List attributes (`INTS`, `FLOATS`, `STRINGS`), such as Conv `pads`, `kernel_shape` and `strides`, are read instead of silently replaced by defaults. The parser copy had wrong `AttributeType` values.
- Negative `int32` values in tensors and attributes parse.
- LSTM `activation_alpha` and `activation_beta` are read instead of silently replaced by defaults. Like ONNX Runtime, each value goes to the next activation that takes that argument (for example LeakyRelu, HardSigmoid or ScaledTanh).
- Installing the loader together with `tools.doji.formats.onnx` no longer defines the `Doji.ONNX` types twice (CS0433).
- A missing external weights file throws `FileNotFoundException` instead of returning `null`.
- External tensor data is read from its `offset` entry. Before, tensors were read one after another from the start of the file, so files with gaps or another order loaded wrong weights.
- Tensors stored in typed fields other than `float_data`, `int32_data`, `int64_data` and `double_data`, for example float16 or int8 values in `int32_data`, load instead of throwing.
- Conversion errors throw the public `OnnxImportException`, whose message names the model file and lists every error. Before, some errors returned a broken model, and unsupported operators gave only "See errors for details".
- Conversion warnings, such as an unsupported attribute value replaced by its default, are logged with `Debug.LogWarning`.

### Changed

- The loader compiles into its own assembly, `Doji.AI.ONNXModelLoader`, and parses with `tools.doji.formats.onnx` instead of its own parser copy. Reference `Doji.AI.ONNXModelLoader` from your assembly definition. Use `OnnxModelSignatureReader` from `tools.doji.formats.onnx`.
- The converter types moved from namespace `Unity.InferenceEngine` to `Doji.AI.InferenceEngine`. Only an `InternalsVisibleTo` attribute still compiles into the Sentis assembly.
- Requires Unity 6000.3 or later, like `tools.doji.formats.onnx`.
- `OnnxModelLoader.Load` reads the file with `OnnxModelData` and builds float32 and int32 constants on the file buffer without a copy. Float32 models load faster. The model keeps the file buffer while it lives.

## [1.0.0] - 2025-10-27

- Initial Release