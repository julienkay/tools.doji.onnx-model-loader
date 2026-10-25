# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- List attributes (`INTS`, `FLOATS`, `STRINGS`), such as Conv `pads`, `kernel_shape` and `strides`, are read instead of silently replaced by defaults. The parser copy had wrong `AttributeType` values.
- Negative `int32` values in tensors and attributes parse.
- Installing the loader together with `tools.doji.formats.onnx` no longer defines the `Doji.ONNX` types twice (CS0433).

### Changed

- The loader compiles into its own assembly, `Doji.AI.ONNXModelLoader`, and parses with `tools.doji.formats.onnx` instead of its own parser copy. Reference `Doji.AI.ONNXModelLoader` from your assembly definition. Use `OnnxModelSignatureReader` from `tools.doji.formats.onnx`.
- The converter types moved from namespace `Unity.InferenceEngine` to `Doji.AI.InferenceEngine`. Only an `InternalsVisibleTo` attribute still compiles into the Sentis assembly.

## [1.0.0] - 2025-10-27

- Initial Release