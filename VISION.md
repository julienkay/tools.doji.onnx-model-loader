# ONNX Model Loader

## Purpose

ONNX Model Loader loads ONNX files at runtime into Unity Inference Engine
(Sentis) models. Applications can use models that were not imported in the
Editor, for example models that users supply or download after install.

The result must match what the Sentis Editor importer produces for the same
file.

## Users

- Developers of applications that load models chosen after build time.
- Packages that download models at runtime, such as a future runtime mode of
  `tools.doji.model-hub`.

## Principles

- **Same model as the importer.** Follow the Sentis ONNX importer behavior.
  A model that imports in the Editor must load at runtime with the same
  layers, constants, and outputs.
- **Parse with `tools.doji.formats.onnx`.** The loader has no parser code of
  its own. Parsing changes go into `tools.doji.formats.onnx` first. Users get
  `OnnxModelSignatureReader` through the package dependency.
- **Track Sentis explicitly.** The converter depends on Sentis internals.
  Declare the supported Sentis version, and update the converter with each
  Sentis upgrade.
- **Read only what you need.** Metadata and signature APIs must not load
  weights. Their memory use depends on the returned data, not on the file
  size.
- **Show clear errors.** Unsupported operators or data types give an error
  that names the operator and the model.

## Out of scope

- Running inference. Sentis does that.
- Inference runtimes other than Sentis, such as ONNX Runtime.
- Editing, optimizing, or quantizing models.

## Decision test

Does the change help a Unity application load an ONNX model at runtime that
behaves the same as an imported one? If not, do not make the change.
