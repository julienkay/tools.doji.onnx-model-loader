# Maintainer Notes

This directory contains maintainer-facing context for ONNX Model Loader. Keep
user-facing package documentation under `docs/`; keep implementation rationale,
cross-package boundaries, and parser architecture notes here.

## Start Here

- [Project Brief](project-brief.md) - stable package intent, repository layout,
  dependency boundaries, and validation constraints.
- [Streaming ONNX Readers](architecture/streaming-onnx-readers.md) - metadata
  and graph-signature parsers that intentionally avoid full `ModelProto`
  deserialization.

