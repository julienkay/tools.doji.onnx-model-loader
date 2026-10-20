# AGENTS.md

## Project Intent

This repository contains the Unity package `tools.doji.onnx-model-loader` and its companion project `ONNX Model Loader`.

Loads ONNX files at runtime into Unity Inference Engine (Sentis) models.

Unity 6000.3 for development; the package supports 6000.3 LTS and later.

## Package Direction

- Follow `VISION.md` for durable product direction, scope and the decision test.
- Match the Sentis ONNX importer: a model that imports in the Editor must load at runtime with the same result.
- Parse with `tools.doji.formats.onnx`. `Runtime/InferenceEngine/Onnx/Onnx.cs` and `ProtoReader.cs` are internal copies: never edit them; change `G:\Unity\tools.doji.formats.onnx`, then run `tools/sync-onnx-parser/sync.sh`. `OnnxParserCopyTests` fails while a copy is out of date.
- Metadata and signature readers stream the file and seek past payloads; memory depends on the returned data, not on the file size.
- The package is in active development. Prefer the current design over backwards compatibility.
- Keep documentation about the current design and behavior. Update it with the code; remove temporary notes once the work they track is done.

## Source Of Truth

Prefer these locations:

- repository root: orient from the repository root for searches, docs, and cross-package work; from this companion Unity project working directory, the repository root is `../..`
- `tools.doji.onnx-model-loader/`: local package under development
  - `Runtime/InferenceEngine`: converter, loader and streaming readers, compiled into `Unity.InferenceEngine` through an asmref
  - `Tests/Editor`: EditMode tests and small test models
- `docs/`: user-facing package documentation only; keep maintainer notes and internal architecture docs out of DocFX content
- `internal-docs/`: design notes for maintainers; `internal-docs/README.md` indexes them. Read
  `internal-docs/architecture.md` before a change to conversion, streaming readers, or the package boundary.
- `projects/ONNX Model Loader/Assets`: companion-project assets and test scenes

Prefer working in the package folder when functionality is package-owned.

## Conventions

When adding code:

- organize scripts by task area; prefer colocated or clearly scoped folders such as docs-, tooling-, or workflow-specific script locations over a shared miscellaneous scripts bucket
