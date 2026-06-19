# AGENTS.md

## Repository

ONNX Model Loader is a Unity package that loads ONNX models at runtime for use
with Unity Inference Engine. The package lives in
`com.doji.onnx-model-loader/`; `projects/ONNX Model Loader/` is the Unity
development project used to work on and test the package.

This package depends on `com.doji.formats.onnx` for the generated ONNX schema
types and shared protobuf runtime helpers (`Doji.ONNX.Onnx.cs`,
`ProtoReader.cs`). Do not modify generated ONNX schema code in this package;
put loader-specific parsing utilities here, and make shared protobuf helper or
generator/template changes in `G:\Unity\com.doji.formats.onnx`.

## Source Of Truth

Prefer these locations:

- repository root: orient from the repository root for searches, docs, and
  cross-package work; from the companion Unity project working directory, the
  repository root is `../..`
- `com.doji.onnx-model-loader/`: local package under development
- `docs/`: user-facing package documentation only; keep maintainer notes and
  internal architecture docs out of DocFX content
- `.codex/project-brief.md`: stable project intent, package boundaries, and
  constraints
- `.codex/architecture/`: durable maintainer reference docs; read only the
  files relevant to the current task
- `projects/ONNX Model Loader/Assets`: companion-project assets and test scenes

Prefer working in the package folder when functionality is package-owned.

## Agent Instructions

- Read [docs/index.md](docs/index.md) before making project-level changes.
- Read [.codex/project-brief.md](.codex/project-brief.md) before changing
  package architecture, generated-code boundaries, or streaming ONNX parsers.
- Unity project: do not run `dotnet build` or compile checks unless explicitly requested.
- 
