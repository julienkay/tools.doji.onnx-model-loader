# Getting started

## Installation

ONNX Model Loader is published on OpenUPM. Add a scoped registry for
`tools.doji` to `Packages/manifest.json`, then add the package:

```json
{
  "scopedRegistries": [
    {
      "name": "OpenUPM",
      "url": "https://package.openupm.com",
      "scopes": [ "tools.doji" ]
    }
  ],
  "dependencies": {
    "tools.doji.onnx-model-loader": "1.0.0"
  }
}
```

Package Manager also installs the dependencies:

- Inference Engine (`com.unity.ai.inference`) 2.6.1
- ONNX Format (`tools.doji.formats.onnx`), which parses the ONNX files

ONNX Model Loader requires Unity 6000.3 LTS or later.

If you build a package on top of ONNX Model Loader, add
`tools.doji.onnx-model-loader` to the `dependencies` of your own
`package.json` instead.

## Reference the assembly

If your code is in an assembly definition, add `Doji.AI.ONNXModelLoader` and
`Unity.InferenceEngine` to its references. Code in `Assembly-CSharp` needs no
change.

## Your first model

`OnnxModelLoader.Load` reads the file, converts it, and returns an Inference
Engine `Model`. Run it with a `Worker` as you would run an imported model:

```csharp
using Doji.AI;
using Unity.InferenceEngine;
using UnityEngine;

public class RunModel : MonoBehaviour
{
    Worker m_Worker;

    void Start()
    {
        string path = System.IO.Path.Combine(Application.persistentDataPath, "model.onnx");
        Model model = OnnxModelLoader.Load(path);
        m_Worker = new Worker(model, BackendType.GPUCompute);
    }

    void OnDestroy()
    {
        m_Worker?.Dispose();
    }
}
```

`Load` works in the Editor and in players. It reads a local file. On Android
and Web, `StreamingAssets` is not a local file: download or copy the model
into `Application.persistentDataPath` first.
