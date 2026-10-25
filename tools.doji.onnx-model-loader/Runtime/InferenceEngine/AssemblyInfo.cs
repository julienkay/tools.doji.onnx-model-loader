using System.Runtime.CompilerServices;

// The loader compiles into its own assembly and builds models from Sentis internals, such as layers and Model
// construction. This is the only code this package adds to Unity.InferenceEngine.
[assembly: InternalsVisibleTo("Doji.AI.ONNXModelLoader")]
[assembly: InternalsVisibleTo("Doji.AI.ONNXModelLoader.Editor.Tests")]
