using System;
using System.Collections.Generic;
using System.Linq;
using Doji.ONNX;
using Unity.InferenceEngine;
using Unity.InferenceEngine.Compiler.Passes.Optimization;
using Unity.InferenceEngine.Graph;
using Unity.Jobs;
using UnityEngine;

namespace Doji.AI.InferenceEngine
{
    /// <summary>
    /// Quantizes weights while a model loads, one weight at a time, so a large model never holds all its float32
    /// weights. A weight is a stand-in input while the graph is optimized, because the optimizer would fold a
    /// dequantized constant back into float32. Then it is loaded, quantized and dequantized in the graph, as
    /// <see cref="ModelQuantizer.QuantizeWeights"/> does.
    /// </summary>
    static class WeightQuantizer
    {
        /// <summary>
        /// The weight input of each operator whose layer Inference Engine quantizes.
        /// </summary>
        static readonly Dictionary<string, int> k_WeightInputs = new()
        {
            { "MatMul", 1 }, { "Gemm", 1 }, { "Conv", 1 }, { "ConvTranspose", 1 }, { "Gather", 0 }
        };

        /// <summary>
        /// Names of the float initializers of rank 2 or more that are a weight input of a quantized layer.
        /// </summary>
        public static HashSet<string> FindWeights(GraphProto graph)
        {
            var floats = graph.Initializer
                .Where(i => i.Dims.Count >= 2 && (i.DataType == (int)TensorProto.Types.DataType.Float || i.DataType == (int)TensorProto.Types.DataType.Float16))
                .Select(i => i.Name)
                .ToHashSet();
            var weights = new HashSet<string>();
            foreach (var node in graph.Node)
            {
                if (k_WeightInputs.TryGetValue(node.OpType, out var index) && index < node.Input.Count && floats.Contains(node.Input[index]))
                    weights.Add(node.Input[index]);
            }
            return weights;
        }

        /// <summary>
        /// Adds an input with the shape of the weight, which the optimizer cannot fold.
        /// </summary>
        public static Node AddStandIn(GraphModule gm, TensorProto weight)
        {
            var shape = weight.Dims.Select(d => checked((int)d)).ToArray();
            return gm.Input(weight.Name, DataType.Float, new DynamicTensorShape(shape.AsSpan()));
        }

        /// <summary>
        /// Loads the weight of a stand-in, quantizes it, and replaces the stand-in with the dequantized weight.
        /// </summary>
        /// <returns>The bytes of the float32 weight it loaded, which are garbage now.</returns>
        public static long Replace(GraphModule gm, Node standIn, Func<ConstantTensor> loadWeight, QuantizationType quantization, CPUBackend backend)
        {
            if (standIn.erased)
                return 0;
            if (standIn.users.Count == 0)
            {
                gm.graph.EraseNode(standIn);
                return 0;
            }

            var weight = loadWeight();
            if (quantization == QuantizationType.Float16)
            {
                var quantized = GraphPassUtil.AddConstant(gm, standIn, new ConstantTensor(weight.shape, DataType.Short, ToHalf(weight)));
                GraphPassUtil.ReplaceNode(standIn, "Cast", new Argument[] { quantized, DataType.Float });
            }
            else
            {
                var (data, scale, zeroPoint) = ToUint8(weight, backend);
                var quantized = GraphPassUtil.AddConstant(gm, standIn, new ConstantTensor(weight.shape, DataType.Byte, data));
                GraphPassUtil.ReplaceNode(standIn, "DequantizeUint8", new Argument[] { quantized, scale, zeroPoint });
            }
            return weight.array.Count;
        }

        /// <summary>
        /// Bytes of <paramref name="count"/> elements of <paramref name="size"/> bytes, padded to whole ints.
        /// </summary>
        static byte[] PaddedBytes(int count, int size) => new byte[(count * size + sizeof(int) - 1) / sizeof(int) * sizeof(int)];

        static unsafe byte[] ToHalf(ConstantTensor weight)
        {
            var count = weight.shape.length;
            var data = PaddedBytes(count, sizeof(short));
            fixed (byte* dst = data, src = weight.array.Array)
            {
                new BurstJobsQuantizeTensor.CastFloatToHalfJob
                {
                    src = (float*)(src + weight.array.Offset),
                    dst = (ushort*)dst
                }.Schedule(count, 32).Complete();
            }
            return data;
        }

        static unsafe (byte[] data, float scale, byte zeroPoint) ToUint8(ConstantTensor weight, CPUBackend backend)
        {
            using var x = weight.ToTensor() as Tensor<float>;
            using var min = new Tensor<float>(new TensorShape());
            using var max = new Tensor<float>(new TensorShape());
            backend.ReduceMin(x, min, null);
            backend.ReduceMax(x, max, null);
            min.CompleteAllPendingOperations();
            max.CompleteAllPendingOperations();
            var minValue = min.GetItem<float>(0);
            var maxValue = max.GetItem<float>(0);
            var scale = (Mathf.Max(0, maxValue) - Mathf.Min(0, minValue)) / 255f;
            var zeroPoint = (byte)Mathf.RoundToInt(Mathf.Clamp(-minValue / scale, 0, 255));

            var count = weight.shape.length;
            var data = PaddedBytes(count, sizeof(byte));
            fixed (byte* dst = data, src = weight.array.Array)
            {
                new BurstJobsQuantizeTensor.QuantizeUint8Job
                {
                    src = (float*)(src + weight.array.Offset),
                    dst = dst,
                    scale = scale,
                    zeroPoint = zeroPoint
                }.Schedule(count, 32).Complete();
            }
            return (data, scale, zeroPoint);
        }
    }
}
