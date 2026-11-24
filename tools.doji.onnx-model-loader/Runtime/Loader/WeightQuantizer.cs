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
        /// Loads the weight of a stand-in, quantizes it, and replaces the stand-in with the dequantized weight. The
        /// weight is dequantized right before its first use, so only the float32 weights in use are on the device.
        /// </summary>
        /// <param name="order">The position of each node in the graph.</param>
        /// <param name="deferred">When not null, a weight of more than 16 elements gets no data: the stand-in is
        /// replaced with an input of the quantized type, and <paramref name="deferred"/> gets the name of that input
        /// and the function that quantizes the float32 weight. Saving uses it to quantize while the file is written.
        /// </param>
        /// <returns>The bytes of the float32 weight it loaded, which are garbage now.</returns>
        public static long Replace(GraphModule gm, Node standIn, IReadOnlyDictionary<Node, int> order, Func<ConstantTensor> loadWeight, QuantizationType quantization, CPUBackend backend, IDictionary<string, Func<ConstantTensor, byte[]>> deferred = null)
        {
            if (standIn.erased)
                return 0;
            if (standIn.users.Count == 0)
            {
                gm.graph.EraseNode(standIn);
                return 0;
            }

            Node firstUser = null;
            foreach (var user in standIn.users)
            {
                if (firstUser == null || order[user] < order[firstUser])
                    firstUser = user;
            }

            // Shape inference reads the values of a constant with 16 elements or fewer, so a small weight gets its data now.
            var shape = standIn.partialTensor.shape;
            var defer = deferred != null && shape.ToTensorShape().length > TensorShape.maxRank * 2;
            long garbage = 0;
            Node quantized;
            string op;
            Argument[] args;
            if (quantization == QuantizationType.Float16)
            {
                if (defer)
                {
                    using (gm.graph.InsertingBefore(firstUser))
                        quantized = gm.Input(standIn.name + "_float16", DataType.Short, shape);
                    deferred[quantized.name] = ToHalf;
                }
                else
                {
                    var weight = loadWeight();
                    quantized = GraphPassUtil.AddConstant(gm, standIn, new ConstantTensor(weight.shape, DataType.Short, ToHalf(weight)));
                    garbage = weight.array.Count;
                }
                op = "Cast";
                args = new Argument[] { quantized, DataType.Float };
            }
            else
            {
                // The scale and zero point are layer arguments, so they come from the weight now, even when its data is deferred.
                var weight = loadWeight();
                var (scale, zeroPoint) = Uint8Range(weight, backend);
                garbage = weight.array.Count;
                if (defer)
                {
                    using (gm.graph.InsertingBefore(firstUser))
                        quantized = gm.Input(standIn.name + "_uint8", DataType.Byte, shape);
                    deferred[quantized.name] = w => ToUint8(w, scale, zeroPoint);
                }
                else
                {
                    quantized = GraphPassUtil.AddConstant(gm, standIn, new ConstantTensor(weight.shape, DataType.Byte, ToUint8(weight, scale, zeroPoint)));
                }
                op = "DequantizeUint8";
                args = new Argument[] { quantized, scale, zeroPoint };
            }

            Node dequantized;
            using (gm.graph.InsertingBefore(firstUser))
                dequantized = gm.graph.CallFunction(op, args);
            standIn.ReplaceAllUsesWith(dequantized);
            gm.graph.EraseNode(standIn);
            return garbage;
        }

        /// <summary>The bytes of a weight of <paramref name="count"/> elements quantized to <paramref name="dataType"/>, padded to whole ints.</summary>
        public static int QuantizedLengthBytes(int count, DataType dataType) =>
            PaddedBytes(count, dataType == DataType.Short ? sizeof(short) : sizeof(byte)).Length;

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

        static (float scale, byte zeroPoint) Uint8Range(ConstantTensor weight, CPUBackend backend)
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
            return (scale, zeroPoint);
        }

        static unsafe byte[] ToUint8(ConstantTensor weight, float scale, byte zeroPoint)
        {
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
            return data;
        }
    }
}
