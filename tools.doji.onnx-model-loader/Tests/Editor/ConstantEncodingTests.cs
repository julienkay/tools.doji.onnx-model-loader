using System;
using System.IO;
using System.Linq;
using Doji.AI;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;

namespace Doji.AI.ONNXModelLoader.Editor.Tests
{
    /// <summary>Initializer data in every encoding the loader reads gives the same constant.</summary>
    public class ConstantEncodingTests
    {
        const int DataTypeFloat16 = 10;
        const int DataTypeDouble = 11;
        static readonly float[] s_Values = { -2.5f, -1f, 0f, 0.5f, 1f, 1.5f, 2f, 4f, 8f };
        // Big enough that the weight fills most of the file, so the loader borrows it.
        static readonly float[] s_LargeValues = Enumerable.Range(0, 32 * 32).Select(i => i * 0.25f).ToArray();

        public enum Encoding { RawFloat, FloatData, RawFloat16, RawDouble, DoubleData, ExternalWithOffset }

        [Test]
        public void Load_InitializerEncoding_GivesSameValues([Values] Encoding encoding)
        {
            var directory = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
            Directory.CreateDirectory(directory);
            try
            {
                var modelPath = Path.Combine(directory, "model.onnx");
                File.WriteAllBytes(modelPath, BuildAddModel(Weight(encoding, directory, s_Values), 3));
                var model = OnnxModelLoader.Load(modelPath);

                using var worker = new Worker(model, BackendType.CPU);
                using var input = new Tensor<float>(new TensorShape(1, 1, 3, 3), new float[9]);
                worker.Schedule(input);
                var output = ((Tensor<float>)worker.PeekOutput()).DownloadToArray();

                Assert.That(output, Is.EqualTo(s_Values));
            }
            finally
            {
                Directory.Delete(directory, true);
            }
        }

        [Test]
        public void Load_RawFloatData_BorrowsFileBuffer()
        {
            var model = LoadTempModel(Weight(Encoding.RawFloat, null, s_LargeValues), 32);

            // A borrowed constant is a segment of the file buffer, after the bytes that precede raw_data.
            var constant = model.constants.Single();
            Assert.That(constant.array.Offset, Is.GreaterThan(0));
            Assert.That(constant.array.Count, Is.EqualTo(sizeof(float) * s_LargeValues.Length));
        }

        [Test]
        public void Load_ConvertedData_DoesNotKeepFileBuffer()
        {
            var model = LoadTempModel(Weight(Encoding.RawFloat16, null, s_LargeValues), 32);

            var constant = model.constants.Single();
            Assert.That(constant.array.Offset, Is.EqualTo(0));
            Assert.That(constant.array.Array.Length, Is.EqualTo(sizeof(float) * s_LargeValues.Length));
        }

        static Model LoadTempModel(ProtoWriter weight, int size)
        {
            var modelPath = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".onnx");
            try
            {
                File.WriteAllBytes(modelPath, BuildAddModel(weight, size));
                return OnnxModelLoader.Load(modelPath);
            }
            finally
            {
                File.Delete(modelPath);
            }
        }

        static ProtoWriter Weight(Encoding encoding, string directory, float[] values)
        {
            int size = (int)Math.Sqrt(values.Length);
            var tensor = new ProtoWriter().PackedInt64(1, 1, 1, size, size);
            switch (encoding)
            {
                case Encoding.RawFloat:
                    return tensor.Int64(2, ProtoWriter.DataTypeFloat).String(8, "W").Bytes(9, FloatBytes(values));
                case Encoding.FloatData:
                    // Packed fixed32 values, as float_data is written.
                    return tensor.Int64(2, ProtoWriter.DataTypeFloat).String(8, "W").Bytes(4, FloatBytes(values));
                case Encoding.RawFloat16:
                {
                    var bytes = values.SelectMany(v => BitConverter.GetBytes(Mathf.FloatToHalf(v))).ToArray();
                    return tensor.Int64(2, DataTypeFloat16).String(8, "W").Bytes(9, bytes);
                }
                case Encoding.RawDouble:
                    return tensor.Int64(2, DataTypeDouble).String(8, "W").Bytes(9, DoubleBytes(values));
                case Encoding.DoubleData:
                    return tensor.Int64(2, DataTypeDouble).String(8, "W").Bytes(10, DoubleBytes(values));
                case Encoding.ExternalWithOffset:
                {
                    // 12 bytes of other data before the tensor: only the offset entry finds it.
                    const int offset = 12;
                    File.WriteAllBytes(Path.Combine(directory, "weights.bin"), new byte[offset].Concat(FloatBytes(values)).ToArray());
                    return tensor
                        .Int64(2, ProtoWriter.DataTypeFloat)
                        .String(8, "W")
                        .Message(13, new ProtoWriter().String(1, "location").String(2, "weights.bin"))
                        .Message(13, new ProtoWriter().String(1, "offset").String(2, offset.ToString()))
                        .Message(13, new ProtoWriter().String(1, "length").String(2, (sizeof(float) * values.Length).ToString()))
                        .Int64(14, 1); // data_location = EXTERNAL
                }
                default:
                    throw new ArgumentOutOfRangeException(nameof(encoding));
            }
        }

        static byte[] FloatBytes(float[] values) => values.SelectMany(BitConverter.GetBytes).ToArray();
        static byte[] DoubleBytes(float[] values) => values.SelectMany(v => BitConverter.GetBytes((double)v)).ToArray();

        /// <summary>Y = Add(X, W) with X and W of shape 1x1xsizexsize.</summary>
        static byte[] BuildAddModel(ProtoWriter weight, int size)
        {
            var node = new ProtoWriter().String(1, "X").String(1, "W").String(2, "Y").String(4, "Add");
            var graph = new ProtoWriter()
                .Message(1, node)
                .String(2, "add")
                .Message(5, weight)
                .Message(11, ProtoWriter.FloatTensorValueInfo("X", 1, 1, size, size))
                .Message(12, ProtoWriter.FloatTensorValueInfo("Y", null));
            return new ProtoWriter()
                .Int64(1, 8)
                .Message(8, new ProtoWriter().String(1, "").Int64(2, 13))
                .Message(7, graph)
                .ToArray();
        }
    }
}
