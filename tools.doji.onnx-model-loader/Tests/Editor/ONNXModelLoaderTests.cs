using System.IO;
using System.Linq;
using System.Text;
using Doji.AI;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace Doji.AI.ONNXModelLoader.Editor.Tests
{
    public class ONNXModelLoaderTests
    {
        const string ModelsDirectory = "Packages/tools.doji.onnx-model-loader/Tests/Editor/Models";

        [TestCase("Adder.onnx", 2, 1, 1, "Add")]
        [TestCase("Model.onnx", 1, 1, 12, "Relu")]
        [TestCase("MultipleInputMultipleOuput.onnx", 2, 2, 2, "Relu", "Add")]
        [TestCase("ReluModel.onnx", 1, 1, 1, "Relu")]
        [TestCase("SingleInputSingleOuput.onnx", 1, 1, 1, "Relu")]
        // Sentis optimizes Sub(input, constant 1) into the equivalent ScalarMad layer.
        [TestCase("SubtractConstantOne.onnx", 1, 1, 1, "ScalarMad")]
        public void Convert_LoadsModelAndCreatesExpectedLayer(
            string fileName,
            int expectedInputCount,
            int expectedOutputCount,
            int expectedLayerCount,
            params string[] expectedOpNames)
        {
            var model = LoadModel(fileName);

            Assert.That(model.inputs, Has.Count.EqualTo(expectedInputCount));
            Assert.That(model.outputs, Has.Count.EqualTo(expectedOutputCount));
            Assert.That(model.layers, Has.Count.EqualTo(expectedLayerCount));

            var opNames = model.layers.Select(layer => layer.opName).ToArray();
            foreach (var expectedOpName in expectedOpNames)
                Assert.That(opNames, Does.Contain(expectedOpName));
        }

        [Test]
        public void Convert_WithPathOverload_LoadsOnnxModel()
        {
            var modelPath = GetModelPath("ReluModel.onnx");

            Assert.DoesNotThrow(() => OnnxModelLoader.Load(modelPath));
        }

        [Test]
        public void Convert_ConvWithPads_AppliesPadsAttribute()
        {
            var modelPath = Path.GetTempFileName();
            try
            {
                File.WriteAllBytes(modelPath, BuildPaddedConvModel());
                var model = OnnxModelLoader.Load(modelPath);

                using var worker = new Worker(model, BackendType.CPU);
                using var input = new Tensor<float>(new TensorShape(1, 1, 3, 3), Enumerable.Repeat(1f, 9).ToArray());
                worker.Schedule(input);
                var output = (Tensor<float>)worker.PeekOutput();

                // pads=[1,1,1,1] grows the 3x3 input to 5x5. Without the attribute it stays 3x3.
                Assert.That(output.shape, Is.EqualTo(new TensorShape(1, 1, 5, 5)));
                var values = output.DownloadToArray();
                Assert.That(values[0], Is.EqualTo(0f), "padded corner");
                Assert.That(values[12], Is.EqualTo(2f), "center: input 1 times weight 2");
            }
            finally
            {
                File.Delete(modelPath);
            }
        }

        /// <summary>Y = Conv(X, W) with a 1x1 kernel of value 2 and pads=[1,1,1,1].</summary>
        static byte[] BuildPaddedConvModel()
        {
            var weight = new ProtoWriter()
                .PackedInt64(1, 1, 1, 1, 1)
                .Int64(2, ProtoWriter.DataTypeFloat)
                .String(8, "W")
                .Bytes(9, new ProtoWriter().RawFloat(2f).ToArray());
            var conv = new ProtoWriter()
                .String(1, "X")
                .String(1, "W")
                .String(2, "Y")
                .String(4, "Conv")
                .Message(5, ProtoWriter.IntsAttribute("kernel_shape", 1, 1))
                .Message(5, ProtoWriter.IntsAttribute("pads", 1, 1, 1, 1));
            var graph = new ProtoWriter()
                .Message(1, conv)
                .String(2, "padded_conv")
                .Message(5, weight)
                .Message(11, ProtoWriter.FloatTensorValueInfo("X", 1, 1, 3, 3))
                .Message(12, ProtoWriter.FloatTensorValueInfo("Y", null));
            return new ProtoWriter()
                .Int64(1, 8)
                .Message(8, new ProtoWriter().String(1, "").Int64(2, 13))
                .Message(7, graph)
                .ToArray();
        }

        [Test]
        public void LoadMetadata_SkipsGraphPayload()
        {
            var modelPath = Path.GetTempFileName();
            try
            {
                WriteMetadataOnlyOnnx(modelPath, graphPayloadSize: 1024 * 1024);

                var metadata = OnnxModelLoader.LoadMetadata(modelPath);

                Assert.That(metadata.IRVersion, Is.EqualTo(10));
                Assert.That(metadata.ProducerName, Is.EqualTo("metadata-test"));
                Assert.That(metadata.ProducerVersion, Is.EqualTo("1.2.3"));
                Assert.That(metadata.Domain, Is.EqualTo("tools.doji.test"));
                Assert.That(metadata.ModelVersion, Is.EqualTo(42));
                Assert.That(metadata.DocString, Is.EqualTo("metadata without graph load"));
                Assert.That(metadata.MetadataProps, Contains.Key("purpose"));
                Assert.That(metadata.MetadataProps["purpose"], Is.EqualTo("partial-load"));
            }
            finally
            {
                File.Delete(modelPath);
            }
        }

        [Test]
        public void LoadModelMetadata_WithStream_ReturnsMetadataOnlyModelProto()
        {
            using var stream = new MemoryStream();
            WriteMetadataOnlyOnnx(stream, graphPayloadSize: 1024 * 1024);
            stream.Position = 0;

            var modelProto = OnnxModelLoader.LoadModelMetadata(stream);

            Assert.That(modelProto.Graph, Is.Null);
            Assert.That(modelProto.IrVersion, Is.EqualTo(10));
            Assert.That(modelProto.ProducerName, Is.EqualTo("metadata-test"));
            Assert.That(modelProto.ProducerVersion, Is.EqualTo("1.2.3"));
            Assert.That(modelProto.Domain, Is.EqualTo("tools.doji.test"));
            Assert.That(modelProto.ModelVersion, Is.EqualTo(42));
            Assert.That(modelProto.DocString, Is.EqualTo("metadata without graph load"));
            Assert.That(modelProto.MetadataProps, Has.Count.EqualTo(1));
            Assert.That(modelProto.MetadataProps[0].Key, Is.EqualTo("purpose"));
            Assert.That(modelProto.MetadataProps[0].Value, Is.EqualTo("partial-load"));
        }

        static Model LoadModel(string fileName)
        {
            var modelPath = GetModelPath(fileName);

            Model model = null;
            Assert.DoesNotThrow(() => model = OnnxModelLoader.Load(modelPath));
            Assert.That(model, Is.Not.Null);

            return model;
        }

        static string GetModelPath(string fileName)
        {
            var modelPath = Path.Combine(ModelsDirectory, fileName);
            Assert.That(File.Exists(modelPath), Is.True, $"Missing test ONNX model at {modelPath}");
            return modelPath;
        }

        static void WriteMetadataOnlyOnnx(string path, int graphPayloadSize)
        {
            using var stream = File.Create(path);
            WriteMetadataOnlyOnnx(stream, graphPayloadSize);
        }

        static void WriteMetadataOnlyOnnx(Stream stream, int graphPayloadSize)
        {
            WriteVarintField(stream, 1, 10);
            WriteStringField(stream, 2, "metadata-test");
            WriteStringField(stream, 3, "1.2.3");
            WriteStringField(stream, 4, "tools.doji.test");
            WriteVarintField(stream, 5, 42);
            WriteStringField(stream, 6, "metadata without graph load");

            WriteTag(stream, 7, 2);
            WriteVarint(stream, (ulong)graphPayloadSize);
            var graphPayload = new byte[graphPayloadSize];
            for (var i = 0; i < graphPayload.Length; i++)
                graphPayload[i] = 0xff;
            stream.Write(graphPayload, 0, graphPayload.Length);

            using var metadataEntry = new MemoryStream();
            WriteStringField(metadataEntry, 1, "purpose");
            WriteStringField(metadataEntry, 2, "partial-load");
            var metadataBytes = metadataEntry.ToArray();

            WriteTag(stream, 14, 2);
            WriteVarint(stream, (ulong)metadataBytes.Length);
            stream.Write(metadataBytes, 0, metadataBytes.Length);
        }

        static void WriteStringField(Stream stream, int field, string value)
        {
            var bytes = Encoding.UTF8.GetBytes(value);
            WriteTag(stream, field, 2);
            WriteVarint(stream, (ulong)bytes.Length);
            stream.Write(bytes, 0, bytes.Length);
        }

        static void WriteVarintField(Stream stream, int field, ulong value)
        {
            WriteTag(stream, field, 0);
            WriteVarint(stream, value);
        }

        static void WriteTag(Stream stream, int field, int wireType)
        {
            WriteVarint(stream, (ulong)((field << 3) | wireType));
        }

        static void WriteVarint(Stream stream, ulong value)
        {
            while (value >= 0x80)
            {
                stream.WriteByte((byte)(value | 0x80));
                value >>= 7;
            }

            stream.WriteByte((byte)value);
        }
    }
}
