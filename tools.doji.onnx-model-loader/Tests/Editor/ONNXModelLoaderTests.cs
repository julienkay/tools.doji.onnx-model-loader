using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Doji.AI;
using Doji.AI.InferenceEngine;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.TestTools;

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
        public void Load_MissingExternalWeights_ThrowsFileNotFound()
        {
            var weight = ProtoWriter.ExternalFloatTensor("W", "missing-weights.bin", 1, 1, 1, 1);
            var exception = LoadTempModelThrows<FileNotFoundException>(ConvNode(), weight);

            Assert.That(exception.Message, Does.Contain("missing-weights.bin"));
        }

        [Test]
        public void Load_UnsupportedOperator_ThrowsWithOperatorAndModelName()
        {
            var node = new ProtoWriter().String(1, "X").String(2, "Y").String(4, "NotAnOnnxOperator");
            var exception = LoadTempModelThrows<OnnxImportException>(node);

            Assert.That(exception.Message, Does.Contain("NotAnOnnxOperator"));
            Assert.That(exception.Message, Does.Contain(".onnx"));
            Assert.That(exception.Message, Does.Not.Contain("<b>"));
        }

        /// <summary>
        /// A node of the ai.doji domain becomes the Sentis layer of the same name, which tools.doji.sentis-utils
        /// adds. Without that package, the load fails and names it.
        /// </summary>
        [Test]
        public void Load_DojiOperator_BecomesTheLayerOfTheSameName()
        {
            string[] inputs = { "q", "k", "v", "g", "beta", "state" };
            var node = new ProtoWriter();
            foreach (var input in inputs)
                node.String(1, input);
            node.String(2, "o").String(2, "final").String(4, "GatedDeltaRule").String(7, ONNXModelConverter.DojiDomain)
                .Message(5, ProtoWriter.FloatAttribute("scale", 0.5f));
            var graph = new ProtoWriter().Message(1, node).String(2, "graph");
            long[][] shapes = { new long[] { 1, 2, 3, 4 }, new long[] { 1, 2, 3, 4 }, new long[] { 1, 2, 3, 5 }, new long[] { 1, 2, 3 },
                new long[] { 1, 2, 3 }, new long[] { 1, 3, 4, 5 } };
            for (var i = 0; i < inputs.Length; i++)
                graph.Message(11, ProtoWriter.FloatTensorValueInfo(inputs[i], shapes[i]));
            graph.Message(12, ProtoWriter.FloatTensorValueInfo("o", null)).Message(12, ProtoWriter.FloatTensorValueInfo("final", null));
            var bytes = new ProtoWriter()
                .Int64(1, 8)
                .Message(8, new ProtoWriter().String(1, "").Int64(2, 17))
                .Message(8, new ProtoWriter().String(1, ONNXModelConverter.DojiDomain).Int64(2, 1))
                .Message(7, graph)
                .ToArray();
            var modelPath = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".onnx");
            try
            {
                File.WriteAllBytes(modelPath, bytes);
                bool hasLayer = typeof(Model).Assembly.GetType("Unity.InferenceEngine.Layers.GatedDeltaRule") != null;
                if (!hasLayer)
                {
                    var exception = Assert.Throws<OnnxImportException>(() => OnnxModelLoader.Load(modelPath));
                    Assert.That(exception.Message, Does.Contain("tools.doji.sentis-utils"));
                    return;
                }
                var model = OnnxModelLoader.Load(modelPath);
                Assert.That(model.layers.Select(l => l.opName), Is.EqualTo(new[] { "GatedDeltaRule" }));
                Assert.That(model.outputs.Count, Is.EqualTo(2));
            }
            finally
            {
                File.Delete(modelPath);
            }
        }

        [Test]
        public void Load_ConversionError_Throws()
        {
            // Constant needs one of its value attributes; the converter records an error and goes on.
            var node = new ProtoWriter().String(2, "Y").String(4, "Constant");
            var exception = LoadTempModelThrows<OnnxImportException>(node);

            Assert.That(exception.Message, Does.Contain("Constant"));
        }

        [Test]
        public void OnnxImportException_IsPublic()
        {
            Assert.That(typeof(OnnxImportException).IsPublic, Is.True);
        }

        [Test]
        public void Load_UnsupportedAttributeValue_LogsWarning()
        {
            var node = ConvNode().Message(5, ProtoWriter.StringAttribute("auto_pad", "NOT_A_MODE"));
            var weight = new ProtoWriter()
                .PackedInt64(1, 1, 1, 1, 1)
                .Int64(2, ProtoWriter.DataTypeFloat)
                .String(8, "W")
                .Bytes(9, new ProtoWriter().RawFloat(2f).ToArray());

            LogAssert.Expect(LogType.Warning, new Regex(@"\.onnx: .*auto_pad `NOT_A_MODE`"));
            var modelPath = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".onnx");
            try
            {
                File.WriteAllBytes(modelPath, BuildSingleNodeModel(node, weight));
                Assert.That(OnnxModelLoader.Load(modelPath), Is.Not.Null);
            }
            finally
            {
                File.Delete(modelPath);
            }
        }

        [TestCase(QuantizationType.Uint8, "DequantizeUint8", DataType.Byte, 0.03f)]
        [TestCase(QuantizationType.Float16, "Cast", DataType.Short, 0f)]
        public void Load_WithQuantization_QuantizesWeights(QuantizationType quantization, string dequantizeOp, DataType weightType, float tolerance)
        {
            // Y = MatMul(X, W) with W 3x2. X is all ones, so Y has the column sums of W in each row.
            var node = new ProtoWriter().String(1, "X").String(1, "W").String(2, "Y").String(4, "MatMul");
            var values = new ProtoWriter();
            foreach (var value in new[] { 1f, -2f, 0.5f, 3f, -1f, 2f })
                values.RawFloat(value);
            var weight = new ProtoWriter()
                .PackedInt64(1, 3, 2)
                .Int64(2, ProtoWriter.DataTypeFloat)
                .String(8, "W")
                .Bytes(9, values.ToArray());

            var modelPath = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".onnx");
            try
            {
                File.WriteAllBytes(modelPath, BuildSingleNodeModel(node, weight));
                var model = OnnxModelLoader.Load(modelPath, quantization);

                Assert.That(model.inputs, Has.Count.EqualTo(1), "The stand-in of the weight is no input.");
                Assert.That(model.layers.Select(layer => layer.opName), Does.Contain(dequantizeOp));
                Assert.That(model.constants.Select(constant => constant.dataType), Does.Contain(weightType));

                using var worker = new Worker(model, BackendType.CPU);
                using var input = new Tensor<float>(new TensorShape(1, 1, 3, 3), Enumerable.Repeat(1f, 9).ToArray());
                worker.Schedule(input);
                var output = ((Tensor<float>)worker.PeekOutput()).DownloadToArray();
                Assert.That(output, Is.EqualTo(new[] { 0.5f, 3f, 0.5f, 3f, 0.5f, 3f }).Within(tolerance));
            }
            finally
            {
                File.Delete(modelPath);
            }
        }

        // A weight of 16 elements or fewer loads with the model; a larger one loads, and is quantized, while the file
        // is written. Either way the saved model computes what the loaded model computes.
        [Test]
        public void Save_StreamsWeights_SameOutputAsLoad(
            [Values(2, 8)] int columns,
            [Values(null, QuantizationType.Float16, QuantizationType.Uint8)] QuantizationType? quantization)
        {
            // Y = MatMul(X, W) with W 3 x columns.
            var node = new ProtoWriter().String(1, "X").String(1, "W").String(2, "Y").String(4, "MatMul");
            var weight = FloatWeight("W", 3, columns, i => i * 0.25f - 1f);

            var modelPath = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".onnx");
            var sentisPath = Path.ChangeExtension(modelPath, ".sentis");
            try
            {
                File.WriteAllBytes(modelPath, BuildSingleNodeModel(node, weight));
                OnnxModelLoader.Save(modelPath, sentisPath, quantization);
                var saved = ModelLoader.Load(sentisPath);
                var loaded = OnnxModelLoader.Load(modelPath, quantization);
                Assert.That(saved.inputs, Has.Count.EqualTo(1), "The stand-in of the weight is no input.");
                Assert.That(ConstantTypes(saved), Is.EqualTo(ConstantTypes(loaded)));

                var input = Enumerable.Range(0, 9).Select(i => i - 4f).ToArray();
                Assert.That(Run(saved, input), Is.EqualTo(Run(loaded, input)));
            }
            finally
            {
                File.Delete(modelPath);
                File.Delete(sentisPath);
            }
        }

        // Two weights with their own uint8 scale, one of them used twice: each streamed weight gets its own data.
        [Test]
        public void Save_SeveralStreamedWeights_SameOutputAsLoad(
            [Values(null, QuantizationType.Float16, QuantizationType.Uint8)] QuantizationType? quantization)
        {
            // Y = MatMul(Add(MatMul(X, W1), MatMul(X, W1)), W2) with W1 3 x 8 and W2 8 x 3.
            var nodes = new[]
            {
                new ProtoWriter().String(1, "X").String(1, "W1").String(2, "A").String(4, "MatMul"),
                new ProtoWriter().String(1, "X").String(1, "W1").String(2, "B").String(4, "MatMul"),
                new ProtoWriter().String(1, "A").String(1, "B").String(2, "C").String(4, "Add"),
                new ProtoWriter().String(1, "C").String(1, "W2").String(2, "Y").String(4, "MatMul")
            };
            var w1 = FloatWeight("W1", 3, 8, i => i * 0.5f - 3f);
            var w2 = FloatWeight("W2", 8, 3, i => 0.1f - i * 0.01f);

            var modelPath = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".onnx");
            var sentisPath = Path.ChangeExtension(modelPath, ".sentis");
            try
            {
                File.WriteAllBytes(modelPath, BuildModel(nodes, w1, w2));
                OnnxModelLoader.Save(modelPath, sentisPath, quantization);
                var saved = ModelLoader.Load(sentisPath);
                var loaded = OnnxModelLoader.Load(modelPath, quantization);
                Assert.That(saved.inputs, Has.Count.EqualTo(1));
                Assert.That(ConstantTypes(saved), Is.EqualTo(ConstantTypes(loaded)));

                var input = Enumerable.Range(0, 9).Select(i => i * 0.5f - 2f).ToArray();
                Assert.That(Run(saved, input), Is.EqualTo(Run(loaded, input)));
            }
            finally
            {
                File.Delete(modelPath);
                File.Delete(sentisPath);
            }
        }

        // Before Save writes the file, a weight of more than 16 elements has no data yet: its constant has only the
        // type and size of the weight in the file, quantized or not. So Save never holds all weights in memory.
        [TestCase(null, DataType.Float, 3 * 8 * 4)]
        [TestCase(QuantizationType.Float16, DataType.Short, 3 * 8 * 2)]
        [TestCase(QuantizationType.Uint8, DataType.Byte, 3 * 8)]
        // The quantized data is padded to whole ints.
        [TestCase(QuantizationType.Float16, DataType.Short, 3 * 7 * 2 + 2, 7)]
        [TestCase(QuantizationType.Uint8, DataType.Byte, 3 * 7 + 3, 7)]
        public void Save_WeightHasNoDataBeforeItIsWritten(QuantizationType? quantization, DataType weightType, int lengthBytes, int columns = 8)
        {
            var node = new ProtoWriter().String(1, "X").String(1, "W").String(2, "Y").String(4, "MatMul");
            var weight = FloatWeight("W", 3, columns, i => i * 0.3f - 2f);

            var modelPath = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".onnx");
            var sentisPath = Path.ChangeExtension(modelPath, ".sentis");
            try
            {
                File.WriteAllBytes(modelPath, BuildSingleNodeModel(node, weight));
                Constant[] streamed = null;
                OnnxModelLoader.Save(modelPath, sentisPath, quantization, model =>
                    streamed = model.constants.Where(c => c.array.Array == null && c.lengthBytes > 0).ToArray());

                Assert.That(streamed, Has.Length.EqualTo(1));
                Assert.That(streamed[0].dataType, Is.EqualTo(weightType));
                Assert.That(streamed[0].lengthBytes, Is.EqualTo(lengthBytes));
                Assert.That(streamed[0].shape, Is.EqualTo(new TensorShape(3, columns)));

                var input = Enumerable.Range(0, 9).Select(i => i - 4f).ToArray();
                Assert.That(Run(ModelLoader.Load(sentisPath), input), Is.EqualTo(Run(OnnxModelLoader.Load(modelPath, quantization), input)));
            }
            finally
            {
                File.Delete(modelPath);
                File.Delete(sentisPath);
            }
        }

        static ProtoWriter FloatWeight(string name, int rows, int columns, System.Func<int, float> value)
        {
            var values = new ProtoWriter();
            for (var i = 0; i < rows * columns; i++)
                values.RawFloat(value(i));
            return new ProtoWriter()
                .PackedInt64(1, rows, columns)
                .Int64(2, ProtoWriter.DataTypeFloat)
                .String(8, name)
                .Bytes(9, values.ToArray());
        }

        static DataType[] ConstantTypes(Model model) =>
            model.constants.Select(c => c.dataType).OrderBy(t => t).ToArray();

        [Test]
        public void ChunkHeader_MatchesFlatBufferBuilder([NUnit.Framework.Range(0, 8)] int length)
        {
            var data = Enumerable.Range(1, length).Select(i => (byte)i).ToArray();
            var builder = new Unity.InferenceEngine.Google.FlatBuffers.FlatBufferBuilder(32 + length);
            var storage = SentisFlatBuffer.Buffer.CreateStorageVectorBlock(builder, data);
            SentisFlatBuffer.Buffer.StartBuffer(builder);
            SentisFlatBuffer.Buffer.AddStorage(builder, storage);
            builder.FinishSizePrefixed(SentisFlatBuffer.Buffer.EndBuffer(builder).Value);

            var streamed = StreamedModelWriter.ChunkHeader(length)
                .Concat(data)
                .Concat(new byte[StreamedModelWriter.ChunkPadding(length)]);
            Assert.That(streamed.ToArray(), Is.EqualTo(builder.DataBuffer.ToSizedArray()));
        }

        static float[] Run(Model model, float[] input)
        {
            using var worker = new Worker(model, BackendType.CPU);
            using var tensor = new Tensor<float>(new TensorShape(1, 1, 3, 3), input);
            worker.Schedule(tensor);
            return ((Tensor<float>)worker.PeekOutput()).DownloadToArray();
        }

        static ProtoWriter ConvNode()
        {
            return new ProtoWriter()
                .String(1, "X")
                .String(1, "W")
                .String(2, "Y")
                .String(4, "Conv")
                .Message(5, ProtoWriter.IntsAttribute("kernel_shape", 1, 1));
        }

        static T LoadTempModelThrows<T>(ProtoWriter node, params ProtoWriter[] initializers) where T : System.Exception
        {
            var modelPath = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".onnx");
            try
            {
                File.WriteAllBytes(modelPath, BuildSingleNodeModel(node, initializers));
                return Assert.Throws<T>(() => OnnxModelLoader.Load(modelPath));
            }
            finally
            {
                File.Delete(modelPath);
            }
        }

        /// <summary>A model with input X (1x1x3x3), output Y and one node.</summary>
        static byte[] BuildSingleNodeModel(ProtoWriter node, params ProtoWriter[] initializers) =>
            BuildModel(new[] { node }, initializers);

        /// <summary>A model with input X (1x1x3x3), output Y and the nodes in order.</summary>
        static byte[] BuildModel(ProtoWriter[] nodes, params ProtoWriter[] initializers)
        {
            var graph = new ProtoWriter();
            foreach (var node in nodes)
                graph.Message(1, node);
            graph.String(2, "graph");
            foreach (var initializer in initializers)
                graph.Message(5, initializer);
            graph
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
